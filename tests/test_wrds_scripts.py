"""Tests for the WRDS download scripts in scripts/ (pull_spx_*.py), without a WRDS account.

WRDS is a PostgreSQL database. Here a small DuckDB database stands in for it: same library name (optionm_all), same
table and column names as OptionMetrics, about a year of made-up SPX quotes. A fake `wrds` module hands the scripts
a connection to it, so they run their real SQL unchanged. The last test takes the downloaded files through
prepare_wrds.py and hedging.real, the same path the real data takes in notebook 04.

Run with `uv run pytest`.
"""

import importlib.util
import sys
import types
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from hedging import bs, hedge, real

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPX, OTHER = 108105, 500001  # SPX's secid in OptionMetrics, and another underlying the queries must leave out
YEARS = [2019, 2020]
DAYS = pd.bdate_range("2019-04-01", "2020-04-30")  # trading days of the fake data
SIGMA = 0.2  # every fake option is priced at 20% volatility


def _fake_optionmetrics():
    """Builds the fake OptionMetrics library in an in-memory DuckDB database and returns the connection.

    Tables, as on WRDS: securd (security names), secprd (daily prices), idxdvd (dividend yield), zerocd (zero
    curve), and per year opprcdYYYY (option quotes), vsurfdYYYY (volatility surface), fwdprdYYYY (forwards).
    """
    rng = np.random.default_rng(0)
    n = len(DAYS)
    close = 2900 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    prices = pd.DataFrame({"secid": SPX, "date": DAYS.date, "low": close * 0.99, "high": close * 1.01,
                           "close": close, "volume": 0.0, "return": np.r_[0, np.diff(close) / close[:-1]],
                           "cfadj": 1.0, "open": close * (1 + rng.normal(0, 0.002, n)), "cfret": 1.0, "shrout": 0.0})
    prices = pd.concat([prices, prices.assign(secid=OTHER, close=50.0, open=50.0)])

    # Option quotes: weekly Friday expiries up to 100 days out (the scripts keep 0-70 or 20-50) and strikes from
    # 1500 to 4800 (the scripts keep those within 10%, 20% or 30% of the close).
    fridays = pd.date_range("2019-04-05", "2020-08-28", freq="W-FRI")
    strikes = np.arange(1500, 4801, 100.0)
    d, e, k = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(len(fridays)), np.arange(len(strikes)),
                                              indexing="ij"))
    dte = (fridays[e] - DAYS[d]).days.to_numpy()
    keep = (dte >= 0) & (dte <= 100)
    d, e, k, dte = d[keep], e[keep], k[keep], dte[keep]
    quotes = []
    for cp, is_call in (("C", True), ("P", False)):
        mid = bs.price(close[d], strikes[k], np.maximum(dte, 0.5) / 365, SIGMA, is_call)
        bid = np.where(mid < 0.05, 0.0, np.round(mid * 0.98, 2))  # far out of the money: no bid
        quotes.append(pd.DataFrame({
            "secid": SPX, "date": DAYS[d].date, "exdate": fridays[e].date, "cp_flag": cp,
            "strike_price": (strikes[k] * 1000).astype(int), "best_bid": bid, "best_offer": np.round(mid * 1.02 + 0.1, 2),
            "volume": 0.0, "open_interest": 0.0, "impl_volatility": np.where(bid > 0, SIGMA, np.nan),
            "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "optionid": 1.0e8 + k,
            "symbol": "SPXW", "am_settlement": 0, "expiry_indicator": "w", "root": None, "suffix": None,
            "contract_size": 100.0, "ss_flag": "0"}))
    quotes = pd.concat(quotes, ignore_index=True)
    quotes = pd.concat([quotes, quotes.iloc[::50].assign(secid=OTHER)], ignore_index=True)

    # Volatility surface: calls at deltas 10..90, puts at -90..-10, five maturities; impl_strike from the delta.
    surface = []
    for days in (10, 30, 60, 91, 182):
        sd = SIGMA * np.sqrt(days / 365)
        for delta in range(10, 91, 10):
            for cp, signed, n_d1 in (("C", delta, norm.ppf(delta / 100)), ("P", -delta, norm.ppf(1 - delta / 100))):
                surface.append(pd.DataFrame({"secid": SPX, "date": DAYS.date, "days": float(days), "delta": float(signed),
                                             "impl_volatility": SIGMA, "impl_strike": close * np.exp(0.5 * sd * sd - sd * n_d1),
                                             "impl_premium": 1.0, "dispersion": 0.01, "cp_flag": cp}))
    surface = pd.concat(surface, ignore_index=True)

    forwards = pd.DataFrame({"secid": SPX, "date": DAYS.date, "expiration": fridays[fridays.searchsorted(DAYS) + 1]
                             .strftime("%Y-%m-%d"), "amsettlement": 0.0, "forwardprice": close * 1.001})

    con = duckdb.connect()
    con.execute("create schema optionm_all")

    def table(name, df, dates=("date",)):
        """Creates optionm_all.<name> from df, with real DATE columns (Postgres subtracts them to whole days)."""
        con.register("df", df)
        cols = ", ".join(f'cast("{c}" as date) as "{c}"' if c in dates else f'"{c}"' for c in df.columns)
        con.execute(f"create table optionm_all.{name} as select {cols} from df")
        con.unregister("df")

    table("securd", pd.DataFrame({"secid": [SPX, OTHER], "ticker": ["SPX", "XYZ"],
                                  "issuer": ["S&P 500 INDEX", "OTHER"], "issue_type": ["A", "0"]}), dates=())
    table("secprd", prices)
    table("idxdvd", pd.DataFrame({"secid": SPX, "date": DAYS.date, "rate": 1.8}))
    table("zerocd", pd.DataFrame([(day, n, 2.0) for day in DAYS.date for n in (7.0, 30.0, 91.0, 365.0)],
                                 columns=["date", "days", "rate"]))
    for y in YEARS:
        table(f"opprcd{y}", quotes[pd.DatetimeIndex(quotes.date).year == y], dates=("date", "exdate"))
        table(f"vsurfd{y}", surface[pd.DatetimeIndex(surface.date).year == y])
        table(f"fwdprd{y}", forwards[pd.DatetimeIndex(forwards.date).year == y])
    return con


class FakeConnection:
    """The parts of wrds.Connection the scripts use, answered from the DuckDB database."""

    def __init__(self, con):
        self.con = con.cursor()

    def list_libraries(self):
        return [r[0] for r in self.con.execute("select schema_name from information_schema.schemata").fetchall()]

    def list_tables(self, library):
        return [r[0] for r in self.con.execute(
            "select table_name from information_schema.tables where table_schema = ?", [library]).fetchall()]

    def describe_table(self, library, table):
        return self.con.execute("select column_name as name, data_type as type from information_schema.columns "
                                "where table_schema = ? and table_name = ?", [library, table]).df()

    def raw_sql(self, sql, date_cols=None):
        df = self.con.execute(sql).df()
        for c in date_cols or []:
            df[c] = pd.to_datetime(df[c])
        return df

    def close(self):
        self.con.close()


@pytest.fixture(scope="module")
def fake_db():
    """One fake OptionMetrics database for the whole file (it takes a few seconds to build)."""
    return _fake_optionmetrics()


@pytest.fixture
def run(fake_db, tmp_path, monkeypatch):
    """Loads the four scripts with their output folders moved into tmp_path and `wrds` replaced by the fake.

    Returns (scripts, raw): scripts a dict {short name: module}, raw the stand-in for data/raw.
    """
    monkeypatch.setitem(sys.modules, "wrds", types.SimpleNamespace(Connection=lambda: FakeConnection(fake_db)))
    monkeypatch.setattr("shutil.which", lambda *_: None)  # don't start caffeinate (it keeps a Mac awake)
    raw = tmp_path / "raw"
    scripts = {}
    for short, name in [("wrds", "pull_spx_wrds"), ("forwards", "pull_spx_forwards"),
                        ("band10", "pull_spx_options_0_70"), ("band30", "pull_spx_options_0_70_band30")]:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in ("OUT_DIR", "RAW_DIR"):
            if hasattr(mod, attr):
                setattr(mod, attr, raw / "spx_wrds_data")
        if hasattr(mod, "STAGE_DIR"):
            mod.STAGE_DIR = raw / mod.STAGE_DIR.name
            mod.BACKUP_ROOT = raw / "spx_wrds_data" / "old_options"
        scripts[short] = mod
    return scripts, raw


def test_pull_spx_wrds_writes_every_table(run, monkeypatch, capsys):
    """The main download writes all its files, keeps only SPX, applies its filters (20-50 days, strikes within 20%,
    a real bid), and skips files that already exist when run again."""
    scripts, raw = run
    out = raw / "spx_wrds_data"
    monkeypatch.setattr(sys, "argv", ["pull_spx_wrds.py", "--years", *map(str, YEARS)])
    scripts["wrds"].main()
    names = {"spx_prices", "spx_dividend_yield", "zero_curve", "spx_forwards"}
    names |= {f"spx_{kind}_{y}" for kind in ("options", "volsurface") for y in YEARS}
    assert {p.stem for p in out.glob("*.parquet")} == names

    prices = pd.read_parquet(out / "spx_prices.parquet")
    assert (prices.secid == SPX).all() and len(prices) == len(DAYS) and prices.date.is_monotonic_increasing
    o = pd.concat(pd.read_parquet(out / f"spx_options_{y}.parquet") for y in YEARS)
    dte = (o.exdate - o.date).dt.days
    assert len(o) > 0 and dte.between(20, 50).all() and (o.best_bid > 0).all()
    assert (o.strike / o.spot).between(0.8, 1.2).all()
    assert {"symbol", "am_settlement", "expiry_indicator"} <= set(o.columns)  # the optional columns came along
    assert (pd.read_parquet(out / "spx_volsurface_2019.parquet").secid == SPX).all()

    # Run again: nothing is downloaded twice.
    before = {p.name: p.stat().st_mtime_ns for p in out.glob("*.parquet")}
    capsys.readouterr()
    scripts["wrds"].main()
    assert "already done" in capsys.readouterr().out
    assert before == {p.name: p.stat().st_mtime_ns for p in out.glob("*.parquet")}


def test_pull_spx_forwards_reads_every_year(run, fake_db):
    """The forwards script finds the yearly fwdprd tables and joins them, SPX only, in date order."""
    scripts, raw = run
    scripts["forwards"].main()
    fwd = pd.read_parquet(raw / "spx_wrds_data" / "spx_forwards.parquet")
    assert len(fwd) == len(DAYS) and (fwd.secid == SPX).all() and fwd.date.is_monotonic_increasing
    assert list(fwd.columns) == ["secid", "date", "expiration", "amsettlement", "forwardprice"]


@pytest.mark.parametrize("short, band", [("band10", 0.10), ("band30", 0.30)])
def test_pull_spx_options_count_download_check_install(run, fake_db, capsys, short, band):
    """The 0-70-day option scripts: --count agrees with the download, the download keeps exactly the options
    within 0-70 days and the strike band, --check passes on the fake data, and --install swaps the files in
    while keeping the old ones."""
    scripts, raw = run
    mod = scripts[short]
    assert mod.BAND == band
    mod.count(YEARS)
    counted = int(capsys.readouterr().out.split("total ")[1].split()[0])

    mod.download(YEARS)
    o = mod.load(YEARS)
    assert list(o.columns) == mod.COLS and len(o) == counted

    # The same filter, worked out independently from the fake database.
    q = fake_db.execute("select o.date, o.exdate, o.strike_price / 1000.0 as strike, p.close "
                        "from (select * from optionm_all.opprcd2019 union all select * from optionm_all.opprcd2020) o "
                        "join optionm_all.secprd p on p.secid = o.secid and p.date = o.date "
                        f"where o.secid = {SPX}").df()
    q = q[((q.exdate - q.date).dt.days <= 70) & ((q.strike / q.close - 1).abs() <= band)]
    assert len(o) == len(q)

    # --check reads the index closes from spx_prices.parquet, as on the real download.
    (raw / "spx_wrds_data").mkdir(parents=True, exist_ok=True)
    fake_db.execute(f"select * from optionm_all.secprd where secid = {SPX}").df() \
        .astype({"date": "datetime64[ns]"}).to_parquet(raw / "spx_wrds_data" / "spx_prices.parquet")
    capsys.readouterr()
    mod.check(YEARS)
    assert "ALL CHECKS PASS" in capsys.readouterr().out

    old = raw / "spx_wrds_data" / "spx_options_2019.parquet"
    pd.DataFrame({"old": [1]}).to_parquet(old)
    mod.install(YEARS)
    assert pd.read_parquet(old).equals(pd.read_parquet(mod.STAGE_DIR / old.name))
    backups = list((raw / "spx_wrds_data" / "old_options").glob("*/spx_options_2019.parquet"))
    assert len(backups) == 1 and list(pd.read_parquet(backups[0]).columns) == ["old"]


def test_download_to_hedge(run, monkeypatch, tmp_path):
    """The whole data path of notebook 04 on the fake download: pull_spx_wrds.py and the 0-70-day, 30% band script,
    then prepare_wrds.py, then hedging.real builds one-month episodes and the delta hedge runs on them."""
    scripts, raw = run
    monkeypatch.setattr(sys, "argv", ["pull_spx_wrds.py", "--years", *map(str, YEARS)])
    scripts["wrds"].main()
    scripts["band30"].download(YEARS)

    spec = importlib.util.spec_from_file_location("prepare_wrds", SCRIPTS / "prepare_wrds.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    prepare.OUT = tmp_path / "data"
    prepare.main(raw / "spx_wrds_data", raw / "options_0_70dte_band30")

    history = real.load_history(prepare.OUT / "history.csv")
    options = real.load_options(prepare.OUT / "options.csv")
    surface = real.load_surface(prepare.OUT / "surface.csv")
    assert len(history) == len(DAYS) and history.notna().all().all()
    assert np.allclose(history.atm_iv_30d, SIGMA) and np.allclose(history.rate, 0.02)
    assert options.expiry.isin(history.date).all()

    m, info = real.episodes(history, options, surface)
    assert m.n_paths > 10 and np.allclose(m.S[:, 0], 100, atol=15)
    # The hedged option has its own quote on most days (not all: far out of the money it has no bid) and the
    # second option is found among the listed ones.
    assert info.quoted.mean() > 0.9 and (info.quoted2 > 0).mean() > 0.8
    pnl, _ = hedge.run(m, hedge.delta, cost=5e-4)
    assert np.isfinite(pnl).all() and np.sqrt(np.mean(pnl**2)) < 2.0
