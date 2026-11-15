# pull SPX data from WRDS (optionmetrics) into data/raw/spx_wrds_data
#   spx_prices, spx_options_<year> (20-50 dte, strikes within 20%, bid > 0),
#   spx_volsurface_<year>, spx_forwards, spx_dividend_yield, zero_curve
#
# usage (needs a WRDS account):
#   uv run --with wrds python scripts/pull_spx_wrds.py
#   uv run --with wrds python scripts/pull_spx_wrds.py --years 2020 2021
# files that already exist are skipped, so just rerun if it stops

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"  # the repository's data/raw folder (not in git)
OUT_DIR = RAW / "spx_wrds_data"
TICKER = "SPX"
SECID = 108105  # used if the lookup is unclear
START = "2005-01-01"
END = "2025-08-29"
YEARS = list(range(2005, 2026))
DTE_MIN = 20
DTE_MAX = 50
K_MIN = 0.8  # strike / spot
K_MAX = 1.2

BASE_COLS = ["date", "exdate", "cp_flag", "best_bid", "best_offer", "volume", "open_interest",
             "impl_volatility", "delta", "gamma", "vega", "theta", "optionid"]
# only taken if the table has them (needed to separate SPX and SPXW)
EXTRA_COLS = ["symbol", "am_settlement", "expiry_indicator", "root", "suffix", "contract_size", "ss_flag"]


def save(df, name):
    path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(path.with_suffix(".tmp"), compression="zstd", index=False)
    path.with_suffix(".tmp").rename(path)
    print(" ", name, len(df), "rows")


def get_secid(db):
    try:
        s = db.raw_sql(f"select secid, ticker, issuer, issue_type from optionm_all.securd where ticker = '{TICKER}'")
    except Exception as e:
        print("lookup failed:", e, "-> using", SECID)
        return SECID
    print(s.to_string(index=False))
    if len(s) == 1:
        return int(s.secid[0])
    if SECID in s.secid.astype(int).values:
        return SECID
    raise SystemExit("can't tell which secid is SPX, set SECID by hand")


def table_cols(db, table):
    try:
        return set(db.describe_table("optionm_all", table)["name"].str.lower())
    except Exception:
        return set()


def get_forwards(db, secid, years):
    # forwards are split into yearly tables (fwdprd2005, ...) in our library
    tables = db.list_tables(library="optionm_all")
    if "fwdprd" in tables:
        return db.raw_sql(f"select * from optionm_all.fwdprd where secid = {secid} "
                          f"and date between '{START}' and '{END}'", date_cols=["date"])
    parts = []
    for y in years:
        if f"fwdprd{y}" in tables:
            parts.append(db.raw_sql(f"select * from optionm_all.fwdprd{y} where secid = {secid}",
                                    date_cols=["date"]))
    if not parts:
        print("  no forward tables found")
        return None
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


def main():
    import wrds

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+")
    years = ap.parse_args().years or YEARS

    # stop the mac sleeping during long downloads
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection()  # asks for username, or uses ~/.pgpass
    secid = get_secid(db)
    print("secid", secid)

    other = {
        "spx_prices": f"""select secid, date, low, high, close, volume, return, cfadj, open, cfret, shrout
                          from optionm_all.secprd where secid = {secid}
                          and date between '{START}' and '{END}' order by date""",
        "spx_dividend_yield": f"""select * from optionm_all.idxdvd where secid = {secid}
                                  and date between '{START}' and '{END}'""",
        "zero_curve": f"""select * from optionm_all.zerocd
                          where date between '{START}' and '{END}' order by date, days""",
    }
    for name, sql in other.items():
        if (OUT_DIR / f"{name}.parquet").exists():
            print(" ", name, "already done")
            continue
        try:
            save(db.raw_sql(sql, date_cols=["date"]), name)
        except Exception as e:
            print(" ", name, "FAILED:", str(e).splitlines()[0])

    if not (OUT_DIR / "spx_forwards.parquet").exists():
        fwd = get_forwards(db, secid, years)
        if fwd is not None:
            save(fwd, "spx_forwards")

    for y in years:
        if (OUT_DIR / f"spx_options_{y}.parquet").exists():
            print(y, "options already done")
        else:
            t = time.time()
            have = table_cols(db, f"opprcd{y}")
            extra = [c for c in EXTRA_COLS if c in have]
            cols = ", ".join("o." + c for c in BASE_COLS + extra)
            sql = f"""
            select {cols}, o.strike_price / 1000.0 as strike, p.close as spot
            from optionm_all.opprcd{y} o
            join optionm_all.secprd p on p.secid = o.secid and p.date = o.date
            where o.secid = {secid}
              and (o.exdate - o.date) between {DTE_MIN} and {DTE_MAX}
              and o.strike_price / 1000.0 / p.close between {K_MIN} and {K_MAX}
              and o.best_bid > 0
            """
            df = db.raw_sql(sql, date_cols=["date", "exdate"])
            df = df.sort_values(["date", "exdate", "cp_flag", "strike"]).reset_index(drop=True)
            save(df, f"spx_options_{y}")
            dte = (df.exdate - df.date).dt.days
            print(y, df.date.nunique(), "days, dte", dte.min(), "-", dte.max(), ",", round(time.time() - t), "s")

        if not (OUT_DIR / f"spx_volsurface_{y}.parquet").exists():
            vs = db.raw_sql(f"select * from optionm_all.vsurfd{y} where secid = {secid}", date_cols=["date"])
            save(vs, f"spx_volsurface_{y}")

    db.close()
    print("done, files in", OUT_DIR)


if __name__ == "__main__":
    main()
