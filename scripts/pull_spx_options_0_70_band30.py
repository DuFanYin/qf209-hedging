# pull SPX option prices from WRDS (optionmetrics), 0-70 days to expiry
# strikes within 30% of the index -> data/raw/options_0_70dte_band30
#
# usage (needs a WRDS account; prefix each with uv run --with wrds, from the repository root):
#   python scripts/pull_spx_options_0_70_band30.py --count     how many rows, before downloading
#   python scripts/pull_spx_options_0_70_band30.py             download, one file a year
#   python scripts/pull_spx_options_0_70_band30.py --check     spot-check a few dates
#   python scripts/pull_spx_options_0_70_band30.py --install   copy into spx_wrds_data (not needed for prepare_wrds.py)
#   (add --years 2019 2020 ... to only do some years)

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"  # the repository's data/raw folder (not in git)
RAW_DIR = RAW / "spx_wrds_data"
BAND = 0.30  # keep strikes with |K/S - 1| <= BAND
STAGE_DIR = RAW / "options_0_70dte_band30"
BACKUP_ROOT = RAW_DIR / "old_options"

SECID = 108105  # SPX
START = "2005-01-03"
END = "2025-08-29"
YEARS = list(range(2019, 2026)) + list(range(2005, 2019))  # do test period first
DTE_MIN = 0
DTE_MAX = 70
DROP_ZERO_BID = False

CHECK_DATES = ["2019-06-03", "2020-03-16", "2022-06-15", "2024-07-01", "2025-06-02"]
COLS = ["date", "exdate", "cp_flag", "strike", "best_bid", "best_offer", "impl_volatility", "am_settlement"]


def connect():
    import wrds
    db = wrds.Connection()  # asks for username, or uses ~/.pgpass
    lib = "optionm_all" if "optionm_all" in db.list_libraries() else "optionm"
    print("library:", lib)
    return db, lib


def base_query(lib, y):
    q = f"""
    from {lib}.opprcd{y} o
    join {lib}.secprd p on p.secid = o.secid and p.date = o.date
    where o.secid = {SECID}
      and o.date between '{START}' and '{END}'
      and (o.exdate - o.date) between {DTE_MIN} and {DTE_MAX}
      and abs(o.strike_price / 1000.0 / p.close - 1) <= {BAND}
    """
    if DROP_ZERO_BID:
        q += " and o.best_bid > 0"
    return q


def count(years):
    db, lib = connect()
    tables = db.list_tables(library=lib)
    total = 0
    for y in years:
        if f"opprcd{y}" not in tables:
            print(y, "no table")
            continue
        n = int(db.raw_sql("select count(*) as n " + base_query(lib, y)).n[0])
        total += n
        print(y, n, "rows")
    db.close()
    print("total", total, "rows, roughly", round(total * 10 / 1e9, 2), "GB")


def download(years):
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    db, lib = connect()
    tables = db.list_tables(library=lib)
    for y in years:
        out = STAGE_DIR / f"spx_options_{y}.parquet"
        if out.exists():
            print(y, "already done")
            continue
        if f"opprcd{y}" not in tables:
            print(y, "no table")
            continue
        t = time.time()
        sql = """select o.date, o.exdate, o.cp_flag, o.strike_price / 1000.0 as strike,
                 o.best_bid, o.best_offer, o.impl_volatility, o.am_settlement """ + base_query(lib, y)
        df = db.raw_sql(sql, date_cols=["date", "exdate"])
        df = df[COLS].sort_values(["date", "exdate", "cp_flag", "strike", "am_settlement"]).reset_index(drop=True)
        df.to_parquet(out.with_suffix(".tmp"), compression="zstd", index=False)
        out.with_suffix(".tmp").rename(out)
        dte = (df.exdate - df.date).dt.days
        print(y, len(df), "rows,", df.date.nunique(), "days, dte", dte.min(), "-", dte.max(),
              ",", round(time.time() - t), "s")
    db.close()
    print("done, band", BAND, "files in", STAGE_DIR)


def load(years):
    dfs = []
    for y in years:
        f = STAGE_DIR / f"spx_options_{y}.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def check(years):
    # for each date: is there a ~60d call, and does a ~30d atm call get quoted until expiry
    pf = RAW_DIR / "spx_prices.parquet"
    close = pd.read_parquet(pf).set_index("date").close if pf.exists() else None
    key = ["exdate", "strike", "cp_flag", "am_settlement"]
    all_ok = True

    for d in CHECK_DATES:
        d = pd.Timestamp(d)
        if d.year not in years:
            continue
        o = load([d.year, d.year + 1])
        if o is None:
            print(d.date(), "not downloaded")
            continue
        day = o[(o.date == d) & (o.cp_flag == "C")].copy()
        if len(day) == 0:
            print(d.date(), "no quotes")
            continue

        s = close.get(d) if close is not None else day.strike.median()
        day["dte"] = (day.exdate - day.date).dt.days
        day["m"] = (day.strike / s - 1).abs()
        print("\n", d.date(), "close", round(s, 2))

        c60 = day[day.dte.between(55, 65)].sort_values("m")
        ok60 = len(c60) > 0
        if ok60:
            r = c60.iloc[0]
            print("  60d call:", r.exdate.date(), r.dte, "days, K", r.strike, "bid/ask", r.best_bid, r.best_offer)
        else:
            print("  60d call: none  FAIL")

        c30 = day[day.dte.between(27, 33)].sort_values(["m", "dte"])
        ok30 = False
        if len(c30) > 0:
            r = c30.iloc[0]
            life = o.merge(pd.DataFrame([r[key]]), on=key)
            life = life[life.date >= d].sort_values("date")
            life["dte"] = (life.exdate - life.date).dt.days
            last_wk = life[life.dte <= 7]
            ok30 = len(last_wk) >= 3 and life.dte.min() <= 3
            print("  30d call:", r.exdate.date(), "K", r.strike, "quoted", len(life), "days,",
                  len(last_wk), "in last week, last quote", life.date.max().date(),
                  "PASS" if ok30 else "FAIL")
        else:
            print("  30d call: none  FAIL")

        all_ok = all_ok and ok60 and ok30

    print("\nALL CHECKS PASS" if all_ok else "\nSOME CHECKS FAILED")
    o = load(years)
    if o is not None:
        dte = (o.exdate - o.date).dt.days
        print("dte range", dte.min(), "-", dte.max(), "| cp", sorted(o.cp_flag.unique()))


def install(years):
    done = [y for y in years if (STAGE_DIR / f"spx_options_{y}.parquet").exists()]
    if not done:
        print("nothing to install")
        return
    # keep whatever is installed now in a dated backup folder
    bk = BACKUP_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    bk.mkdir(parents=True, exist_ok=True)
    for y in done:
        cur = RAW_DIR / f"spx_options_{y}.parquet"
        if cur.exists():
            shutil.move(str(cur), bk / cur.name)
        shutil.copy2(STAGE_DIR / cur.name, cur)
        print(y, "installed")
    print("old files in", bk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--years", type=int, nargs="+")
    args = ap.parse_args()
    years = args.years or YEARS

    # stop the mac sleeping during long downloads
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])

    if args.count:
        count(years)
    elif args.check:
        check(years)
    elif args.install:
        install(years)
    else:
        download(years)
