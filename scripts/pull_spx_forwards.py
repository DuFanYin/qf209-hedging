# pull SPX forward prices from WRDS into data/raw/spx_wrds_data/spx_forwards.parquet
# forwards are split into yearly tables (fwdprd2005, fwdprd2006, ...) so this goes through all of them
#
# usage (needs a WRDS account): uv run --with wrds python scripts/pull_spx_forwards.py

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"  # the repository's data/raw folder (not in git)
OUT_DIR = RAW / "spx_wrds_data"
SECID = 108105  # SPX
START = "2005-01-01"
END = "2025-08-29"


def main():
    import wrds

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection()  # asks for username, or uses ~/.pgpass

    tables = db.list_tables(library="optionm_all")
    names = sorted(t for t in tables if t.startswith("fwdprd") and t[6:].isdigit())
    if not names:
        print("no fwdprd<year> tables found")
        db.close()
        return
    print("found", names[0], "to", names[-1], f"({len(names)} tables)")

    parts = []
    for t in names:
        df = db.raw_sql(f"select * from optionm_all.{t} where secid = {SECID} "
                        f"and date between '{START}' and '{END}'", date_cols=["date"])
        print(" ", t, len(df), "rows")
        parts.append(df)
    db.close()

    fwd = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    out = OUT_DIR / "spx_forwards.parquet"
    fwd.to_parquet(out, compression="zstd", index=False)
    print("saved", len(fwd), "rows,", fwd.date.min().date(), "to", fwd.date.max().date())


if __name__ == "__main__":
    main()
