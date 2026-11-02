"""W4's data: turns the WRDS OptionMetrics download (SPX, secid 108105; one folder of parquet files) into the CSV
tables hedging.real reads. Run once:

    uv run python scripts/prepare_wrds.py data/raw/spx_wrds_data [data/raw/options_0_70dte_band30]

The optional second folder holds the option prices (spx_options_YYYY.parquet) with 0–70 days to expiry; when it is
given they replace the main folder's, which only cover 20–50 days.

history.csv  date, close, atm_iv_30d (30-day surface, mean of the 50-delta call and put), and the extras the
             episodes use: open (the AM settlement proxy), rate (30-day zero rate), dividend_yield; decimals.
options.csv  date, expiry, strike, call, bid, ask, iv, am; every expiry in the download (20–50 or 0–70 days),
             strikes within 30% of the close. Expiry is the last trading day on or before OptionMetrics' exdate
             (Saturdays before 2008, holidays).
surface.csv  date, days, strike, iv: the volatility surface up to 91 days, out-of-the-money side (puts below the
             50-delta call, calls from it up), for the days the option table no longer covers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data"
MONEYNESS = 0.30  # keep strikes within 30% of the close: in a crash the hedged strike stays in the table
SURFACE_DAYS = 91  # keep surface maturities up to 91 calendar days


def history(raw):
    """Builds the daily history table from the raw folder.

    Output: (DataFrame with date, close, atm_iv_30d, open, rate, dividend_yield; rates and volatilities as
    decimals, gaps filled with the previous day's value; Series counting the gaps filled in each column).
    """
    prices = pd.read_parquet(raw / "spx_prices.parquet", columns=["date", "close", "open"])
    surf = pd.concat([pd.read_parquet(f, columns=["date", "days", "delta", "impl_volatility"])
                      for f in sorted(raw.glob("spx_volsurface_*.parquet"))])
    # 30-day at-the-money iv: average of the 50-delta call (delta 50) and the 50-delta put (delta -50).
    atm = surf[(surf.days == 30) & (surf.delta.abs() == 50)].groupby("date").impl_volatility.mean()
    zero = pd.read_parquet(raw / "zero_curve.parquet")
    # Each day's zero curve has rates at a few maturities; interpolate the 30-day rate.
    rate = zero.groupby("date").apply(lambda z: np.interp(30, z.days.to_numpy(float), z.rate.to_numpy(float)))
    div = pd.read_parquet(raw / "spx_dividend_yield.parquet").set_index("date").rate
    out = prices.set_index("date").astype(float)
    # The source quotes rates and yields in percent, so divide them by 100; the iv is already a decimal.
    out["atm_iv_30d"] = atm
    out["rate"] = rate / 100
    out["dividend_yield"] = div / 100
    gaps = out.isna().sum()
    out = out.ffill()  # the few days the surface or the curve skips
    return out.reset_index()[["date", "close", "atm_iv_30d", "open", "rate", "dividend_yield"]], gaps


def options(raw, dates, close):
    """Builds the option quotes table from the raw yearly files.

    Inputs: raw the folder with the spx_options_YYYY.parquet files, dates the trading days (DatetimeIndex), close the closes indexed by date.
    Output: DataFrame with date, expiry, strike, call, bid, ask, iv, am.
    """
    cols = ["date", "exdate", "cp_flag", "best_bid", "best_offer", "impl_volatility", "am_settlement", "strike"]
    tables = []
    # One file at a time, filtering as we go, to keep memory down.
    for f in sorted(raw.glob("spx_options_*.parquet")):
        o = pd.read_parquet(f, columns=cols)
        spot = close.reindex(o.date).to_numpy()
        o = o[np.abs(o.strike.to_numpy(float) / spot - 1) <= MONEYNESS]
        # OptionMetrics' exdate can be a Saturday (before 2008) or a holiday; the option's last trading day is the
        # last trading date on or before it.
        tables.append(pd.DataFrame({
            "date": o.date, "expiry": dates[np.searchsorted(dates, o.exdate, side="right") - 1],
            "strike": o.strike.astype(float), "call": (o.cp_flag == "C").astype(int),
            "bid": o.best_bid.astype(float), "ask": o.best_offer.astype(float),
            "iv": o.impl_volatility.astype(float), "am": o.am_settlement.astype(int)}))
    return pd.concat(tables, ignore_index=True)


def surface(raw):
    """Builds the volatility surface table: date, days, strike, iv, up to SURFACE_DAYS, out-of-the-money side."""
    cols = ["date", "days", "delta", "impl_volatility", "impl_strike", "cp_flag"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(raw.glob("spx_volsurface_*.parquet"))])
    # Puts with delta above -50 lie below the money; calls with delta up to 50 lie at or above it.
    otm = ((s.cp_flag == "P") & (s.delta > -50)) | ((s.cp_flag == "C") & (s.delta <= 50))
    s = s[otm & (s.days <= SURFACE_DAYS) & (s.impl_strike > 0) & s.impl_volatility.notna()]  # 6% unfitted nodes
    return pd.DataFrame({"date": s.date, "days": s.days.astype(int), "strike": s.impl_strike.astype(float),
                         "iv": s.impl_volatility.astype(float)}).sort_values(["date", "days", "strike"])


def main(raw, option_files=None):
    """Reads the raw folder, writes data/history.csv, data/options.csv and data/surface.csv, and prints a summary.
    option_files: an optional folder whose spx_options_YYYY.parquet files are used instead of the raw folder's."""
    raw = Path(raw)
    OUT.mkdir(exist_ok=True)
    h, gaps = history(raw)
    print("history:", len(h), "days", h.date.min().date(), "to", h.date.max().date(), "| filled:", gaps.to_dict())
    h.to_csv(OUT / "history.csv", index=False)
    dates = pd.DatetimeIndex(h.date)
    o = options(Path(option_files or raw), dates, h.set_index("date").close)
    print("options:", len(o), "rows on", o.date.nunique(), "days;", o.expiry.nunique(), "expiries")
    o.to_csv(OUT / "options.csv", index=False)
    s = surface(raw)
    print("surface:", len(s), "rows on", s.date.nunique(), "days")
    s.to_csv(OUT / "surface.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
