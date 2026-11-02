"""W4, the real market, daily. Three tables, as CSV (scripts/prepare_wrds.py writes them from the WRDS OptionMetrics
download):

history: one row a day
    date        trading day
    close       the underlying's close
    atm_iv_30d  30-day at-the-money implied volatility, as a decimal (0.18)
    optional: open (settles AM options), rate and dividend_yield (continuous, decimals; 0 when absent)

options: one row an option a day
    date, expiry   trading day, expiry (its last trading day)
    strike         strike
    call           1 for a call, 0 for a put
    bid, ask       closing quotes
    iv             implied volatility at the mid, as a decimal
    optional: am   1 for AM settlement (at the expiry day's open)

surface: the volatility surface a day, for the days an option has no quote
    date, days, strike, iv   calendar days to maturity, strike, implied volatility

load_history feeds the generator (W3) and the Heston calibration (W2) with real data. episodes turns the option
table into a Market the hedges run in.
"""

import numpy as np
import pandas as pd

from hedging import bs
from hedging.worlds import YEAR, Market, S0

HISTORY_COLUMNS = ["date", "close", "atm_iv_30d"]
OPTION_COLUMNS = ["date", "expiry", "strike", "call", "bid", "ask", "iv"]
SURFACE_COLUMNS = ["date", "days", "strike", "iv"]


def _read(path, columns, optional=()):
    """Reads a CSV, checks the required columns are there, keeps those plus any optional ones present, and sorts
    by date (and expiry, for the option table). Raises ValueError naming any missing column."""
    table = pd.read_csv(path, parse_dates=[c for c in ("date", "expiry") if c in columns])
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {columns}")
    columns = columns + [c for c in optional if c in table.columns]
    return table[columns].sort_values(columns[:2] if "expiry" in columns else "date").reset_index(drop=True)


def load_history(path):
    """The daily history; its close and atm_iv_30d go to generator.fit and worlds.calibrate_heston.

    Output: DataFrame with date, close, atm_iv_30d and whichever of open, rate, dividend_yield the file has.
    """
    return _read(path, HISTORY_COLUMNS, ("open", "rate", "dividend_yield"))


def load_options(path):
    """The option quotes table (see the module docstring), with the am column if present."""
    return _read(path, OPTION_COLUMNS, ("am",))


def load_surface(path):
    """The volatility surface table (see the module docstring)."""
    return _read(path, SURFACE_COLUMNS)


class _Surface:
    """Implied volatility at any strike and maturity: linear in log-strike along each maturity of the day's surface
    (flat beyond its ends), linear in total variance across maturities, flat below the shortest. A day missing
    from the surface uses the last one before it.

    Build it once from the surface table; then call it as vol(date, strike, calendar_days) to get an implied
    volatility (decimal).
    """

    def __init__(self, surface):
        self.dates = pd.DatetimeIndex(surface.date.unique())
        surface = surface.sort_values(["date", "days", "strike"])
        # For each date, a list of (maturity in days, log strikes, ivs), one entry per maturity, for fast lookup.
        self.days = {d: [(n, np.log(g.strike.to_numpy()), g.iv.to_numpy()) for n, g in s.groupby("days")]
                     for d, s in surface.groupby("date")}

    def __call__(self, date, K, days):
        # The latest surface date on or before `date`.
        nodes = self.days[self.dates[self.dates.searchsorted(date, side="right") - 1]]
        n = np.array([m for m, _, _ in nodes], float)
        # Step 1: along each maturity, interpolate in log-strike to get the iv at strike K.
        iv = np.array([np.interp(np.log(K), x, y) for _, x, y in nodes])
        if days <= n[0]:
            return iv[0]
        # Step 2: across maturities, interpolate total variance (iv^2 * days), then convert back to a volatility.
        return np.sqrt(np.interp(days, n, iv**2 * n) / days)  # beyond the longest: its total variance, flat


def episodes(history, options, surface, days=21, moneyness=1.0, call=True, start_every=5, since=None):
    """Markets of real paths. Start days at least start_every trading days apart (from `since`), each the first
    day with an option expiring exactly `days` trading days later; of those, the strike nearest moneyness * close
    (PM settlement preferred), followed to expiry:

    S   the forward to the expiry, close * exp((rate - dividend_yield) * calendar days / 365), ending at the
        settlement price (the expiry's close, or its open for AM settlement);
    V   the quoted mid, carried forward to expiry, at the start; the payoff at expiry; Black–Scholes on S between;
    iv  the option's own quote while the option table has it, the surface at its strike and maturity after;
    H   the second option: same strike and type, the listed expiry nearest twice the time (2 * days trading
        days), at least days // 2 trading days after the first, that is quoted at that strike on the start day.
        Its quoted mid, carried forward to the first expiry, on the days it is quoted; Black–Scholes on S at the
        surface's volatility on the days it is not. When no such option is listed, a model option at exactly
        twice the time, priced from the surface every day (the fallback the 20–50-day download forced on every
        episode);
    iv2 its own quoted iv, or the surface's, on the same days.

    Every episode is scaled to strike 100 (S0 = 100 at the money), the units of the simulated worlds. Returns the
    Market (one row a start day; T2 one value per row, since each second option has its own expiry) and a table
    per row: start, expiry, strike, settlement, the option's relative half spread at the start and its quoted iv
    against the surface's; the second option's expiry, its relative half spread at the start (the cost of trading
    it; the first option's when it is not quoted), and the share of days each option has its own quote.

    Inputs: history, options, surface the three tables from the load_ functions; days the option's life in trading
    days; moneyness the target strike as a multiple of the close (1.0 = at the money); call or put; start_every the
    minimum gap between start days, in trading days; since an optional first start date.
    Output: (Market with arrays (n_episodes, days + 1), DataFrame with one row per episode).
    """
    dates = pd.DatetimeIndex(history.date)
    close = history.close.to_numpy(float)
    open_ = history.open.to_numpy(float) if "open" in history else close
    rate = history.rate.to_numpy(float) if "rate" in history else np.zeros(len(dates))
    # Carry = interest rate minus dividend yield: the forward grows at this rate relative to the spot.
    carry = rate - (history.dividend_yield.to_numpy(float) if "dividend_yield" in history else 0.0)
    # Trading dates extended past the end of the history (weekdays only, no holidays), so the second option's
    # expiry date can be found even for episodes near the end of the data.
    calendar = dates.append(pd.bdate_range(dates[-1] + pd.offsets.BDay(), periods=2 * days))
    # Keep options of the chosen type with a real bid and an implied volatility.
    opts = options[(options.call == int(call)) & (options.bid > 0) & options.iv.notna()].copy()
    if "am" not in opts:
        opts["am"] = 0
    # starts_on: each day's options, to choose an episode's option on its start day.
    # quoted: the same rows indexed by (date, expiry, am, strike), to find that option's own iv on later days.
    starts_on = dict(tuple(opts.groupby("date")))
    quoted = opts.set_index(["date", "expiry", "am", "strike"]).sort_index()
    vol = _Surface(surface)

    first = 0 if since is None else dates.searchsorted(pd.Timestamp(since))
    # The same daily grid for every episode, in years: 0, 1/252, ..., days/252.
    t = np.linspace(0.0, days / YEAR, days + 1)
    T = t[-1]
    paths, rows = [], []
    last = -start_every
    for s in range(first, len(dates) - days):
        # Leave at least start_every trading days after the last accepted start.
        if s - last < start_every:
            continue
        end = s + days
        today = starts_on.get(dates[s])
        if today is None:
            continue
        # Only an option expiring exactly `days` trading days later fits the 21-day grid; otherwise try tomorrow.
        today = today[today.expiry == dates[end]]
        if today.empty:
            continue
        today = today[today.am == today.am.min()]  # PM settlement when both are listed
        pick = today.iloc[np.argmin(np.abs(today.strike.to_numpy() - moneyness * close[s]))]
        K, am = pick.strike, int(pick.am)
        # The second option: same strike, quoted today, the expiry nearest twice the time (PM when both are
        # listed), and at least days // 2 trading days after the first (one expiring days after it would cancel
        # almost none of its vega). No candidate: the model option at exactly twice the time.
        target = calendar[s + 2 * days]
        later = starts_on[dates[s]]
        later = later[(later.strike == K) & (later.expiry >= calendar[end + days // 2])]
        if later.empty:
            exp2, am2, pick2 = target, None, None
        else:
            gap = np.abs((later.expiry - target).dt.days.to_numpy())
            later = later[gap == gap.min()]
            later = later[later.expiry == later.expiry.max()]  # equally near: the later one
            pick2 = later.iloc[np.argmin(later.am.to_numpy())]
            exp2, am2 = pick2.expiry, int(pick2.am)
        T2 = (calendar.searchsorted(exp2) - s) / YEAR
        # Calendar days left, from each day of the episode, to the option's expiry (cal) and the second one's (cal2).
        cal = np.array([(dates[end] - dates[i]).days for i in range(s, end + 1)], float)
        cal2 = np.array([(exp2 - dates[i]).days for i in range(s, end + 1)], float)
        settle = open_[end] if am else close[end]
        # Forward price to expiry on each day before expiry; on the expiry day, the settlement price itself.
        S = np.append(close[s:end] * np.exp(carry[s:end] * cal[:-1] / 365), settle)
        iv, iv2, H = np.empty(days + 1), np.empty(days + 1), np.empty(days + 1)
        own, own2 = np.zeros(days + 1, bool), np.zeros(days + 1, bool)
        for j, i in enumerate(range(s, end + 1)):
            # Use the option's own quoted iv when it was quoted that day; otherwise read the surface.
            key = (dates[i], dates[end], am, K)
            own[j] = key in quoted.index and i < end
            iv[j] = np.atleast_1d(quoted.loc[key, "iv"])[0] if own[j] else vol(dates[i], K, cal[j])
            # The second option: its quoted mid (grown at the rate to the first expiry, like the premium) and iv.
            key2 = (dates[i], exp2, am2, K)
            own2[j] = am2 is not None and key2 in quoted.index
            if own2[j]:
                q = quoted.loc[key2].iloc[0]
                H[j] = 0.5 * (q.bid + q.ask) * np.exp(rate[i] * cal[j] / 365)
                iv2[j] = q.iv
            else:
                iv2[j] = vol(dates[i], K, cal2[j])
                H[j] = bs.price(S[j], K, T2 - t[j], iv2[j], call)
        # No meaningful iv on the expiry day; it is never used for a trade, so repeat the day before.
        iv[-1] = iv[-2]
        V = bs.price(S, K, T - t, iv, call)
        mid = 0.5 * (pick.bid + pick.ask)
        # The premium actually received: the quoted mid, grown at the rate to expiry so it is in the same
        # (forward, rate-zero) terms as everything else.
        V[0] = mid * np.exp(rate[s] * cal[0] / 365)
        # Rescale prices so the strike is 100, like the simulated worlds; volatilities are unaffected.
        scale = S0 / K
        paths.append((S * scale, V * scale, iv, H * scale, iv2, T2))
        last = s
        half_spread = 0.5 * (pick.ask - pick.bid) / mid
        rows.append({"start": dates[s], "expiry": dates[end], "strike": K, "close": close[s],
                     "settlement": "AM" if am else "PM", "half_spread": half_spread,
                     "iv_quoted": pick.iv, "iv_surface": vol(dates[s], K, cal[0]),
                     "expiry2": exp2, "half_spread2": half_spread if pick2 is None else
                     (pick2.ask - pick2.bid) / (pick2.ask + pick2.bid),
                     "quoted": own[:-1].mean(), "quoted2": own2.mean()})
    if not paths:
        raise ValueError("no episode: no option expires exactly `days` trading days after a start day")
    S, V, iv, H, iv2, T2 = (np.array(a) for a in zip(*paths))
    return Market(t, S, S0, V, iv, S0, T2, H, iv2, T=T, call=call), pd.DataFrame(rows)
