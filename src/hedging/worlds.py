"""The worlds the hedges run in. Each builds a Market: paths of the underlying, the hedged option and a second
option for gamma and vega hedges, with their implied volatilities, on one time grid.

W1 Black–Scholes, W2 Heston, the reference market (Bates: stochastic volatility with jumps, standing in for the
real history until real data is loaded through hedging.real), W3 the generator's paths.

Units: S0 = 100, strike 100 (at the money), time in years with 252 trading days a year.
"""

from dataclasses import dataclass, replace

import numpy as np

from hedging import bs
from hedging.models import StochasticVol, gbm

DAYS = 21  # the hedged option: 30 calendar days, 21 trading days
YEAR = 252
T = DAYS / YEAR
S0 = 100.0


@dataclass
class Market:
    """Rows are paths, columns the times t[0] .. t[N]. Prices are forward prices (rate 0). The trader is short one
    option (strike K, expiry T); the second option (K2, T2 > T) is a hedge instrument, unwound at t[N] = T.

    Fields (n_paths rows, N + 1 columns where a 2-D array):
        t            times in years, array (N + 1,), from 0 to T
        S            underlying prices, (n_paths, N + 1)
        K            strike of the option sold
        V            price of the option sold; column 0 is the premium received, the last column the payoff
        iv           its implied volatility, (n_paths, N + 1), annual
        K2, T2       strike and expiry (years) of the second option; T2 may be one value per path, (n_paths,)
                     (W4: each real second option has its own listed expiry)
        H            price of the second option, (n_paths, N + 1)
        iv2          its implied volatility, (n_paths, N + 1)
        T            expiry of the option sold, years
        call         True for a call, False for a put
        model_delta  the world's own model delta, (n_paths, N + 1), where the world has one (W2), else None
    """

    t: np.ndarray
    S: np.ndarray
    K: float
    V: np.ndarray
    iv: np.ndarray
    K2: float
    T2: float | np.ndarray
    H: np.ndarray
    iv2: np.ndarray
    T: float = T
    call: bool = True
    model_delta: np.ndarray | None = None

    @property
    def n_paths(self):
        """Number of scenarios (rows)."""
        return self.S.shape[0]

    @property
    def n_steps(self):
        """Number of time steps (columns minus one)."""
        return self.t.size - 1

    def every(self, k):
        """The same paths on a coarser grid: every k-th time.

        This is how hedging frequencies are compared on identical scenarios: simulate on a fine grid once, then
        keep every k-th column (e.g. k = 78 turns a 5-minute grid into a daily one). k must divide n_steps.
        """
        cols = slice(None, None, k)
        assert self.n_steps % k == 0
        pick = lambda a: None if a is None else a[:, cols]
        return replace(self, t=self.t[cols], S=self.S[:, cols], V=self.V[:, cols], iv=self.iv[:, cols],
                       H=self.H[:, cols], iv2=self.iv2[:, cols], model_delta=pick(self.model_delta))

    def take(self, rows):
        """A Market with only the chosen paths (rows: indices or a boolean mask), e.g. W4's test period."""
        pick = lambda a: None if a is None else a[rows]
        return replace(self, S=self.S[rows], V=self.V[rows], iv=self.iv[rows], H=self.H[rows],
                       iv2=self.iv2[rows], T2=self.T2[rows] if np.ndim(self.T2) else self.T2,
                       model_delta=pick(self.model_delta))


def _instrument(K=None, T2=2 * T):
    """Strike and expiry of the second option: at the money (S0) and twice the hedged option's life by default."""
    return S0 if K is None else K, T2


def black_scholes(n_paths, steps_per_day, sigma=0.2, rng=None, mu=0.0, K=S0, call=True):
    """W1: geometric Brownian motion; every option priced at the true volatility.

    Inputs: n_paths scenarios, steps_per_day grid points per trading day (78 = every 5 minutes), sigma annual
    volatility, rng numpy random Generator, mu real-world drift, K strike, call or put.
    Output: a Market with 21 * steps_per_day steps. Implied volatility is sigma everywhere (no smile, no moves).
    """
    rng = rng or np.random.default_rng(0)
    n = DAYS * steps_per_day
    t = np.linspace(0.0, T, n + 1)
    S = gbm(S0, sigma, T, n, n_paths, rng, mu)
    K2, T2 = _instrument()
    iv = np.broadcast_to(np.float64(sigma), S.shape)
    V = bs.price(S, K, T - t, sigma, call)
    H = bs.price(S, K2, T2 - t, sigma)
    return Market(t, S, K, V, iv, K2, T2, H, iv, call=call)


# Time value below which an implied volatility is not trusted (in price units, with S0 = 100).
MIN_TIME_VALUE = 1e-3


def _implied_vol(model, value, S, K, tau, v, call):
    """Implied volatility backed out of the model price. Deep in or out of the money the time value is below the
    pricer's accuracy and the inversion is meaningless (it gives near-zero volatility, and greek ratios blow up):
    there the model's own volatility over the option's life is used, as a trader would."""
    iv = bs.implied_vol(value, S, K, tau, call)
    # Time value = price minus what the option would pay if exercised now.
    time_value = value - np.maximum(S - K if call else K - S, 0.0)
    level = np.sqrt(model.total_variance(tau, v) / np.maximum(tau, 1e-12))
    return np.where(time_value < MIN_TIME_VALUE, level, iv)


def stochastic_vol(model: StochasticVol, n_paths, steps_per_day, rng=None, v0=None, mu=0.0, K=S0, call=True,
                   substeps=4, with_model_delta=True):
    """W2 (Heston) or the reference market (Bates): options priced by the model, implied volatilities backed out.
    v0 None draws the starting variance from the stationary distribution; a number starts every path there.

    Inputs: model the StochasticVol parameters, n_paths, steps_per_day, rng, v0, mu as in black_scholes; substeps
    extra simulation steps per grid step; with_model_delta False skips the model's own delta (slow to compute).
    Output: a Market. Unlike W1, implied volatility now differs by path and moves over time.
    """
    rng = rng or np.random.default_rng(0)
    n = DAYS * steps_per_day
    t = np.linspace(0.0, T, n + 1)
    start = model.stationary_variance(n_paths, rng) if v0 is None else v0
    S, v = model.simulate(S0, start, T, n, n_paths, rng, substeps, mu)
    K2, T2 = _instrument()
    tau, tau2 = np.broadcast_to(T - t, S.shape), np.broadcast_to(T2 - t, S.shape)
    V = model.price(S, K, tau, v, call)
    H = model.call(S, K2, tau2, v)
    iv = _implied_vol(model, V, S, K, tau, v, call)
    # At expiry there is no time left and no implied volatility; repeat the previous one (it is never traded on).
    iv[:, -1] = iv[:, -2]
    iv2 = _implied_vol(model, H, S, K2, tau2, v, True)
    delta = model.delta(S, K, tau, v, call) if with_model_delta else None
    return Market(t, S, K, V, iv, K2, T2, H, iv2, call=call, model_delta=delta)


def from_factor_paths(S, atm_iv, K=S0, call=True):
    """W3, and any world given as daily paths of the underlying and its 30-day at-the-money implied volatility
    (the generator's output): both options priced by Black–Scholes at that volatility, a flat smile.

    Inputs: S and atm_iv, arrays (n_paths, n_days + 1), prices starting at S0 and annual volatilities.
    Output: a Market on a daily grid over the option's life.
    """
    n = S.shape[1] - 1
    t = np.linspace(0.0, T, n + 1)
    K2, T2 = _instrument()
    V = bs.price(S, K, T - t, atm_iv, call)
    H = bs.price(S, K2, T2 - t, atm_iv)
    return Market(t, S, K, V, atm_iv, K2, T2, H, atm_iv, call=call)


# The reference market: SPX-like stochastic volatility with downward jumps. W2's Heston has no jumps.
REFERENCE = StochasticVol(kappa=3.0, theta=0.035, xi=0.55, rho=-0.75, lam=1.0, mu_j=-0.05, sigma_j=0.07)
HESTON = StochasticVol(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def calibrate_heston(close, atm_iv, dt=1.0 / YEAR):
    """Heston fitted to a daily history by moments, taking the squared 30-day ATM volatility as the variance:
    theta its mean, kappa from its daily autocorrelation, xi from the size of its changes, rho from their
    correlation with returns. No jumps: fitted to the reference market, this is a misspecified model.

    Inputs: close daily closes and atm_iv daily implied volatilities (decimals), 1-D arrays of the same length;
    dt the length of a day in years. Output: a StochasticVol (no jumps).
    """
    v = np.asarray(atm_iv, float) ** 2
    r = np.diff(np.log(np.asarray(close, float)))
    dv = np.diff(v)
    # Regress tomorrow's variance on today's. In Heston the slope is exp(-kappa * dt): the closer to 1, the slower
    # variance returns to its mean. Clipping keeps the logarithm finite.
    slope = np.polyfit(v[:-1], v[1:], 1)[0]
    kappa = -np.log(np.clip(slope, 1e-6, 1 - 1e-6)) / dt
    # In Heston a daily variance change has standard deviation xi * sqrt(v * dt); divide by sqrt(v) and sqrt(dt).
    xi = np.std(dv / np.sqrt(v[:-1])) / np.sqrt(dt)
    rho = np.corrcoef(r, dv)[0, 1]
    return StochasticVol(kappa=float(kappa), theta=float(v.mean()), xi=float(xi), rho=float(rho))


def reference_history(model=REFERENCE, years=10, rng=None, substeps=8):
    """One long daily history of the reference market (close and 30-day ATM implied volatility), the stand-in for
    the real history the generator learns from.

    Output: (close, atm_iv), two 1-D arrays of length years * 252 + 1, close starting at S0.
    """
    rng = rng or np.random.default_rng(7)
    n = years * YEAR
    S, v = model.simulate(S0, model.theta, years, n, 1, rng, substeps)
    return S[0], model.atm_iv(T, v[0])
