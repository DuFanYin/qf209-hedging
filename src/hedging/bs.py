"""Black–Scholes on forward prices (rate 0): prices, greeks, implied volatility, and the closed forms the
simulations are checked against (Derman–Kamal, Leland, Whalley–Wilmott).

Units: S and K are prices (the project uses S0 = K = 100 for an at-the-money option), tau is in years, sigma is
annual (0.2 = 20%). Arguments can be numbers or numpy arrays of matching (broadcastable) shapes.
"""

import numpy as np
from scipy.special import ndtr

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _d1(S, K, tau, sigma):
    """The Black–Scholes d1 term, and s = sigma * sqrt(tau) (the standard deviation of log S to expiry)."""
    # Floors keep the division safe at expiry (tau = 0) or at zero volatility; those cases are handled by callers.
    s = np.maximum(sigma, 1e-8) * np.sqrt(np.maximum(tau, 1e-12))
    return (np.log(S / K) + 0.5 * s * s) / s, s


def price(S, K, tau, sigma, call=True):
    """Black–Scholes price of a European call (or put) on a forward, rate zero.

    Inputs: S price of the underlying, K strike, tau years to expiry, sigma annual volatility; numbers or arrays.
    Output: the option price, same shape as the broadcast inputs, in the same units as S and K.
    At expiry (tau <= 0) it returns the payoff.
    """
    d1, s = _d1(S, K, tau, sigma)
    c = np.where(tau > 0, S * ndtr(d1) - K * ndtr(d1 - s), np.maximum(S - K, 0.0))
    # Put-call parity with rate zero: call - put = S - K.
    return c if call else c - (S - K)


def delta(S, K, tau, sigma, call=True):
    """Black–Scholes delta: units of the underlying held to hedge one option.

    Same inputs as price(). Output between 0 and 1 for a call, between -1 and 0 for a put. At expiry it is 1 if the
    option ends in the money and 0 otherwise.
    """
    d1, _ = _d1(S, K, tau, sigma)
    d = np.where(tau > 0, ndtr(d1), (S > K).astype(float))
    return d if call else d - 1.0


def gamma(S, K, tau, sigma):
    """Black–Scholes gamma: change in delta per unit change in S (the same for a call and a put). Zero at expiry."""
    d1, s = _d1(S, K, tau, sigma)
    return np.where(tau > 0, np.exp(-0.5 * d1 * d1) * INV_SQRT_2PI / (S * s), 0.0)


def vega(S, K, tau, sigma):
    """Per unit of volatility (1.0 = 100 vol points).

    Black–Scholes vega: change in the option price per unit change in sigma (the same for a call and a put). Divide
    by 100 for the change per volatility point. Zero at expiry.
    """
    d1, _ = _d1(S, K, tau, sigma)
    return np.where(tau > 0, S * np.exp(-0.5 * d1 * d1) * INV_SQRT_2PI * np.sqrt(np.maximum(tau, 0.0)), 0.0)


def implied_vol(value, S, K, tau, call=True, lo=1e-4, hi=5.0, iterations=60):
    """Bisection on the call price, vectorised; puts go through parity. Prices outside the no-arbitrage bounds clip.

    Finds the volatility at which the Black–Scholes price equals `value`. Inputs as in price(), with `value` the
    observed option price; output is an annual volatility, same shape as the broadcast inputs. The search stays in
    [lo, hi]; 60 halvings of that range give far more precision than needed.
    """
    # Turn a put into the matching call price, so only one case needs solving.
    c = value if call else value + (S - K)
    # A call is worth at least its payoff and at most S; outside that range no volatility fits, so clip just inside.
    c = np.clip(c, np.maximum(S - K, 0.0) + 1e-12, S - 1e-12)
    lo = np.full(np.broadcast(c, tau).shape, lo)
    hi = np.full_like(lo, hi)
    for _ in range(iterations):
        # The call price rises with volatility, so a price above target means the volatility guess is too high.
        mid = 0.5 * (lo + hi)
        above = price(S, K, tau, mid) > c
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def derman_kamal_std(S, K, T, sigma, n_steps):
    """Standard deviation of the discrete delta-hedging error of an at-the-money option (Derman–Kamal):
    sqrt(pi/4) * vega * sigma / sqrt(N).

    N = n_steps is the number of rebalances over the option's life T (years). The error halves when N is multiplied
    by four. Output in the same price units as S.
    """
    return np.sqrt(np.pi / 4.0) * vega(S, K, T, sigma) * sigma / np.sqrt(n_steps)


def leland_sigma(sigma, cost, dt):
    """Leland's volatility for the seller of an option, hedging every dt with a one-way proportional cost.

    Inputs: sigma annual volatility, cost as a fraction of traded value (5 bp = 0.0005), dt years between
    rebalances. Output: the raised volatility to use in delta(). The seller hedges as if volatility were higher,
    which charges for the costs up front and smooths the trading.
    """
    # 2 * cost is the round-trip cost (buy then sell).
    return sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * 2.0 * cost / (sigma * np.sqrt(dt)))


def whalley_wilmott_half_width(S, gamma_, cost, risk_aversion):
    """Half-width of the no-trade band around delta, in units of the underlying per option (rate 0).

    Inputs: S price, gamma_ the option's gamma, cost one-way proportional cost (fraction), risk_aversion the
    hedger's dislike of risk (higher means a narrower band). The band is wider when costs are high and also where
    gamma is large: there delta itself moves a lot, and chasing it every step would cost too much.
    """
    return (1.5 * cost * S * gamma_**2 / risk_aversion) ** (1.0 / 3.0)
