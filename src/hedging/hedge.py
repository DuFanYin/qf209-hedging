"""The hedging engine and the classical strategies. The trader sells the option at the world's price at t[0], trades
the underlying (and the second option) at every time on the grid, and settles at T.

A strategy is a function (market, i, phi, psi) -> (phi, psi): the new holdings of the underlying and of the second
option at time i, given the holdings before. Costs are proportional: `cost` on the underlying's traded value,
`cost_option` on the second option's.

Units: hedging errors are in price units (S0 = 100; the 30-day at-the-money call costs about 2.3). Costs are
fractions of traded value (5 bp = 0.0005).
"""

import numpy as np
import pandas as pd

from hedging import bs

CVAR_LEVEL = 0.95


def run(m, strategy, cost=0.0, cost_option=0.0):
    """Hedging error per path (premium - payoff + trading gains - costs) and turnover (traded underlying value
    per unit of S0).

    Inputs: m a Market, strategy as described in the module docstring, cost the proportional cost on the
    underlying (fraction), cost_option the proportional cost on the second option (one number, or an array with
    one value per path). Output: (pnl, turnover), two arrays (n_paths,). Positive pnl is a gain for the trader.
    """
    # Start flat (no hedge) with the premium in hand. Rate zero, so cash earns nothing and needs no tracking beyond
    # this running total.
    phi = np.zeros(m.n_paths)
    psi = np.zeros(m.n_paths)
    pnl = m.V[:, 0].copy()
    turnover = np.zeros(m.n_paths)
    for i in range(m.n_steps):
        # 1. Decide the new holdings at time i, using only what is known at time i.
        new_phi, new_psi = strategy(m, i, phi, psi)
        new_psi = np.broadcast_to(new_psi, phi.shape)
        # 2. Pay the costs of moving from the old holdings to the new ones, at today's prices.
        traded = np.abs(new_phi - phi) * m.S[:, i]
        pnl -= cost * traded + cost_option * np.abs(new_psi - psi) * m.H[:, i]
        turnover += traded / m.S[:, 0]
        phi, psi = new_phi, new_psi
        # 3. Hold until the next time and collect the gain or loss on both positions.
        pnl += phi * (m.S[:, i + 1] - m.S[:, i]) + psi * (m.H[:, i + 1] - m.H[:, i])
    # At expiry, close both hedge positions (paying costs once more) and pay the option's payoff V[:, -1].
    pnl -= cost * np.abs(phi) * m.S[:, -1] + cost_option * np.abs(psi) * m.H[:, -1]
    turnover += np.abs(phi) * m.S[:, -1] / m.S[:, 0]
    return pnl - m.V[:, -1], turnover


def cvar(pnl, level=CVAR_LEVEL):
    """Expected loss in the worst (1 - level) of paths, as a positive number.

    Input: pnl array of hedging errors. Output: a number; with level 0.95, the mean loss over the worst 5%.
    """
    loss = np.sort(-pnl)
    return loss[int(np.floor(level * loss.size)):].mean()


def metrics(pnl, turnover):
    """Summary of one strategy's results: RMSE, CVaR 95%, mean and standard deviation of the hedging error, and
    mean turnover. Inputs are the two arrays returned by run(); output is a dict of floats."""
    return {"rmse": float(np.sqrt(np.mean(pnl**2))), "cvar95": float(cvar(pnl)), "mean": float(pnl.mean()),
            "std": float(pnl.std()), "turnover": float(turnover.mean())}


def evaluate(m, strategies, cost=0.0, cost_option=0.0, **labels):
    """One row of metrics per strategy.

    Inputs: m a Market, strategies a dict {name: strategy}, costs as in run(), labels extra columns copied into
    every row (e.g. world="W1", cost_bp=5). Output: a pandas DataFrame, one row per strategy.
    """
    rows = []
    for name, strategy in strategies.items():
        pnl, turnover = run(m, strategy, cost, cost_option)
        rows.append({**labels, "strategy": name, **metrics(pnl, turnover)})
    return pd.DataFrame(rows)


# The classical strategies. Greeks are Black–Scholes at each option's own implied volatility (the practitioner's
# hedge); `model_delta` uses the world's own model where it has one.
# Each takes (m, i, phi, psi) and returns (new phi, new psi); see the module docstring.

def _greeks(m, i):
    """Inputs the greeks need at time i: the underlying's prices, and the time left on each option (years)."""
    tau, tau2 = m.T - m.t[i], m.T2 - m.t[i]
    S = m.S[:, i]
    return S, tau, tau2


def unhedged(m, i, phi, psi):
    """No hedge at all: hold nothing. A baseline."""
    return np.zeros_like(phi), 0.0


def delta(m, i, phi, psi):
    """Black–Scholes delta hedge at the option's current implied volatility, rebalanced at every step."""
    S, tau, _ = _greeks(m, i)
    return bs.delta(S, m.K, tau, m.iv[:, i], m.call), 0.0


def model_delta(m, i, phi, psi):
    """Delta from the world's own model (W2: Heston), where the Market carries one."""
    return m.model_delta[:, i], 0.0


def leland(cost):
    """Leland's hedge for this cost level: Black–Scholes delta at Leland's raised volatility. Returns a strategy."""
    def strategy(m, i, phi, psi):
        S, tau, _ = _greeks(m, i)
        # The raise depends on how often the hedge is rebalanced, i.e. on the grid's step.
        dt = m.t[i + 1] - m.t[i]
        return bs.delta(S, m.K, tau, bs.leland_sigma(m.iv[:, i], cost, dt), m.call), 0.0
    return strategy


def whalley_wilmott(cost, risk_aversion=1.0):
    """Trade only when the holding leaves the no-trade band around delta, and then only to its edge.

    Returns a strategy for this cost level and risk aversion.
    """
    def strategy(m, i, phi, psi):
        S, tau, _ = _greeks(m, i)
        target = bs.delta(S, m.K, tau, m.iv[:, i], m.call)
        half = bs.whalley_wilmott_half_width(S, bs.gamma(S, m.K, tau, m.iv[:, i]), cost, risk_aversion)
        # At the start, buy delta outright; after that, clipping to [delta - half, delta + half] leaves the holding
        # unchanged inside the band and moves it only to the nearest edge outside it.
        start = i == 0
        return (target if start else np.clip(phi, target - half, target + half)), 0.0
    return strategy


# Below these the second option (far from the money) has no gamma or vega to give, and the ratio of greeks
# explodes: it is not held. About a thousandth of an at-the-money option's (S = 100).
MIN_GAMMA, MIN_VEGA = 1e-4, 1e-2


def _ratio(mine, theirs, floor):
    """mine / theirs where theirs is above floor, else 0: how many second options offset the hedged option's greek."""
    return np.where(theirs > floor, mine / np.maximum(theirs, floor), 0.0)


def delta_gamma(m, i, phi, psi):
    """Hold enough of the second option to cancel the sold option's gamma, then the underlying to cancel delta."""
    S, tau, tau2 = _greeks(m, i)
    psi = _ratio(bs.gamma(S, m.K, tau, m.iv[:, i]), bs.gamma(S, m.K2, tau2, m.iv2[:, i]), MIN_GAMMA)
    # The second option brings its own delta, so the underlying only covers what is left.
    return bs.delta(S, m.K, tau, m.iv[:, i], m.call) - psi * bs.delta(S, m.K2, tau2, m.iv2[:, i]), psi


def delta_vega(m, i, phi, psi):
    """Hold enough of the second option to cancel the sold option's vega, then the underlying to cancel delta."""
    S, tau, tau2 = _greeks(m, i)
    psi = _ratio(bs.vega(S, m.K, tau, m.iv[:, i]), bs.vega(S, m.K2, tau2, m.iv2[:, i]), MIN_VEGA)
    return bs.delta(S, m.K, tau, m.iv[:, i], m.call) - psi * bs.delta(S, m.K2, tau2, m.iv2[:, i]), psi


def classical(cost):
    """The textbook strategies, with the cost-aware ones set for this cost level.

    Input: cost as a fraction. Output: dict {name: strategy}; Leland and Whalley–Wilmott are included only when
    cost > 0 (without costs they are the plain delta hedge).
    """
    out = {"delta": delta, "delta_gamma": delta_gamma, "delta_vega": delta_vega}
    if cost > 0:
        out["leland"] = leland(cost)
        out["whalley_wilmott"] = whalley_wilmott(cost)
    return out
