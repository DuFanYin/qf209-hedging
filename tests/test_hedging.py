"""Unit tests: pricing against parity and closed forms, the hedging engine, and the deep hedger's accounting.
Run with `uv run pytest`."""

import numpy as np
import pytest

from hedging import bs, deep, hedge, worlds
from hedging.models import StochasticVol

K = np.array([80.0, 95.0, 100.0, 105.0, 120.0])


def test_put_call_parity_and_implied_vol_round_trip():
    """Call minus put equals S - K (rate zero), and implied_vol recovers the volatility a price was made with."""
    c = bs.price(100.0, K, 0.25, 0.3)
    p = bs.price(100.0, K, 0.25, 0.3, call=False)
    assert np.allclose(c - p, 100.0 - K)
    assert np.allclose(bs.implied_vol(c, 100.0, K, 0.25), 0.3, atol=1e-8)
    assert np.allclose(bs.implied_vol(p, 100.0, K, 0.25, call=False), 0.3, atol=1e-8)


def test_greeks_match_finite_differences():
    """Delta, gamma and vega formulas agree with repricing after small bumps to S or sigma."""
    h = 1e-4
    up, down = bs.price(100.0 + h, K, 0.25, 0.3), bs.price(100.0 - h, K, 0.25, 0.3)
    assert np.allclose(bs.delta(100.0, K, 0.25, 0.3), (up - down) / (2 * h), atol=1e-7)
    assert np.allclose(bs.gamma(100.0, K, 0.25, 0.3),
                       (up - 2 * bs.price(100.0, K, 0.25, 0.3) + down) / h**2, atol=1e-4)
    assert np.allclose(bs.vega(100.0, K, 0.25, 0.3),
                       (bs.price(100.0, K, 0.25, 0.3 + h) - bs.price(100.0, K, 0.25, 0.3 - h)) / (2 * h), atol=1e-6)


@pytest.mark.parametrize("tau", [1 / 12, 1 / 252, 1 / 252 / 78])
def test_stochastic_vol_without_vol_of_vol_is_black_scholes(tau):
    """With almost no vol of vol, Heston's price must equal Black–Scholes at 20% volatility, from a month down to
    five minutes before expiry (where the numerical integral is hardest)."""
    flat = StochasticVol(kappa=2.0, theta=0.04, xi=1e-4, rho=-0.7)
    assert np.allclose(flat.call(100.0, K, tau, 0.04), bs.price(100.0, K, tau, 0.2), atol=1e-4)


def test_bates_price_matches_monte_carlo():
    """The reference market's pricing formula agrees with the average payoff over 100,000 simulated paths, within
    four standard errors: the pricer and the simulator describe the same model."""
    model = worlds.REFERENCE
    S, _ = model.simulate(100.0, model.theta, 1 / 12, 21, 100_000, np.random.default_rng(1), substeps=8)
    for k in (95.0, 100.0, 105.0):
        payoff = np.maximum(S[:, -1] - k, 0.0)
        assert abs(model.call(100.0, k, 1 / 12, model.theta) - payoff.mean()) < 4 * payoff.std() / np.sqrt(S.shape[0])


def test_discrete_delta_hedging_error_follows_derman_kamal():
    """In W1 without costs, the spread of the delta hedge's error matches Derman–Kamal within 6%, hedging six
    times a day and daily."""
    m = worlds.black_scholes(20_000, 6, rng=np.random.default_rng(3))
    for k in (6, 1):
        mm = m.every(k)
        pnl, _ = hedge.run(mm, hedge.delta)
        expected = bs.derman_kamal_std(100.0, 100.0, worlds.T, 0.2, mm.n_steps)
        assert abs(pnl.std() / expected - 1) < 0.06


def test_costs_lower_the_mean_by_the_cost_of_turnover():
    """Adding a proportional cost lowers each path's result by exactly cost * traded value (turnover * S0)."""
    m = worlds.black_scholes(2_000, 1, rng=np.random.default_rng(4))
    free, turnover = hedge.run(m, hedge.delta)
    costly, _ = hedge.run(m, hedge.delta, cost=0.001)
    assert np.allclose(free - costly, 0.001 * turnover * 100.0)


def test_untrained_deep_policy_is_black_scholes_delta():
    """An untrained deep hedger is exactly the Black–Scholes delta hedge, and its PyTorch accounting (deep.pnl)
    gives the same results as the main engine (hedge.run)."""
    m = worlds.black_scholes(500, 1, rng=np.random.default_rng(5))
    policy = deep.Policy().double()
    expected, _ = hedge.run(m, hedge.delta, cost=0.0005)
    assert np.allclose(deep.pnl(policy, m, 0.0005).detach().numpy(), expected)
    assert np.allclose(hedge.run(m, deep.strategy(policy), cost=0.0005)[0], expected)
