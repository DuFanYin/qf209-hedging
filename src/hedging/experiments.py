"""The experiments behind the notebooks. Each returns a table; save() writes it to results/, the data set delivered
with the report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hedging import deep, generator, hedge, worlds

RESULTS = Path(__file__).resolve().parents[2] / "results"
COSTS_BP = [0, 1, 5, 10]  # proportional cost on the underlying, basis points of traded value
OPTION_COST = 0.01  # the second option's half spread, relative to its price
TRANSFER_COST = 5e-4  # cost on the underlying in the transfer and decomposition experiments: 5 bp


def save(table, name):
    """Writes the table to results/<name>.csv and returns it unchanged (so a notebook cell can save and show it)."""
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{name}.csv", index=False)
    return table


def grid(m, base_per_day, per_day, costs_bp=COSTS_BP, strategies=hedge.classical, world=""):
    """Every strategy at every rebalancing frequency (per day) and cost level, on the same paths.

    Inputs: m a Market simulated with base_per_day steps a day, per_day the frequencies to test (each must divide
    base_per_day), costs_bp cost levels in basis points, strategies a function cost -> {name: strategy}, world a
    label. Output: a DataFrame, one row per (frequency, cost, strategy), with the metrics of hedge.metrics.
    """
    tables = []
    for n in per_day:
        # Thin the fine grid to n steps a day; the scenarios stay identical across frequencies.
        mm = m.every(base_per_day // n)
        for bp in costs_bp:
            chosen = strategies(bp / 1e4)
            if mm.model_delta is not None:
                chosen = {**chosen, "model_delta": hedge.model_delta}
            # The second option is free to trade only in the zero-cost case.
            tables.append(hedge.evaluate(mm, chosen, bp / 1e4, OPTION_COST if bp else 0.0,
                                         world=world, per_day=n, cost_bp=bp))
    return pd.concat(tables, ignore_index=True)


def w1(n_paths=10_000, seed=1):
    """W1 on a 5-minute grid (78 a day), hedged daily up to every 5 minutes.

    Returns (market, table of grid()).
    """
    m = worlds.black_scholes(n_paths, 78, rng=np.random.default_rng(seed))
    return m, grid(m, 78, [1, 2, 6, 13, 26, 78], world="W1 Black-Scholes")


def w2(n_paths=5_000, seed=2):
    """W2 on an hourly grid (6 a day), hedged daily up to hourly; every path starts at the long-run variance.

    Returns (market, table of grid()).
    """
    model = worlds.HESTON
    m = worlds.stochastic_vol(model, n_paths, 6, rng=np.random.default_rng(seed), v0=model.theta)
    return m, grid(m, 6, [1, 2, 3, 6], world="W2 Heston")


def training_worlds(close, atm_iv, gen_model, n_paths=20_000, seed=10, history_end=None):
    """The worlds the deep hedger learns in, all daily and all fitted to the same history: W1 at the history's
    mean implied volatility, W2 calibrated to it, W3 the generator; and the reference market itself, the
    upper bound (only knowable in a simulation).

    Inputs: close and atm_iv the daily history, gen_model a fitted generator, n_paths scenarios per world.
    Output: dict {world name: Market}.
    """
    rng = np.random.default_rng(seed)
    S, iv = generator.sample(gen_model, close, atm_iv, n_paths, worlds.DAYS, seed=seed, history_end=history_end)
    return {
        "W1 Black-Scholes": worlds.black_scholes(n_paths, 1, sigma=float(np.mean(atm_iv)), rng=rng),
        "W2 Heston (fitted)": worlds.stochastic_vol(worlds.calibrate_heston(close, atm_iv), n_paths, 1, rng=rng,
                                                    with_model_delta=False),
        "W3 generator": worlds.from_factor_paths(S, iv),
        "Reference (upper bound)": worlds.stochastic_vol(worlds.REFERENCE, n_paths, 1, rng=rng, substeps=8,
                                                         with_model_delta=False),
    }


def test_market(n_paths=5_000, seed=99):
    """The test: fresh daily paths of the reference market (the real market's place until W4 is loaded).

    A different seed from training, so the hedgers are scored on scenarios they have not seen.
    """
    return worlds.stochastic_vol(worlds.REFERENCE, n_paths, 1, rng=np.random.default_rng(seed), substeps=8,
                                 with_model_delta=False)


def transfer(train, test, cost=TRANSFER_COST, iterations=800, cost_option=OPTION_COST):
    """The classical hedges, and a deep hedger trained in each world, all scored on the test market.
    Returns the scores, the hedging error of every test path by strategy, and the trained policies.
    cost_option may be one number a test path (W4: each episode's own half spread).

    Inputs: train dict {name: Market} from training_worlds, test the Market to score on, cost on the underlying,
    iterations of deep training. Output: (scores DataFrame one row per strategy, errors DataFrame one column per
    strategy and one row per test path, dict {name: trained Policy}).
    """
    # Each entry is (strategy, the cost of trading the second option for it); deep hedgers never trade it.
    strategies = {name: (s, cost_option) for name, s in hedge.classical(cost).items()}
    policies = {}
    for name, m in train.items():
        policies[name] = deep.train(m, cost, iterations=iterations)
        strategies[f"deep ({name})"] = (deep.strategy(policies[name]), 0.0)
    rows, errors = [], {}
    for name, (strategy, cost_option) in strategies.items():
        errors[name], turnover = hedge.run(test, strategy, cost, cost_option)
        rows.append({"strategy": name, **hedge.metrics(errors[name], turnover)})
    return pd.DataFrame(rows), pd.DataFrame(errors), policies


def paired_bootstrap(errors, baseline, n=2000, seed=0, block=1):
    """95% intervals for each strategy's RMSE and CVaR minus the baseline's, resampling test paths (the same paths
    for both, so the comparison is paired).

    block > 1 resamples runs of `block` consecutive paths instead of single paths (a moving-block bootstrap). Use it
    when neighbouring paths are not independent: W4's real episodes start about a week apart but each lasts a
    month, so neighbours share most of their days. Simulated paths (W3) are independent, so block = 1 there.

    Inputs: errors the DataFrame of hedging errors from transfer(), baseline the column to compare against, n
    number of resamples. Output: a DataFrame, one row per other strategy: the difference in RMSE and in CVaR 95%
    (negative means better than the baseline) and the 2.5% and 97.5% percentiles of each (columns *_lo, *_hi).
    An interval that excludes zero means the difference is unlikely to be luck.
    """
    rng = np.random.default_rng(seed)
    N = len(errors)
    # idx: n rows of N path indices, each row one resampled test set.
    if block == 1:
        idx = rng.integers(0, N, (n, N))
    else:
        # Enough random block starts to cover N paths (ceil(N / block) blocks), each block expanded to its
        # `block` consecutive indices, and the result cut back to N.
        starts = rng.integers(0, N - block + 1, (n, -(-N // block)))
        idx = (starts[..., None] + np.arange(block)).reshape(n, -1)[:, :N]
    rows = []
    for name in errors.columns.drop(baseline):
        # The same resampled indices for the strategy and the baseline: that is what makes it paired.
        a, b = errors[name].to_numpy()[idx], errors[baseline].to_numpy()[idx]
        d_rmse = np.sqrt((a**2).mean(1)) - np.sqrt((b**2).mean(1))
        d_cvar = np.array([hedge.cvar(x) - hedge.cvar(y) for x, y in zip(a, b)])
        rows.append({"strategy": name, "vs": baseline,
                     "d_rmse": np.sqrt((errors[name] ** 2).mean()) - np.sqrt((errors[baseline] ** 2).mean()),
                     "d_rmse_lo": np.percentile(d_rmse, 2.5), "d_rmse_hi": np.percentile(d_rmse, 97.5),
                     "d_cvar95": hedge.cvar(errors[name].to_numpy()) - hedge.cvar(errors[baseline].to_numpy()),
                     "d_cvar95_lo": np.percentile(d_cvar, 2.5), "d_cvar95_hi": np.percentile(d_cvar, 97.5)})
    return pd.DataFrame(rows)


def decomposition(test, sigma, cost=TRANSFER_COST, n_paths=10_000, seed=3):
    """The Black–Scholes delta hedge as each source of error is added: near-continuous (5-minute) hedging in W1,
    then daily, then with costs, then in the test market (stochastic volatility, jumps, the hedger using each
    day's implied volatility).

    Inputs: test the market for the last step, sigma the W1 volatility (match it to the test market's level),
    cost on the underlying. Output: a DataFrame, one row per step, with the metrics of hedge.metrics.
    """
    m = worlds.black_scholes(n_paths, 78, sigma=sigma, rng=np.random.default_rng(seed))
    steps = [("Continuous (5-minute, W1)", m, 0.0), ("+ Discrete (daily)", m.every(78), 0.0),
             ("+ Costs", m.every(78), cost), ("+ Misspecification (test market)", test, cost)]
    rows = []
    for name, market, c in steps:
        pnl, turnover = hedge.run(market, hedge.delta, c)
        rows.append({"step": name, **hedge.metrics(pnl, turnover)})
    return pd.DataFrame(rows)
