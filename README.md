# QF209 A5: Derivative Hedging Strategies

We sell a one-month at-the-money call and hedge it to expiry, and measure how much of the hedging error comes from hedging at discrete times, from transaction costs, and from using the wrong model. The same experiment runs in four settings: Black–Scholes (W1), Heston (W2), paths from a neural-network generator fitted to a price history (W3), and real S&P 500 options from 2005 to 2025 (W4). The report is in [report/REPORT.md](report/REPORT.md); [report/PROJECT_EXPLAINED.md](report/PROJECT_EXPLAINED.md) is a short non-technical summary.

Main results:

1. Without costs, the delta hedge's error falls with the square root of the rebalancing frequency, within 3% of the Derman–Kamal formula.
2. With costs there is an optimal frequency, which falls as costs rise (about 26, 6 and 2 rebalances a day at 1, 5 and 10 bp).
3. Model misspecification is the largest source of error in the tail. On real SPX, daily delta hedging has an RMSE of 0.64% of the index and a CVaR 95% of 1.94%, against 0.37% and 0.96% in the Black–Scholes world at the same frequency and costs.
4. Delta–vega hedging with a second, longer option gives the lowest error in every setting; on real quotes, 0.36% against 0.64% for delta. Delta–gamma helps much less.
5. Deep hedgers trained in the generator's world beat those trained in W1 or W2. On real data the W3 hedger has a clearly lower CVaR than delta (1.52% against 1.94%); its lower RMSE (0.54% against 0.64%) is not significant with six years of test data.

## Layout

```
report/REPORT.md            report
report/PROJECT_EXPLAINED.md non-technical summary
notebooks/01–04             one notebook per world, run in order
src/hedging/
    bs.py                   Black–Scholes prices, greeks, implied volatility, closed-form benchmarks
    models.py               GBM, Heston and Bates paths; Heston/Bates pricing
    worlds.py               builds each world as a Market (underlying, option and hedge-option paths)
    hedge.py                hedging engine and classical strategies
    deep.py                 deep hedger
    generator.py            mixture density network generator (W3)
    real.py                 SPX data loading and episode construction (W4)
    experiments.py          experiment runners used by the notebooks
scripts/prepare_wrds.py     converts the WRDS download into the CSV files real.py reads
results/                    tables (CSV) and figures (PNG)
tests/                      unit tests
data/                       WRDS data, not in the repository
```

## Running

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv run pytest
uv run jupyter lab
```

Notebooks 01–03 are simulated with fixed seeds and need no data. Each takes a few minutes. To run everything from the command line (this rewrites `results/`):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Notebook 04 needs the WRDS data. Unzip `spx_wrds_data.zip` and `options_0_70dte_band30.zip` into `data/raw/`, then:

```bash
uv run python scripts/prepare_wrds.py data/raw/spx_wrds_data data/raw/options_0_70dte_band30
```

This writes `data/history.csv`, `data/options.csv` and `data/surface.csv` (about 1.2 GB) in under a minute. Without the second folder the script falls back to the first download's 20–50-day options, and the hedge option is then priced from the volatility surface.

## Data (W4)

WRDS, OptionMetrics IvyDB US, S&P 500 index (secid 108105), 3 January 2005 to 29 August 2025.

| File | Contents | Used for |
|---|---|---|
| `spx_prices.parquet` | daily open and close | index path, settlement prices |
| `spx_options_YYYY.parquet` | option quotes, 20–50 days to expiry | replaced by the second download |
| `options_0_70dte_band30/spx_options_YYYY.parquet` | option quotes (`optionm.opprcdYYYY`: date, exdate, cp_flag, strike_price, best_bid, best_offer, impl_volatility, am_settlement), 0–70 days to expiry, strikes within ±30% of the index | premium, implied volatility of both options, hedge-option prices and spreads |
| `spx_volsurface_YYYY.parquet` | OptionMetrics volatility surface, 10–730 days | days without a quote |
| `zero_curve.parquet`, `spx_dividend_yield.parquet` | rates and dividend yield | forward prices |
| `spx_forwards.parquet` | forwards by expiry | not used |

The ±30% strike band keeps the hedged option in the table after large falls (2008, March 2020).

Known gaps:

- The hedged option has no quote on 6% of its days and the hedge option on 3.5% (1.5% in the test period); those days use the surface, which agrees with quotes within 0.14 volatility points at the start (median). 24 trades (4 in the test period) have no suitable listed hedge option and use one priced from the surface.
- Before about 2012 SPX had only monthly expiries, so there are 12–16 trades a year then against 45–50 after.
- No intraday data, so the rebalancing-frequency results are simulation only. No futures data: we hedge with the index forward at an assumed 5 bp.
- The surface is missing on 17 days, the 30-day volatility on 17 and the rate curve on 1 (filled forward). The dividend yield looks wrong on 54 days in early 2008 (left as is; it is in the training period).
- There is one history. The 291 test trades overlap (weekly starts, one-month life), so we use a block bootstrap and the real-data intervals are wide.
