# Non-technical summary

When a bank sells an option it takes on a risk. Take a call on the S&P 500: if the index rises over the month, the bank pays the client the rise, and it has been paid a premium up front for that. To avoid a large loss when the index rallies, the bank hedges by holding some of the index itself and adjusting the amount as the index moves.

Black–Scholes says this hedge can be exact, but only if the bank can adjust it continuously, at no cost, and knows the right model for how prices move. None of that is true in practice. This project measures how much each of the three costs, and compares hedging methods.

We sell a one-month option, hedge it to expiry and record the hedging error: what the bank is left with after collecting the premium, paying the client and running the hedge. A perfect hedge leaves zero. We look at the typical size of the error and at the average loss in the worst 5% of cases. The experiment is repeated in four settings: a textbook market where Black–Scholes is exactly right, a market where volatility moves randomly, a market simulated by a neural network trained on price history, and real S&P 500 data from 2005 to 2025. The simulated settings let us switch on one problem at a time; the real data checks the results.

The methods compared are the textbook delta hedge, two variants that trade less often to save costs, two hedges that also hold a second, longer-dated option (one aimed at gamma, one at volatility, or vega), and a neural network that learns how much of the index to hold.

The main findings:

- Hedging once a day instead of continuously adds error in exactly the way theory predicts.
- With trading costs there is a best rebalancing frequency, and it falls as costs rise: about 26 times a day at a cost of 0.01% per trade, 6 at 0.05% and twice at 0.1%.
- Using the wrong model does the most damage, and mostly in bad months. On real data the daily delta hedge's typical error is 0.64% of the index, against 0.37% if the textbook model were right, and its average loss in the worst 5% of trades is 1.94% against 0.96%.
- Also hedging volatility with the second option works best everywhere. On real data, using the second option's actual market prices and trading costs, it cuts the typical error from 0.64% to 0.36%. Hedging gamma instead helps little, because it leaves the volatility risk in place.
- A neural-network hedger is only as good as the market it is trained in. Trained in the textbook market, it does worse than the delta hedge on real data. Trained in the neural-network-simulated market, it is the best of the methods that trade only the index, with a clearly smaller worst-case loss (1.52% against 1.94%). Six years of test data are not enough to tell whether its typical error is also lower.

The main limits are that the real data is daily, so the question of how often to hedge can only be studied in simulation, and that there is only one real history: the 291 test trades overlap and carry about as much information as 70 independent months.

The full results are in [REPORT.md](REPORT.md).
