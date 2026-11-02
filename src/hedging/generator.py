"""W3's generator: a recurrent mixture density network. It learns the daily joint law of the underlying's log return
and the change in log 30-day ATM implied volatility, given the past, from one daily history (the reference market
now, the real one through hedging.real), and samples new paths from it.

Shapes: histories are 1-D arrays of daily values; generated paths are arrays (n_paths, n_days + 1).
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

torch.set_num_threads(4)
LOG_2PI = float(np.log(2.0 * np.pi))


class Generator(nn.Module):
    """GRU plus a mixture-of-Gaussians output layer.

    hidden is the size of the GRU's memory, n_mix the number of mixture components. The buffers loc and scale
    store the training data's mean and standard deviation (per feature), so the model can standardise its inputs
    and turn its outputs back into returns and volatility changes.
    """

    def __init__(self, hidden=32, n_mix=4):
        super().__init__()
        self.n_mix = n_mix
        # Input each day: (log return, change in log iv, log iv level).
        self.gru = nn.GRU(3, hidden, batch_first=True)
        # Six numbers per component: weight, two means, two log standard deviations, one correlation.
        self.head = nn.Linear(hidden, n_mix * 6)
        self.register_buffer("loc", torch.zeros(3))
        self.register_buffer("scale", torch.ones(3))

    def _mixture(self, h):
        """Turns the GRU state into the mixture's parameters: log weights, means, log standard deviations and
        correlations, each with a component axis of size n_mix."""
        p = self.head(h).unflatten(-1, (self.n_mix, 6))
        # Weights must be positive and sum to one (log-softmax gives their logarithms).
        logits = torch.log_softmax(p[..., 0], -1)
        # Standard deviations kept in a sane range, and correlation kept inside (-0.95, 0.95), for stability.
        mean, log_sd = p[..., 1:3], torch.clamp(p[..., 3:5], -7.0, 3.0)
        rho = 0.95 * torch.tanh(p[..., 5])
        return logits, mean, log_sd, rho

    def log_prob(self, h, y):
        """Log-likelihood of the observed next-day values y (standardised return and iv change, last axis of size
        2) under the mixture predicted from the GRU state h. Output has y's shape without the last axis."""
        logits, mean, log_sd, rho = self._mixture(h)
        # Log-density of a two-dimensional Gaussian with correlation rho, for every component at once.
        z = (y.unsqueeze(-2) - mean) / torch.exp(log_sd)
        one = 1.0 - rho * rho
        q = (z[..., 0] ** 2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / one
        log_p = -LOG_2PI - log_sd.sum(-1) - 0.5 * torch.log(one) - 0.5 * q
        # Mixture density = sum over components of weight * density, computed in logs to avoid underflow.
        return torch.logsumexp(logits + log_p, -1)

    def sample_step(self, h, gen):
        """Draws one day for each path: tensor (n_paths, 2) of standardised (return, change in log iv).

        h is the GRU state, shape (n_paths, hidden); gen a torch random Generator.
        """
        logits, mean, log_sd, rho = self._mixture(h)
        # Two-stage draw: first pick a component for each path according to the weights...
        k = torch.multinomial(torch.exp(logits), 1, generator=gen).squeeze(-1)
        pick = lambda a: a[torch.arange(a.shape[0]), k]
        m, sd, r = pick(mean), torch.exp(pick(log_sd)), pick(rho)
        # ...then draw from that component's bell curve. Two independent normals e1, e2 are combined so the second
        # coordinate has correlation r with the first.
        e1, e2 = torch.randn(m.shape, generator=gen, dtype=m.dtype).unbind(-1)
        return torch.stack([m[:, 0] + sd[:, 0] * e1,
                            m[:, 1] + sd[:, 1] * (r * e1 + torch.sqrt(1 - r * r) * e2)], -1)


def _inputs(close, atm_iv):
    """Per day t >= 1: (log return, change in log iv, log iv); the target of day t is (return, change) of t + 1.

    Output: array (n_days - 1, 3); row j describes day j + 1 of the history.
    """
    r = np.diff(np.log(close))
    a = np.log(atm_iv)
    return np.stack([r, np.diff(a), a[1:]], -1)


def fit(close, atm_iv, hidden=32, n_mix=4, epochs=300, window=128, batch=64, lr=3e-3, val_fraction=0.2, seed=0):
    """Fits on the first part of the history, keeps the state with the best likelihood on the rest.
    Returns the model and the per-epoch (train, validation) negative log-likelihoods.

    Inputs: close and atm_iv 1-D daily arrays (prices; implied volatility as a decimal), hidden and n_mix the model
    size, epochs number of training steps (each one batch, not a full pass over the data), window days per training
    sequence, batch sequences per step, lr learning rate, val_fraction share of days held out at the end.
    Output: (model, history DataFrame with columns train_nll and val_nll, one row per step).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = _inputs(np.asarray(close, float), np.asarray(atm_iv, float))
    cut = int(len(x) * (1 - val_fraction))
    model = Generator(hidden, n_mix).double()
    # Standardise with training-period statistics only, so nothing leaks in from the validation days.
    model.loc[:] = torch.as_tensor(x[:cut].mean(0))
    model.scale[:] = torch.as_tensor(x[:cut].std(0))
    xs = (torch.as_tensor(x) - model.loc) / model.scale
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, history = np.inf, None, []
    for _ in range(epochs):
        model.train()
        # A batch of random windows from the training period; each day predicts the next day's return and iv change.
        starts = rng.integers(0, cut - window - 1, batch)
        seq = torch.stack([xs[s:s + window + 1] for s in starts])
        h, _ = model.gru(seq[:, :-1])
        loss = -model.log_prob(h, seq[:, 1:, :2]).mean()
        opt.zero_grad()
        loss.backward()
        # Limit the size of each update; recurrent networks otherwise occasionally take a destructive step.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            # Run through the whole history once and score every day's prediction; the first cut - 1 predictions
            # are of training days, the rest of validation days.
            h, _ = model.gru(xs[None, :-1])
            nll = -model.log_prob(h, xs[None, 1:, :2])[0]
            train_nll, val_nll = float(nll[:cut - 1].mean()), float(nll[cut - 1:].mean())
        history.append((train_nll, val_nll))
        # Early stopping: remember the weights that predicted the validation days best.
        if val_nll < best:
            best, best_state = val_nll, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history, columns=["train_nll", "val_nll"])


def sample(model, close, atm_iv, n_paths, n_days, context=60, seed=0, S0=100.0, history_end=None):
    """n_paths paths of n_days, each continuing a random context window of the history (up to history_end).
    Returns (S, atm_iv), shape (n_paths, n_days + 1), S starting at S0.

    Inputs: model a fitted Generator, close and atm_iv the daily history, context days of history read before
    generating (they set the starting volatility regime), seed for reproducibility, S0 the starting price,
    history_end an optional cut-off (a row index of _inputs, i.e. one less than the day's index in close) so that
    no context comes from after it. atm_iv starts at the history's level at the end of the context window.
    """
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = _inputs(np.asarray(close, float), np.asarray(atm_iv, float))
    end = len(x) if history_end is None else history_end
    starts = rng.integers(0, end - context, n_paths)
    xs = (torch.as_tensor(x) - model.loc) / model.scale
    ctx = torch.stack([xs[s:s + context] for s in starts])
    # Log iv on the last day of each context window: the starting volatility of each new path.
    a = torch.as_tensor(x[starts + context - 1, 2])
    r_out, a_out = [], [a]
    with torch.no_grad():
        # Read the context windows to set the GRU's memory.
        _, h = model.gru(ctx)
        for _ in range(n_days):
            # Draw tomorrow in standardised units and convert back to a log return and a change in log iv.
            y = model.sample_step(h[-1], gen) * model.scale[:2] + model.loc[:2]
            a = a + y[:, 1]
            r_out.append(y[:, 0])
            a_out.append(a)
            # Feed the drawn day back in, so the next day depends on the simulated past, not the real one.
            step = (torch.stack([y[:, 0], y[:, 1], a], -1) - model.loc) / model.scale
            _, h = model.gru(step[:, None], h)
    r = torch.stack(r_out, 1).numpy()
    # Prices from cumulated log returns, starting at S0.
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(r, 1)]))
    return S, np.exp(torch.stack(a_out, 1).numpy())


def stylized_facts(close, atm_iv):
    """The facts a generator of equity-index paths should reproduce, for one series (1-D) or many paths (2-D,
    statistics pooled over paths).

    Output: a dict:
        vol_annual              standard deviation of daily log returns, annualised
        skew, excess_kurtosis   asymmetry and fat-tailedness of daily returns (0 for a bell curve)
        acf_abs_return_1, _10   autocorrelation of |return| at lags 1 and 10 (volatility clustering)
        corr_return_iv_change   correlation of returns with changes in log iv (leverage effect; negative for SPX)
        iv_mean                 average implied volatility
        iv_change_sd            standard deviation of daily changes in log iv
        acf_iv_change_1         autocorrelation of daily changes in log iv at lag 1
    """
    close, atm_iv = np.atleast_2d(close), np.atleast_2d(atm_iv)
    r = np.diff(np.log(close), axis=1)
    da = np.diff(np.log(atm_iv), axis=1)

    def acf(x, lag):
        """Autocorrelation at `lag`, pooled over rows; each row is demeaned separately."""
        x = x - x.mean(1, keepdims=True)
        total = (x * x).sum()
        return float((x[:, lag:] * x[:, :-lag]).sum() / total) if total > 0 else float("nan")

    def corr(a, b):
        """Correlation, or nan when either series is constant (e.g. W1's flat implied volatility)."""
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    return {
        "vol_annual": float(r.std() * np.sqrt(252)),
        "skew": float(stats.skew(r.ravel())),
        "excess_kurtosis": float(stats.kurtosis(r.ravel())),
        "acf_abs_return_1": acf(np.abs(r), 1),
        "acf_abs_return_10": acf(np.abs(r), 10),
        "corr_return_iv_change": corr(r.ravel(), da.ravel()),
        "iv_mean": float(atm_iv.mean()),
        "iv_change_sd": float(da.std()),
        "acf_iv_change_1": acf(da, 1),
    }
