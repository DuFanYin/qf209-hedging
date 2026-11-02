"""Deep hedging (Buehler et al., 2019): a small network sets the holding of the underlying at each step, trained to
minimise the CVaR of the hedging error after costs. It starts from Black–Scholes delta and learns a correction.

Inputs at each step: time left (fraction of T), log-moneyness scaled by iv * sqrt(T), the option's implied
volatility, Black–Scholes delta, and the holding before the trade (so it can learn not to trade inside a band).

Units: as in the rest of the project, prices with S0 = 100, time in years, costs as fractions.
"""

import numpy as np
import torch
from torch import nn

from hedging.hedge import CVAR_LEVEL

# Cap the CPU threads PyTorch uses, so training runs predictably on a laptop.
torch.set_num_threads(4)


class Policy(nn.Module):
    """The hedging network: holding = Black–Scholes delta + a learned correction.

    hidden is the width of each of the two hidden layers. The last layer starts at zero, so an untrained policy
    is exactly the Black–Scholes delta hedge.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        # Zero output layer: the correction is zero until training moves it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.w = nn.Parameter(torch.zeros(()))  # the VaR level in the CVaR objective (Rockafellar–Uryasev)

    def forward(self, S, K, tau, T, iv, phi, call=True):
        """New holding of the underlying for each path at one time step.

        Inputs (torch tensors of shape (n_paths,) unless noted): S prices, K strike (number), tau years left
        (0-d tensor), T the option's life in years (number), iv implied volatilities, phi the holdings before the
        trade, call or put. Output: tensor (n_paths,) of new holdings.
        """
        # Black–Scholes delta, written again in PyTorch so gradients flow through it.
        sd = torch.clamp(iv * torch.sqrt(torch.clamp(tau, min=1e-12)), min=1e-8)
        bs_delta = torch.special.ndtr((torch.log(S / K) + 0.5 * sd * sd) / sd)
        bs_delta = torch.where(tau > 0, bs_delta, (S > K).to(S.dtype))
        if not call:
            bs_delta = bs_delta - 1.0
        # The five features, each roughly of order one so the network trains well.
        x = torch.stack([tau / T * torch.ones_like(S), torch.log(S / K) / (iv * np.sqrt(T)), iv, bs_delta, phi], -1)
        return bs_delta + self.net(x).squeeze(-1)


def _tensors(m, rows=None):
    """The Market's prices, implied volatilities, premium and payoff for the chosen rows, as float64 tensors."""
    rows = slice(None) if rows is None else rows
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a[rows]), dtype=torch.float64)
    return f(m.S), f(np.asarray(m.iv)), f(m.V[:, :1]).squeeze(1), f(m.V[:, -1:]).squeeze(1)


def pnl(policy, m, cost, rows=None):
    """Hedging error per path, differentiable; the same accounting as hedge.run without the second option.

    Inputs: policy a Policy, m a Market, cost proportional cost (fraction), rows optional path indices (a
    training batch). Output: tensor (n_rows,) of hedging errors; positive is a gain.
    """
    S, iv, premium, payoff = _tensors(m, rows)
    phi = torch.zeros(S.shape[0], dtype=S.dtype)
    out = premium.clone()
    for i in range(m.n_steps):
        tau = torch.tensor(m.T - m.t[i], dtype=S.dtype)
        new = policy(S[:, i], m.K, tau, m.T, iv[:, i], phi, m.call)
        # Pay the cost of the trade, then collect the gain on the new holding until the next step.
        out = out - cost * torch.abs(new - phi) * S[:, i] + new * (S[:, i + 1] - S[:, i])
        phi = new
    # Close the hedge at expiry (one more cost) and pay the option's payoff.
    return out - cost * torch.abs(phi) * S[:, -1] - payoff


def train(m, cost, iterations=800, batch=4096, lr=3e-3, seed=0, hidden=64):
    """Trains on the market's paths; returns the policy in float64.

    Inputs: m the training Market, cost proportional cost, iterations number of gradient steps, batch paths per
    step, lr learning rate, seed for reproducibility, hidden network width. Output: a trained Policy.
    """
    torch.manual_seed(seed)
    policy = Policy(hidden).double()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    # The learning rate falls smoothly to zero over training (cosine schedule), which settles the final weights.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        rows = rng.choice(m.n_paths, size=min(batch, m.n_paths), replace=False)
        loss = -pnl(policy, m, cost, rows)
        # Rockafellar–Uryasev: minimising this over the network and w together minimises the CVaR of the loss.
        objective = policy.w + torch.relu(loss - policy.w).mean() / (1.0 - CVAR_LEVEL)
        opt.zero_grad()
        objective.backward()
        opt.step()
        sched.step()
    return policy.eval()


def strategy(policy):
    """The trained policy as a strategy for hedge.run.

    Output: a function (m, i, phi, psi) -> (phi, 0.0); the deep hedger trades the underlying only.
    """
    def step(m, i, phi, psi):
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
        with torch.no_grad():
            new = policy(t(m.S[:, i]), m.K, torch.tensor(m.T - m.t[i], dtype=torch.float64), m.T,
                         t(np.asarray(m.iv)[:, i]), t(phi), m.call)
        return new.numpy(), 0.0
    return step
