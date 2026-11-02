"""Path simulators and the stochastic-volatility pricer: geometric Brownian motion, Heston, and Bates (Heston with
Merton jumps). Prices are forward prices (rate 0); the drift of the paths is a separate, real-world parameter.

Shapes: simulated paths are arrays (n_paths, n_steps + 1), one row per path, column 0 the start. Prices are in the
same units as S0 (100 in the project).
"""

from dataclasses import dataclass

import numpy as np

from hedging import bs

# Gauss–Legendre nodes on [0, pi/2] for the Lewis integral after u = tan(theta) / 2.
# The integral runs from 0 to infinity in u; the change of variable maps it to the finite range [0, pi/2] in theta,
# where 192 fixed nodes and weights compute it accurately. They are computed once, at import.
_X, _W = np.polynomial.legendre.leggauss(192)
_THETA = 0.25 * np.pi * (_X + 1.0)
_THETA_W = 0.25 * np.pi * _W


def gbm(S0, sigma, T, n_steps, n_paths, rng, mu=0.0):
    """Paths of S on n_steps equal steps over T, shape (n_paths, n_steps + 1).

    Inputs: S0 starting price, sigma annual volatility, T years, rng a numpy random Generator, mu annual drift
    (0 by default). Output: array (n_paths, n_steps + 1) of prices, column 0 equal to S0.
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    # Simulate log S exactly (no discretisation error), then exponentiate; the -0.5 sigma^2 term keeps the mean of S
    # growing at rate mu.
    log_s = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_s]))


@dataclass(frozen=True)
class StochasticVol:
    """Heston variance dv = kappa (theta - v) dt + xi sqrt(v) dW, corr(dW, dB) = rho; with lam > 0 the price also
    jumps (Bates): log jump size N(mu_j, sigma_j^2), lam jumps a year, compensated so the forward stays a martingale.

    Parameters (all annual):
        kappa    speed at which variance returns to its long-run level (2 means a half-life of about 4 months)
        theta    long-run variance (0.04 means 20% volatility)
        xi       volatility of variance ("vol of vol")
        rho      correlation between price shocks and variance shocks
        lam      average number of jumps a year (0: plain Heston)
        mu_j     mean of the log jump size (-0.05: jumps of about -5%)
        sigma_j  standard deviation of the log jump size

    "Compensated" means the price's drift is lowered by the average jump effect, so on average the forward price
    neither rises nor falls; that is needed for option prices to be fair.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    @property
    def jump_compensator(self):
        """Expected relative price change from jumps per year: lam * (E[jump factor] - 1)."""
        return self.lam * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0)

    def stationary_variance(self, n, rng):
        """Draws from the stationary (gamma) distribution of v.

        That is the spread of variance levels the model visits in the long run. Returns an array of n variances,
        used to start paths at realistic, varied volatility levels.
        """
        shape = 2.0 * self.kappa * self.theta / self.xi**2
        return rng.gamma(shape, self.xi**2 / (2.0 * self.kappa), size=n)

    def simulate(self, S0, v0, T, n_steps, n_paths, rng, substeps=4, mu=0.0):
        """Full-truncation Euler on n_steps * substeps steps; returns S and v on the n_steps grid.

        Inputs: S0 starting price, v0 starting variance (one number or one per path), T years, rng a numpy random
        Generator, substeps extra small steps between grid points (for accuracy), mu real-world drift.
        Output: (S, V), each array (n_paths, n_steps + 1): prices and variances at the grid times.
        "Full truncation" means a negative variance (possible after a coarse step) is treated as zero.
        """
        dt = T / (n_steps * substeps)
        s = np.full(n_paths, np.log(S0), dtype=float)
        v = np.broadcast_to(np.asarray(v0, dtype=float), (n_paths,)).copy()
        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0], V[:, 0] = S0, v
        c = np.sqrt(1.0 - self.rho**2)
        for i in range(n_steps):
            for _ in range(substeps):
                # Two normal shocks with correlation rho: z1 drives the price, z2 the variance.
                z1 = rng.standard_normal(n_paths)
                z2 = self.rho * z1 + c * rng.standard_normal(n_paths)
                vp = np.maximum(v, 0.0)
                # Log price step: drift less the jump compensator, so jumps do not change the average.
                s += (mu - self.jump_compensator - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
                if self.lam > 0:
                    # k jumps in this small step (Poisson); the sum of k normal log jumps is normal with mean
                    # k * mu_j and standard deviation sqrt(k) * sigma_j.
                    k = rng.poisson(self.lam * dt, n_paths)
                    s += k * self.mu_j + np.sqrt(k) * self.sigma_j * rng.standard_normal(n_paths)
                v = v + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z2
            S[:, i + 1] = np.exp(s)
            V[:, i + 1] = np.maximum(v, 0.0)
        return S, V

    def cf(self, z, tau, v):
        """Characteristic function of log(S_tau / S_0) at complex z (little-trap form), pricing measure.

        Inputs: z complex frequency (array), tau years ahead, v current variance. Output: complex array,
        E[exp(i z log(S_tau / S_0))]. The "little-trap" form is an algebraically equal way of writing Heston's
        formula that avoids a numerical jump in the complex logarithm.
        """
        iz = 1j * z
        b = self.kappa - self.rho * self.xi * iz
        d = np.sqrt(b * b + self.xi**2 * (iz + z * z))
        g = (b - d) / (b + d)
        e = np.exp(-d * tau)
        C = self.kappa * self.theta / self.xi**2 * ((b - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        D = (b - d) / self.xi**2 * (1.0 - e) / (1.0 - g * e)
        # The jump part (Merton), and the compensator that keeps the forward's expected value unchanged.
        jumps = self.lam * tau * (np.exp(iz * self.mu_j - 0.5 * self.sigma_j**2 * z * z) - 1.0)
        return np.exp(C + D * v + jumps - iz * self.jump_compensator * tau)

    def total_variance(self, tau, v):
        """Expected integrated variance of log S over tau, jumps included.

        Roughly volatility squared times time, summed over the option's life. Used as a Black–Scholes stand-in:
        sqrt(total_variance / tau) is the model's average volatility over tau.
        """
        # Variance starts at v and decays towards theta at speed kappa; this is the average of that path times tau.
        k = np.maximum(self.kappa * tau, 1e-12)
        diffusion = self.theta * tau + (v - self.theta) * tau * (1.0 - np.exp(-k)) / k
        return diffusion + self.lam * (self.mu_j**2 + self.sigma_j**2) * tau

    def call(self, S, K, tau, v, chunk=20000):
        """European call by Lewis's formula, C = S - sqrt(SK)/pi * int Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du,
        with a Black–Scholes control variate at the same expected total variance (the integral then only carries
        the difference, which stays smooth near expiry) and u = tan(theta) / 2.

        Inputs: S price, K strike, tau years to expiry, v current variance; numbers or arrays of any matching shape.
        Output: call prices, same shape, never below the payoff max(S - K, 0). Inputs are processed in chunks of
        `chunk` options to limit memory.
        """
        S, K, tau, v = (np.asarray(a, dtype=float) for a in np.broadcast_arrays(S, K, tau, v))
        out = np.empty(S.shape)
        # Flat views, so any input shape is handled as one long list of options; writing into o_ fills `out`.
        s_, k_, t_, v_, o_ = (a.reshape(-1) for a in (S, K, tau, v, out))
        u = 0.5 * np.tan(_THETA)
        z = u - 0.5j
        for a in range(0, s_.size, chunk):
            b = slice(a, a + chunk)
            # Price = Black–Scholes price at the same total variance + (model - Black–Scholes) correction. The
            # correction is small and smooth, so the fixed set of integration nodes computes it accurately.
            w = np.maximum(self.total_variance(t_[b], v_[b]), 1e-12)
            phi = self.cf(z[None, :], t_[b, None], v_[b, None])
            phi_bs = np.exp(-0.5 * w[:, None] * (1j * z[None, :] + z[None, :] ** 2))
            logm = np.log(s_[b] / k_[b])
            integrand = np.real(np.exp(1j * u[None, :] * logm[:, None]) * (phi_bs - phi))
            # Black–Scholes with tau = 1 and sigma = sqrt(w) has total variance w, the same as the model's.
            control = bs.price(s_[b], k_[b], np.ones_like(w), np.sqrt(w))
            # The factor 2 is du / (u^2 + 1/4) after the change of variable u = tan(theta) / 2.
            o_[b] = control + np.sqrt(s_[b] * k_[b]) / np.pi * 2.0 * (integrand @ _THETA_W)
        out = np.where(tau > 0, out, np.maximum(S - K, 0.0))
        return np.maximum(out, np.maximum(S - K, 0.0))

    def price(self, S, K, tau, v, call=True):
        """Call or put price under this model; the put comes from put-call parity (rate zero). Same shapes as call()."""
        c = self.call(S, K, tau, v)
        return c if call else c - (S - K)

    def delta(self, S, K, tau, v, call=True, bump=1e-3):
        """dV/dS at fixed variance, central difference.

        The model's own delta: reprice with S moved up and down by 0.1% and take the slope. Variance is held fixed,
        so it ignores that variance tends to move with the price (rho). Same shapes as call().
        """
        h = bump * np.asarray(S)
        d = (self.call(S + h, K, tau, v) - self.call(S - h, K, tau, v)) / (2.0 * h)
        return d if call else d - 1.0

    def atm_iv(self, tau, v):
        """Black–Scholes implied volatility of the at-the-money call with time tau left.

        Inputs: tau years, v current variance (number or array). Output: annual volatility, same shape as v. This
        is the "30-day ATM implied volatility" series the generator learns from.
        """
        return bs.implied_vol(self.call(100.0, 100.0, tau, v), 100.0, 100.0, tau)
