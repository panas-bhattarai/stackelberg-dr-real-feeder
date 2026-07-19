"""M3: model-free learning households.

Each household knows: the posted prices (public), its own budget C_n
(its wallet), and the comfort it *experiences* after consuming — a
number, not a formula. It does NOT know alpha_n, beta_n, the log shape,
other households, supplies P_k, or that a game is being played.

Learner: two-point bandit gradient ascent (SPSA-flavored) with
Euclidean projection onto the budget set {x >= 0, p.x <= C_n}:

    Delta ~ Rademacher(K);  g_hat = [U(x + d*Delta) - U(x - d*Delta)]
                                    / (2 d) * Delta
    x <- Proj_{budget}(x + eta * g_hat)

with eta_t = eta0/sqrt(t), d_t = d0/t^0.25 (standard two-point bandit
schedules). Both probe points are evaluated at the same prices (two
half-periods of one trading round; LEDGER M3-A2).

The UCs stay exactly M1's Algorithm-2 integral controller, but on a
slower timescale: a price update every T_uc rounds, driven by the MEAN
demand observed over the period (a supplier reads its meters over the
whole period; also smooths exploration noise; LEDGER M3-A3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Game, best_response


def project_budget(x: np.ndarray, p: np.ndarray, C: float,
                   iters: int = 20) -> np.ndarray:
    """Euclidean projection onto {x >= 0, p.x <= C} (active-set on the
    nonnegativity constraints; exact for this polyhedron)."""
    x = np.clip(x, 0.0, None)
    if x @ p <= C:
        return x
    active = np.ones(len(x), dtype=bool)
    for _ in range(iters):
        # project onto hyperplane p.x = C restricted to active coords
        xa = x.copy()
        xa[~active] = 0.0
        lam = (xa @ p - C) / (p[active] @ p[active])
        xa[active] = xa[active] - lam * p[active]
        if np.all(xa[active] >= -1e-12):
            return np.clip(xa, 0.0, None)
        active &= xa >= 0
        if not active.any():
            return np.zeros_like(x)
    return np.clip(xa, 0.0, None)


@dataclass
class LearningResult:
    """Histories over rounds t (thinned by `log_every`)."""
    rounds: np.ndarray          # [T] round index of each log entry
    y: np.ndarray               # [T, K] prices
    x_total: np.ndarray         # [T, N] per-household played demand (MW)
    price_err: np.ndarray       # [T] max_k |y - y*| if y_ref given else nan
    demand_err: np.ndarray      # [T] mean_n |x_n,tot - x*_n,tot|
    regret: np.ndarray          # [T] mean_n cumulative (U_BR - U_played)/t
    seed: int


def run_coupled(game: Game, n_rounds: int, seed: int = 0,
                tau: np.ndarray | None = None,
                eta0_scale: float = 0.15, d0_scale: float = 0.05,
                T_uc: int = 10, sigma: float = 40.0,
                y0: np.ndarray | float | None = None,
                y_ref: np.ndarray | None = None,
                x_ref: np.ndarray | None = None,
                log_every: int = 10) -> LearningResult:
    """Bandit households + slow Algorithm-2 UCs, coupled.

    eta0/d0 scale per household with its budget (demands scale with C_n).
    y_ref/x_ref: known equilibrium for error metrics (agents never see it).
    """
    rng = np.random.default_rng(seed)
    N, K = game.N, game.K
    tau = np.zeros(N) if tau is None else tau
    if y0 is None:
        y0 = game.Ctot / (game.K * game.P.sum())
    y = np.broadcast_to(np.asarray(y0, dtype=float), (K,)).astype(float).copy()

    eta0 = eta0_scale * game.C
    d0 = d0_scale * game.C
    # start from an arbitrary uniform guess: spend budget equally, badly
    peff = y[None, :] + tau[:, None]
    x = (game.C / K)[:, None] / peff
    demand_accum = np.zeros(K)
    accum_count = 0

    logs = {k: [] for k in ("rounds", "y", "x_total", "price_err",
                            "demand_err", "regret")}
    cum_regret = 0.0

    def utility(xrow):
        return game.alpha * np.log(game.beta[:, None] + xrow).sum(axis=1)

    for t in range(1, n_rounds + 1):
        eta = eta0 / np.sqrt(t)
        d = d0 / t ** 0.25
        peff = y[None, :] + tau[:, None]

        delta = rng.choice([-1.0, 1.0], size=(N, K))
        xp = np.empty_like(x); xm = np.empty_like(x)
        for n in range(N):
            xp[n] = project_budget(x[n] + d[n] * delta[n], peff[n], game.C[n])
            xm[n] = project_budget(x[n] - d[n] * delta[n], peff[n], game.C[n])
        up = utility(xp)
        um = utility(xm)
        ghat = ((up - um) / (2.0 * d))[:, None] * delta
        for n in range(N):
            x[n] = project_budget(x[n] + eta[n] * ghat[n], peff[n], game.C[n])

        played = 0.5 * (xp + xm)         # the two half-period probes
        demand_accum += played.sum(axis=0)
        accum_count += 1

        if t % T_uc == 0:                # slow leaders: one UC at a time
            mean_demand = demand_accum / accum_count
            k = (t // T_uc - 1) % K
            y[k] = max(y[k] + (mean_demand[k] - game.P[k]) / sigma, 1e-9)
            demand_accum[:] = 0.0
            accum_count = 0

        if t % log_every == 0 or t == n_rounds:
            logs["rounds"].append(t)
            logs["y"].append(y.copy())
            logs["x_total"].append(played.sum(axis=1))
            logs["price_err"].append(
                np.abs(y - y_ref).max() if y_ref is not None else np.nan)
            logs["demand_err"].append(
                np.abs(played.sum(axis=1) - x_ref.sum(axis=1)).mean()
                if x_ref is not None else np.nan)
            # regret vs clairvoyant best response at CURRENT prices
            xbr_now = _best_response_tau(game, y, tau)
            cum_regret += float((utility(xbr_now) - utility(played)).mean())
            logs["regret"].append(cum_regret / len(logs["rounds"]))

    return LearningResult(
        rounds=np.array(logs["rounds"]), y=np.array(logs["y"]),
        x_total=np.array(logs["x_total"]),
        price_err=np.array(logs["price_err"]),
        demand_err=np.array(logs["demand_err"]),
        regret=np.array(logs["regret"]), seed=seed)


def _best_response_tau(game: Game, y: np.ndarray, tau: np.ndarray) -> np.ndarray:
    from .network_game import best_response_tau
    return best_response_tau(game, y, tau)


def learn_single(game: Game, n: int, y: np.ndarray, n_rounds: int,
                 seed: int = 0, eta0_scale: float = 0.15,
                 d0_scale: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """One household learning against FIXED prices (sanity check).
    Returns (trajectory [T, K], analytical best response)."""
    rng = np.random.default_rng(seed)
    K = game.K
    C, alpha, beta = game.C[n], game.alpha[n], game.beta[n]
    eta0, d0 = eta0_scale * C, d0_scale * C
    x = np.full(K, (C / K) / y.mean())
    traj = []
    for t in range(1, n_rounds + 1):
        eta, d = eta0 / np.sqrt(t), d0 / t ** 0.25
        delta = rng.choice([-1.0, 1.0], size=K)
        xp = project_budget(x + d * delta, y, C)
        xm = project_budget(x - d * delta, y, C)
        up = alpha * np.log(beta + xp).sum()
        um = alpha * np.log(beta + xm).sum()
        x = project_budget(x + eta * ((up - um) / (2 * d)) * delta, y, C)
        traj.append(x.copy())
    xbr = best_response(game, y)[n]
    return np.array(traj), xbr
