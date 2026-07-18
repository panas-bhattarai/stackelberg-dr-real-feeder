"""Algorithm 2 (Table I) of Maharjan et al. 2013: the distributed,
local-information-only price/demand iteration.

Each UC sees only its own capacity P_k and the total demand addressed
to it; each user sees only the current prices. The price update
(eq. 44) is an integral controller on excess demand:

    y_{k,t+1} = y_{k,t} + (sum_n x_{n,k,t} - P_k) / sigma_k

Fixed point  <=>  sell-all condition  sum_n x_{n,k} = P_k  <=>  the
Stackelberg equilibrium (Theorem 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Game, best_response


@dataclass
class RunResult:
    """History of one distributed run. Arrays indexed [t, ...]."""

    y: np.ndarray          # [T, K] prices per iteration
    excess: np.ndarray     # [T, K] sum_n x_{n,k} - P_k
    x_total: np.ndarray    # [T, N] per-user total demand
    utility: np.ndarray    # [T, N] per-user utility
    revenue: np.ndarray    # [T, K] per-UC revenue
    converged: bool
    iterations: int


def run(game: Game, sigma: float | np.ndarray, y0: float | np.ndarray = 1.0,
        max_iter: int = 400, tol: float = 1e-9,
        sequential: bool = True) -> RunResult:
    """Run Algorithm 2.

    sequential=True follows the paper (UCs update one at a time, users
    re-respond after every single price change — Theorem 5's premise).
    sequential=False updates all prices simultaneously, which the paper
    does NOT cover; we use it in the notebook to show why the premise
    matters.
    """
    K, N = game.K, game.N
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (K,)).copy()
    y = np.broadcast_to(np.asarray(y0, dtype=float), (K,)).astype(float).copy()

    ys, excesses, xtots, utils, revs = [], [], [], [], []
    converged = False
    for t in range(max_iter):
        x = best_response(game, y)
        excess = x.sum(axis=0) - game.P

        ys.append(y.copy())
        excesses.append(excess.copy())
        xtots.append(x.sum(axis=1))
        utils.append(game.alpha * np.log(game.beta[:, None] + x).sum(axis=1))
        revs.append(y * x.sum(axis=0))

        if np.max(np.abs(excess)) < tol:
            converged = True
            break

        if sequential:
            for k in range(K):
                xk = best_response(game, y)          # users re-respond
                y[k] = y[k] + (xk[:, k].sum() - game.P[k]) / sigma[k]
        else:
            y = y + excess / sigma
        y = np.clip(y, 1e-12, None)

    return RunResult(y=np.array(ys), excess=np.array(excesses),
                     x_total=np.array(xtots), utility=np.array(utils),
                     revenue=np.array(revs), converged=converged,
                     iterations=len(ys))


def sigma_sufficient(game: Game, y: np.ndarray) -> float:
    """Largest right-hand side of the sufficient condition (eq. 45)
    over all k, n at prices y — any sigma above this is safe there."""
    y = np.asarray(y, dtype=float)
    S = y.sum()
    worst = -np.inf
    for k in range(game.K):
        for n in range(game.N):
            num = ((game.K * game.P[k] - game.beta[n]) * y[k]
                   - game.beta[n] * (S - y[k]) - game.C[n])
            worst = max(worst, num / (game.K * y[k] ** 2))
    return worst
