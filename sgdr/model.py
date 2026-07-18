"""The multi-UC / multi-user Stackelberg demand-response game of
Maharjan et al. (IEEE Trans. Smart Grid, 2013), Sections III-IV.

All equation numbers refer to the paper. See docs/model_extraction.md
for the full extraction with notation mapped to power-engineering terms.

Conventions
-----------
Arrays are indexed [n] for users and [k] for UCs; demand matrices are
x[n, k]. Everything is plain NumPy; no solver is required for the
equilibrium (it is a linear system), and SciPy is used only to
*verify* the closed-form best response against a numerical optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Game:
    """Parameters of the Stackelberg DR game.

    alpha, beta : per-user utility parameters (eq. 1)
    C           : per-user budgets  C_n > 0
    P           : per-UC available power  P_k > 0
    """

    alpha: np.ndarray
    beta: np.ndarray
    C: np.ndarray
    P: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.atleast_1d(np.asarray(self.alpha, dtype=float))
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=float))
        self.C = np.atleast_1d(np.asarray(self.C, dtype=float))
        self.P = np.atleast_1d(np.asarray(self.P, dtype=float))
        if not (len(self.alpha) == len(self.beta) == len(self.C)):
            raise ValueError("alpha, beta, C must have one entry per user")
        if np.any(self.C <= 0) or np.any(self.P <= 0):
            raise ValueError("budgets C_n and capacities P_k must be positive")

    @property
    def N(self) -> int:
        return len(self.C)

    @property
    def K(self) -> int:
        return len(self.P)

    @property
    def B(self) -> float:
        """B = sum_n beta_n (paper notation)."""
        return float(self.beta.sum())

    @property
    def Ctot(self) -> float:
        """C = sum_n C_n (paper notation; renamed to avoid clashing with C_n)."""
        return float(self.C.sum())


def paper_base_game(C1: float = 5.0) -> Game:
    """The Section-VII base case: 3 UCs, 5 users, alpha=beta=1,
    C = [C1, 10, 15, 20, 25], P = [10, 15, 20]."""
    return Game(
        alpha=np.ones(5),
        beta=np.ones(5),
        C=np.array([C1, 10.0, 15.0, 20.0, 25.0]),
        P=np.array([10.0, 15.0, 20.0]),
    )


def paper_large_game(C1: float = 10.0) -> Game:
    """The Fig. 8 large case: 5 UCs, 100 users."""
    C = np.concatenate([[C1], np.full(24, 10.0), np.full(25, 15.0),
                        np.full(25, 20.0), np.full(25, 25.0)])
    return Game(alpha=np.ones(100), beta=np.ones(100), C=C,
                P=np.array([150.0, 150.0, 200.0, 200.0, 250.0]))


# ----------------------------------------------------------------------
# Follower (user) side
# ----------------------------------------------------------------------

def user_utility(x_n: np.ndarray, alpha_n: float, beta_n: float) -> float:
    """Eq. (1): U_n = alpha_n * sum_k ln(beta_n + x_{n,k})."""
    return float(alpha_n * np.log(beta_n + x_n).sum())


def best_response(game: Game, y: np.ndarray) -> np.ndarray:
    """Closed-form optimal demand of every user given prices y.

    Interior case, eq. (21):

        x_{n,k} = (C_n + beta_n * sum_g y_g) / (K y_k) - beta_n

    Corner cases (paper Cases 2-3): if eq. (21) gives x_{n,k} < 0 for
    some UC k, the user buys nothing from it and re-splits the budget
    over the remaining UCs — implemented by dropping the most
    over-priced UC and re-solving on the active set (K -> K'), exactly
    the KKT active-set logic of eqs. (17)-(20).
    """
    y = np.asarray(y, dtype=float)
    x = np.zeros((game.N, game.K))
    for n in range(game.N):
        active = np.ones(game.K, dtype=bool)
        while True:
            Ka = int(active.sum())
            S = y[active].sum()
            xa = (game.C[n] + game.beta[n] * S) / (Ka * y[active]) - game.beta[n]
            if np.all(xa >= 0) or Ka == 1:
                break
            # drop the priciest violating UC and re-solve
            idx = np.flatnonzero(active)
            drop = idx[np.argmin(xa)]
            active[drop] = False
        x[n, np.flatnonzero(active)] = np.clip(xa, 0.0, None)
    return x


def demand_validity(game: Game, y: np.ndarray) -> np.ndarray:
    """Eq. (23): boolean [n, k] mask of interior (x_{n,k} >= 0) validity."""
    y = np.asarray(y, dtype=float)
    S = y.sum()
    rhs = (game.C[:, None] + game.beta[:, None] * (S - y[None, :])) / (
        game.beta[:, None] * (game.K - 1)
    )
    return y[None, :] <= rhs


def best_response_numerical(game: Game, y: np.ndarray, n: int) -> np.ndarray:
    """Solve OP_user (eqs. 2-4) for user n with a numerical optimizer.

    Used only to VERIFY eq. (21); the game itself always uses the
    closed form.
    """
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    a, b, Cn = game.alpha[n], game.beta[n], game.C[n]

    def neg_u(x):
        return -a * np.log(b + x).sum()

    cons = [{"type": "ineq", "fun": lambda x: Cn - y @ x}]
    bounds = [(0.0, None)] * game.K
    x0 = np.full(game.K, Cn / (game.K * y.mean()))
    res = minimize(neg_u, x0, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"SLSQP failed for user {n}: {res.message}")
    return res.x


# ----------------------------------------------------------------------
# Leader (UC) side and the Stackelberg equilibrium
# ----------------------------------------------------------------------

def price_system(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """The linear system A y = F of eqs. (32)-(33).

    D = B(K-1)/K, E = B/K, F = (C/K) 1.
    """
    K, B, C = game.K, game.B, game.Ctot
    D = B * (K - 1) / K
    E = B / K
    A = np.full((K, K), -E)
    np.fill_diagonal(A, game.P + D)
    F = np.full(K, C / K)
    return A, F


def equilibrium_prices(game: Game) -> np.ndarray:
    """Unique equilibrium prices y* (Theorems 1 & 4): solve A y = F."""
    A, F = price_system(game)
    y = np.linalg.solve(A, F)
    if np.any(y <= 0):  # Theorem 1 guarantees this never triggers
        raise RuntimeError("non-positive equilibrium price; model violated")
    return y


def stackelberg_equilibrium(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """(y*, x*) — equilibrium prices and the users' optimal response."""
    y = equilibrium_prices(game)
    x = best_response(game, y)
    return y, x


def revenues(game: Game, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Eq. (24): U_gen,k = y_k * sum_n x_{n,k}."""
    return np.asarray(y) * x.sum(axis=0)


def utilities(game: Game, x: np.ndarray) -> np.ndarray:
    """Per-user utilities at demand matrix x."""
    return np.array([user_utility(x[n], game.alpha[n], game.beta[n])
                     for n in range(game.N)])


def budget_check(game: Game, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-user spend sum_k y_k x_{n,k}; equals C_n at the optimum."""
    return x @ np.asarray(y)


def capacity_check(game: Game, x: np.ndarray) -> np.ndarray:
    """Per-UC served demand sum_n x_{n,k}; equals P_k at equilibrium."""
    return x.sum(axis=0)
