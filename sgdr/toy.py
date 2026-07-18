"""Small pedagogical games for notebook 01 — every object here exists
to teach one concept with power-system framing:

* `ev_matrix_game`      — 2-player, 2-hour EV charging payoff table
                          (Nash vs. social optimum on a discrete game).
* `TransformerGame`     — two households sharing one transformer whose
                          congestion raises the price for both
                          (continuous strategies, best-response curves,
                          tragedy of the commons).
* `stackelberg_toy`     — one utility (leader, sets a flat price) and
                          one household (follower): first-mover logic
                          in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, minimize


# ----------------------------------------------------------------------
# 1. Discrete game: two EVs, two hours, one small transformer
# ----------------------------------------------------------------------

def ev_matrix_game(cheap: float = 1.0, expensive: float = 3.0,
                   overload: float = 2.5) -> np.ndarray:
    """Cost matrix [a1, a2, player] for the 2-EV charging game.

    Actions: 0 = charge in the cheap hour, 1 = charge in the expensive
    hour. If both pick the same hour the transformer overloads and each
    pays `overload` (throttled charging, voltage sag) instead of the
    tariff. Entries are COSTS (lower is better).
    """
    cost = np.empty((2, 2, 2))
    tariff = np.array([cheap, expensive])
    for a1 in (0, 1):
        for a2 in (0, 1):
            if a1 == a2:
                cost[a1, a2] = (overload, overload)
            else:
                cost[a1, a2] = (tariff[a1], tariff[a2])
    return cost


def pure_nash_of_cost_matrix(cost: np.ndarray) -> list[tuple[int, int]]:
    """All pure-strategy Nash equilibria of a 2x2 cost bimatrix, found
    by brute-force no-deviation check (the definition, executed)."""
    eqs = []
    for a1 in (0, 1):
        for a2 in (0, 1):
            dev1 = cost[1 - a1, a2, 0] < cost[a1, a2, 0]
            dev2 = cost[a1, 1 - a2, 1] < cost[a1, a2, 1]
            if not dev1 and not dev2:
                eqs.append((a1, a2))
    return eqs


# ----------------------------------------------------------------------
# 2. Continuous game: two households on one congested transformer
# ----------------------------------------------------------------------

@dataclass
class TransformerGame:
    """Households i=1,2 choose consumption x_i >= 0 [kWh].

    Payoff:  u_i = a_i ln(1 + x_i) - p(X) x_i,  p(X) = p0 + c X,
    X = x_1 + x_2.  The linear price rise p(X) is the congestion of the
    shared transformer: MY consumption raises YOUR price too — that
    externality is the whole story of this game.
    """

    a1: float = 8.0
    a2: float = 6.0
    p0: float = 1.0
    c: float = 0.5
    toll1: float = 0.0   # per-kWh congestion toll on household 1
    toll2: float = 0.0   # per-kWh congestion toll on household 2

    def price(self, X: float) -> float:
        return self.p0 + self.c * X

    def payoff(self, i: int, x1: float, x2: float) -> float:
        a = self.a1 if i == 1 else self.a2
        xi = x1 if i == 1 else x2
        toll = self.toll1 if i == 1 else self.toll2
        return a * np.log1p(xi) - (self.price(x1 + x2) + toll) * xi

    def best_response(self, i: int, xj: float) -> float:
        """Closed form: solve a/(1+x) = p0 + toll + c*xj + 2c*x  (FOC).

        The 2c (not c) is the strategic term: consuming one more kWh
        raises the price on every kWh I already consume.
        """
        a = self.a1 if i == 1 else self.a2
        toll = self.toll1 if i == 1 else self.toll2
        m = self.p0 + toll + self.c * xj
        A, B, C = 2 * self.c, m + 2 * self.c, m - a
        disc = B * B - 4 * A * C
        x = (-B + np.sqrt(disc)) / (2 * A)
        return max(x, 0.0)

    def nash(self, tol: float = 1e-12, max_iter: int = 10_000
             ) -> tuple[float, float]:
        """Nash equilibrium by iterated (sequential) best response."""
        x1 = x2 = 0.0
        for _ in range(max_iter):
            x1n = self.best_response(1, x2)
            x2n = self.best_response(2, x1n)
            if abs(x1n - x1) < tol and abs(x2n - x2) < tol:
                return x1n, x2n
            x1, x2 = x1n, x2n
        return x1, x2

    def br_path(self, x1_0: float = 0.0, x2_0: float = 0.0,
                steps: int = 12) -> np.ndarray:
        """Staircase path of sequential best responses, for plotting."""
        pts = [(x1_0, x2_0)]
        x1, x2 = x1_0, x2_0
        for _ in range(steps):
            x1 = self.best_response(1, x2)
            pts.append((x1, x2))
            x2 = self.best_response(2, x1)
            pts.append((x1, x2))
        return np.array(pts)

    def social_optimum(self) -> tuple[float, float]:
        """max u_1 + u_2 — the 'single dispatcher' benchmark."""
        def neg_welfare(x):
            return -(self.payoff(1, x[0], x[1]) + self.payoff(2, x[0], x[1]))
        res = minimize(neg_welfare, x0=[1.0, 1.0],
                       bounds=[(0, None), (0, None)], method="L-BFGS-B")
        return float(res.x[0]), float(res.x[1])

    def welfare(self, x1: float, x2: float) -> float:
        return self.payoff(1, x1, x2) + self.payoff(2, x1, x2)


# ----------------------------------------------------------------------
# 3. Stackelberg toy: one utility, one household, closed form
# ----------------------------------------------------------------------

def follower_demand(p: float, a: float = 8.0) -> float:
    """Household max a ln(1+x) - p x  =>  x(p) = a/p - 1 (clipped)."""
    return max(a / p - 1.0, 0.0)


def leader_profit(p: float, a: float = 8.0, cgen: float = 1.0) -> float:
    """Utility profit (p - cgen) * x(p), anticipating the follower."""
    return (p - cgen) * follower_demand(p, a)


def stackelberg_toy(a: float = 8.0, cgen: float = 1.0):
    """Returns (p*, x*, profit*) of the leader's problem.

    Closed form: p* = sqrt(a * cgen) — the leader marks up above
    marginal cost because it moves first and knows x(p).
    """
    p_star = float(np.sqrt(a * cgen))
    x_star = follower_demand(p_star, a)
    return p_star, x_star, leader_profit(p_star, a, cgen)
