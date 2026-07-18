"""M2: the Stackelberg DR game with a feeder underneath.

Three layers on top of the M1 machinery:

1. `NetworkGame` — the M1 game populated from the feeder: one household
   per SimBench load (budget proportional to nameplate), K UCs at the
   substation with total supply gamma * (feeder nameplate).
2. `best_response_tau` / `clear_market` — the game with per-household
   congestion adders tau_n. With adders, the effective price household n
   pays UC k is (y_k + tau_n): closed forms for the UC-side equilibrium
   die (the linear system of M1 assumed identical prices for everyone),
   but the paper's own distributed Algorithm 2 survives untouched — the
   integral controller on excess demand does not care that the demand
   function got more complicated.
3. `dual_ascent` — DLMP-lite: project the feeder's voltage and thermal
   constraints into the market as bus-differentiated price adders,
   tau_n = lam_v * s_v[n] + lam_i * s_i[n], with the multipliers
   lam_* updated by projected subgradient ascent on the measured
   violations (AC-PF audited every step). This is deliberately NOT a
   full DLMP/AC-OPF dual solution — see LEDGER M2-A5 for the honest
   scoping — but it is the same economic object: constraint shadow
   prices, resolved per bus by sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Game
from . import feeder as F


# ----------------------------------------------------------------------
# Game construction from the feeder
# ----------------------------------------------------------------------

BETA_MW = 0.001          # log-offset, small vs. household demands (~0.2 MW)
P_SPLIT = np.array([10.0, 15.0, 20.0])   # M1's UC proportions, kept


def make_network_game(fdr: F.Feeder, gamma: float) -> Game:
    """Households: budgets C_n = nameplate p_mw (currency units chosen so
    1 unit buys ~1 MW at price ~1). UCs: total supply gamma * nameplate
    total, split in M1's 10:15:20 proportions."""
    P = P_SPLIT / P_SPLIT.sum() * gamma * fdr.p_total
    return Game(alpha=np.ones(fdr.N), beta=np.full(fdr.N, BETA_MW),
                C=fdr.p_nameplate.copy(), P=P)


# ----------------------------------------------------------------------
# The game with per-household price adders tau
# ----------------------------------------------------------------------

def best_response_tau(game: Game, y: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Optimal demands when household n faces effective prices y_k + tau_n.

    Same KKT logic as M1's best_response — the adder shifts every UC's
    price identically for a given household, so eq. (21) applies with
    y -> y + tau_n per household (active-set corners included).
    """
    y = np.asarray(y, dtype=float)
    x = np.zeros((game.N, game.K))
    for n in range(game.N):
        yn = y + tau[n]
        active = np.ones(game.K, dtype=bool)
        while True:
            Ka = int(active.sum())
            S = yn[active].sum()
            xa = (game.C[n] + game.beta[n] * S) / (Ka * yn[active]) - game.beta[n]
            if np.all(xa >= 0) or Ka == 1:
                break
            idx = np.flatnonzero(active)
            active[idx[np.argmin(xa)]] = False
        x[n, np.flatnonzero(active)] = np.clip(xa, 0.0, None)
    return x


@dataclass
class ClearResult:
    y: np.ndarray            # UC prices at the sell-all fixed point
    x: np.ndarray            # [N, K] demands
    converged: bool
    iterations: int


def clear_market(game: Game, tau: np.ndarray, sigma: float = 40.0,
                 y0: np.ndarray | float | None = None,
                 max_iter: int = 4000, tol: float = 1e-8) -> ClearResult:
    """Algorithm 2 (sequential price adjustment) under adders tau.

    Finds prices y such that every UC sells exactly P_k while households
    best-respond to (y + tau_n). This is the paper's own distributed
    dynamics doing the job its closed form no longer can.
    """
    K = game.K
    if y0 is None:
        y0 = game.Ctot / (game.K * game.P.sum())     # homogeneous-case scale
    y = np.broadcast_to(np.asarray(y0, dtype=float), (K,)).astype(float).copy()
    for t in range(max_iter):
        x = best_response_tau(game, y, tau)
        excess = x.sum(axis=0) - game.P
        if np.max(np.abs(excess)) < tol:
            return ClearResult(y=y, x=x, converged=True, iterations=t)
        for k in range(K):
            xk = best_response_tau(game, y, tau)
            y[k] += (xk[:, k].sum() - game.P[k]) / sigma
            y[k] = max(y[k], 1e-9)
    return ClearResult(y=y, x=best_response_tau(game, y, tau),
                       converged=False, iterations=max_iter)


# ----------------------------------------------------------------------
# DLMP-lite: dual ascent on the feeder's constraints
# ----------------------------------------------------------------------

@dataclass
class AscentResult:
    feasible: bool
    lam_v: np.ndarray            # [N] per-load-bus voltage multipliers
    lam_i: float
    tau: np.ndarray
    x_total: np.ndarray          # final per-household MW
    y: np.ndarray
    audit: dict
    history: list = field(default_factory=list)


def dual_ascent(fdr: F.Feeder, game: Game,
                eta_v: float = 1.0, eta_i: float = 0.02,
                sigma: float = 40.0, max_outer: int = 120,
                sens_every: int = 5, lam_cap: float = 1e4,
                lam_v0: np.ndarray | None = None, lam_i0: float = 0.0,
                margin: float = 5e-4, verbose: bool = False) -> AscentResult:
    """Outer loop: AC-PF audit -> update shadow prices -> re-clear market.

    One multiplier PER LOAD BUS on the voltage limit (projected
    subgradient: lam_b <- max(0, lam_b + eta_v * 100*(VMIN - V_b)) —
    ascends where violated, relaxes where slack), plus one global
    thermal multiplier. Adders: tau = S_v^T lam_v + lam_i * s_i.
    A single global-min voltage constraint is NOT enough — it flip-flops
    between feeder branches (LEDGER M2-D1); per-bus multipliers are the
    honest DLMP-shaped fix. Declares infeasibility on multiplier blow-up
    or exhaustion without a feasible converged state.
    """
    lam_v = np.zeros(fdr.N) if lam_v0 is None else lam_v0.copy()
    lam_i = float(lam_i0)
    tau = np.zeros(fdr.N)
    sens = None
    hist = []
    y_warm = None
    best = None
    for it in range(max_outer):
        if sens is not None:
            tau = sens["S_v"].T @ lam_v + lam_i * sens["s_i"]
        mkt = clear_market(game, tau, sigma=sigma, y0=y_warm)
        y_warm = mkt.y.copy()
        p = mkt.x.sum(axis=1)
        aud = F.audit(fdr, p)
        if not aud["converged"]:
            return AscentResult(False, lam_v, lam_i, tau, p, mkt.y, aud, hist)
        if sens is None or it % sens_every == 0:
            sens = F.sensitivities(fdr, p)
        max_loading = max(aud["max_line_loading"], aud["max_trafo_loading"])
        hist.append({"iter": it, "sum_lam_v": float(lam_v.sum()),
                     "lam_i": lam_i, "min_vm": aud["min_vm"],
                     "max_loading": max_loading,
                     "market_converged": mkt.converged})
        if verbose:
            print(f"  outer {it:3d}: min_vm={aud['min_vm']:.4f} "
                  f"load={max_loading:.1f}% Σlam_v={lam_v.sum():.1f} "
                  f"lam_i={lam_i:.2f}")
        if aud["feasible"] and mkt.converged:
            return AscentResult(True, lam_v, lam_i, tau, p, mkt.y, aud, hist)
        # target VMIN + margin so the subgradient step does not vanish
        # as the gap closes (feasibility itself is checked against VMIN)
        v_gap = (F.VMIN_PU + margin) - aud["vm_load_bus"]  # [N], >0: push
        lam_v = np.clip(lam_v + eta_v * 100.0 * v_gap, 0.0, lam_cap)
        lam_i = min(max(0.0, lam_i + eta_i * (max_loading - F.LOADING_MAX)),
                    lam_cap)
        if lam_v.max() >= lam_cap or lam_i >= lam_cap:
            return AscentResult(False, lam_v, lam_i, tau, p, mkt.y, aud, hist)
    return AscentResult(False, lam_v, lam_i, tau, p, mkt.y, aud, hist)
