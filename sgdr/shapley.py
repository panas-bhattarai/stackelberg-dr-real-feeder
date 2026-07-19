"""M4: Shapley-value attribution of feeder congestion.

Characteristic function (LEDGER M4-A2): for a coalition S of households
consuming their equilibrium demands (everyone else at zero),

    v(S) = sum_b  lam_b * [ V_b(empty) - V_b(S) ]

— the shadow-price-weighted voltage depression at the constrained buses
(lam_b > 0 from the M2 dual ascent). v is monotone, v(empty) = 0, and
NOT additive: AC power flow makes a household's marginal depression
depend on who is already consuming — which is precisely why the Shapley
value (average marginal contribution over orderings) is the principled
way to split the total.

Exact Shapley needs 2^96 coalitions; we use Monte-Carlo permutation
sampling: each sampled ordering adds households one at a time and
records marginal contributions (96 power flows per permutation). Per
ordering the contributions telescope to v(N) exactly, so the efficiency
axiom holds by construction; sampling error is reported per household.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import feeder as F


def _vm_monitored(fdr: F.Feeder, p_mw: np.ndarray, bus_sel: np.ndarray) -> np.ndarray:
    """Voltages at the monitored load-bus positions for consumption p_mw."""
    aud = F.audit(fdr, p_mw)
    if not aud["converged"]:
        raise RuntimeError("PF diverged during Shapley evaluation")
    return aud["vm_load_bus"][bus_sel]


@dataclass
class ShapleyResult:
    phi: np.ndarray            # [N] Shapley estimates (lam·pu units)
    stderr: np.ndarray         # [N] Monte-Carlo standard error
    v_grand: float             # v(N) — total weighted depression
    n_perm: int
    running_mean: np.ndarray   # [n_perm, N] cumulative mean after each perm


def congestion_shapley(fdr: F.Feeder, p_eq: np.ndarray, lam_v: np.ndarray,
                       n_perm: int = 150, seed: int = 0,
                       verbose: bool = False) -> ShapleyResult:
    """Monte-Carlo Shapley of v(S) over households.

    p_eq   : [N] equilibrium consumptions (MW)
    lam_v  : [N] voltage shadow prices per load bus (only >0 entries count)
    """
    rng = np.random.default_rng(seed)
    N = fdr.N
    sel = lam_v > 0
    w = lam_v[sel]

    vm_empty = _vm_monitored(fdr, np.zeros(N), sel)

    def value(p_mask: np.ndarray) -> float:
        vm = _vm_monitored(fdr, p_mask, sel)
        return float(w @ (vm_empty - vm))

    contribs = np.zeros((n_perm, N))
    for m in range(n_perm):
        order = rng.permutation(N)
        p = np.zeros(N)
        v_prev = 0.0
        for n in order:
            p[n] = p_eq[n]
            v_now = value(p)
            contribs[m, n] = v_now - v_prev
            v_prev = v_now
        if verbose and (m + 1) % 10 == 0:
            print(f"  permutation {m + 1}/{n_perm}")

    phi = contribs.mean(axis=0)
    stderr = contribs.std(axis=0, ddof=1) / np.sqrt(n_perm)
    running = np.cumsum(contribs, axis=0) / np.arange(1, n_perm + 1)[:, None]
    return ShapleyResult(phi=phi, stderr=stderr, v_grand=float(contribs[0].sum()),
                        n_perm=n_perm, running_mean=running)


def jain(z: np.ndarray) -> float:
    """Jain's fairness index: 1 = perfectly equal, 1/N = maximally unequal."""
    z = np.asarray(z, dtype=float)
    return float(z.sum() ** 2 / (len(z) * (z ** 2).sum()))


def lorenz(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve points (population share, cumulative share of z)."""
    zs = np.sort(np.asarray(z, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(zs)]) / zs.sum()
    pop = np.linspace(0, 1, len(zs) + 1)
    return pop, cum


def gini(z: np.ndarray) -> float:
    pop, cum = lorenz(z)
    return float(1.0 - 2.0 * np.trapezoid(cum, pop))
