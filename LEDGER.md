# LEDGER — authoritative log of results, assumptions, and deviations

Chronological; newest entries at the bottom of each milestone. Every claim in the
README or notebooks traces to an entry here.

## M1 — Recreation of Maharjan et al. (2013)

### Scope decisions
- **[M1-S1]** Recreate paper Sections III–V (system model, Stackelberg game,
  distributed algorithm) and numerical Sections VII-A/B. **Skip Section VI**
  (price-manipulation attack, individual/common reserve schemes) — not needed for
  the M2–M4 program. Declared in README and notebook 02 header.
- **[M1-S2]** Solve equilibrium prices via the linear system A y = F (paper
  eqs. 30–34) instead of the determinant closed form (eq. 38): eq. 38 is
  typographically garbled in the scanned PDF, and it is derived from (32) anyway.
  Positivity (Theorem 1) verified numerically in every run.

### Assumptions
- **[M1-A1]** Follower corner cases (paper Cases 2–3) implemented as a KKT
  active-set loop: drop the UC with the most negative interior demand, re-solve
  on the active set, repeat. Verified against SLSQP (below).
- **[M1-A2]** "Sequential" updating in Algorithm 2 implemented as: users re-solve
  after *every single* UC price change (paper Table I lines 3–7 inside the UC
  loop). A simultaneous-update variant is provided and labeled as **our
  addition**, not the paper's.

### Results
- **[M1-R1]** Closed-form best response (eq. 21 + corner logic) matches SLSQP on
  40 random price vectors × 5 users (200 problems, price range 0.3–4.0 provoking
  corners): worst deviation 2.3e-05 (optimizer tolerance). Notebook 02 §2.
- **[M1-R2]** Equilibrium feasibility identities exact at machine precision:
  Σ_n x_nk = P_k and Σ_k y_k x_nk = C_n. Notebook 02 §3.
- **[M1-R3]** Paper Figs. 4–7 reproduced quantitatively. Anchors (paper values
  digitized ±few %): prices at C1=2 paper ≈(2.15,1.62,1.30) vs ours
  (2.17,1.62,1.30); demands (1.4,6.3,9.4,12.4,15.5) vs (1.36,6.29,9.37,12.45,15.53);
  revenues (21.5,24.4,26.0) vs (21.7,24.4,26.0); prices at C1=42 ≈(3.38,2.53,2.02)
  vs (3.38,2.53,2.02). Notebook 02 §4.
- **[M1-R4]** Paper Fig. 8 (5 UCs, 100 users, C1→400) reproduced structurally,
  including the equal-capacity ⇒ equal-price pairing (y1=y2, y3=y4 to machine
  precision). Notebook 02 §5.
- **[M1-R5]** Algorithm 2 (σ=40, y0=1) converges to the analytical equilibrium to
  ~1e-9 in 145 iterations (tol 1e-9; visually converged by ~30 as in the paper's
  Figs. 11–12, whose end values match ours). Notebook 02 §6.
- **[M1-R6]** σ sweep: eq. (45) sufficient bound ≈ 12.9 near y*. Measured:
  σ=2 diverges explosively; σ=5 and σ=8 hunt persistently around y* (no
  convergence in 200 iterations, final deviations 1.4 and 0.4); σ=13 (≈ the
  bound) converges fastest (34 iterations to 1e-9); σ=40 (paper's value)
  converges monotonically in 145 iterations. The theoretical margin coincides
  with the practical tuning optimum — classic integral-controller landscape.
  Notebook 02 §6.
- **[M1-R7]** Our-addition experiment: simultaneous UC updates also converge at
  this operating point (slower). Not covered by paper Theorem 5; labeled as ours.

### Deviations / negative findings
- **[M1-D1]** **Paper Figs. 9–10 are not reproducible from the stated
  parameters** (α=β=1, C=[5,10,15,20,25], P=[10,15,20], σ=40). At any fixed
  point of eq. (44), per-UC demand equals P_k, so total converged demand must be
  ΣP=45 and per-user totals ≈(3.0,6.0,9.0,12.0,14.9) [at C1=5]. Fig. 9 shows
  per-user demands converging to ≈(15,24,33,42,51), total ≈165; Fig. 10's
  utilities (≈5.3–8.6) are likewise inconsistent with Fig. 5 (≈2.05 for user 1
  at C1=5). Figs. 11 (prices ≈2.26,1.70,1.35) and 12 (revenues ≈22.5,25.3,27.0)
  ARE consistent with the stated parameters and with our runs. Conclusion:
  Figs. 9–10 used a different, unstated P (and possibly C). We reproduce the
  convergence property with the stated parameters and do not chase the unstated
  set. Notebook 02 §7.
- **[M1-D2]** Pedagogical simplification in notebook 01 §5: the oscillation demo
  uses a linear best-response game (gain r), not the transformer game — the
  transformer game's best-response slope is bounded by 1/2 in magnitude, so its
  simultaneous BR iteration always contracts and cannot oscillate. Chosen
  deliberately so the demo is honest; stated here for the record.

### Verification stance
Every numerical claim in the notebooks is produced by executed cells committed
with their outputs; anchors digitized from the paper's figures are labeled as
such (±few % reading error). No result is quoted from the paper as ours.
