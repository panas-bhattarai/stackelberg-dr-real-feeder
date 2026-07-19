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

## M2 — The game meets the feeder (SimBench 1-MV-rural--0-sw)

### Scope decisions and assumptions
- **[M2-A1]** Feeder: SimBench `1-MV-rural--0-sw` (97 buses, 96 loads, 17.256 MW
  nameplate, 116 km, 2×25 MVA). Rural chosen deliberately: long weak feeders ⇒
  voltage-constrained (verified: thermal never binds below γ≈1.7). Voltage limit
  0.965 pu (SimBench MV planning band, net.bus.min_vm_pu); thermal 100%.
- **[M2-A2]** Mapping: one household per SimBench load; budget C_n = nameplate
  p_mw (equal wealth per kW); q at nameplate Q/P; K=3 UCs at the substation,
  supply split 10:15:20 (M1's proportions), total ΣP = γ·17.256 MW. γ is the
  single experimental knob; sell-all makes total equilibrium demand ≡ γ·17.256.
- **[M2-A3]** DER (sgen) switched off — M2 is a load-only story.
- **[M2-A4]** β_n = 0.001 MW (must be ≪ household demand ~0.2 MW; M1's β=1
  would distort the utility shape at feeder scale).
- **[M2-A5]** "DLMP-lite": per-bus voltage shadow prices + one global thermal
  multiplier, tuned by projected dual ascent against AC-PF audits, resolved to
  households by finite-difference sensitivities (dp=0.05 MW, refreshed every 5
  outer iterations). NOT duals of an AC-OPF: no loss term, no optimality
  guarantee. Multiplier update targets VMIN+0.0005 pu so the subgradient step
  does not vanish as the gap closes (feasibility judged against VMIN itself).
- **[M2-A6]** Single trading period, deterministic; no storage, no DER, no
  inter-temporal coupling.
- **[M2-A7]** Feasibility tolerances: voltage 2e-4 pu (0.02%), loading 0.1%.
  Adopted after observing the dual ascent hover within ~1e-4 pu of the exact
  limit indefinitely (an exactness artifact, not physics — 0.02% is far below
  any real feeder's measurement accuracy). Inner market-clearing budget raised
  2000 -> 4000 iterations for the same reason (extreme adders slow Algorithm 2
  near the frontier).

### Results (all AC-PF audited; see notebook 03)
- **[M2-R1]** M1 machinery transplants unchanged to N=96: closed-form
  equilibrium ≡ Algorithm-2 clearing to ~1e-5.
- **[M2-R2]** Copperplate feasibility boundary γ* = 0.90 (grid step 0.05:
  min_vm 0.9655 at γ=0.90, 0.9615 at γ=0.95). Voltage binds first; thermal only
  ~81% even at γ=1.4. Voltage falls with electrical distance |Z| (1.9–14.7 Ω)
  — but branch-wise, not monotonically (see M2-R4).
- **[M2-R3]** Trajectory infeasibility: at γ=0.9 (equilibrium feasible), price
  discovery from y0=0.6·y* violates the voltage limit for 43 of 300 iterations,
  worst min_vm 0.9155 pu at iteration 0. Steady-state feasibility ≠ trajectory
  feasibility. (The run shows excess-demand tolerance 1e-8 not yet reached at
  iteration 300 — residual is physically immaterial; the trajectory is settled.)
- **[M2-R4]** Per-bus dual ascent restores feasibility at γ=1.1 in ~18 outer
  iterations (13 active bus constraints); total sold unchanged (19.0 MW) — the
  repricing relocates demand, it does not ration it. **τ is NOT a distance
  tariff:** the violating branch (7–11 Ω) pays up to ~0.5 while farther buses on
  healthy branches (13–15 Ω) pay ~0.03 — DLMP prices constraint responsibility,
  not remoteness. Fig. 15b.
- **[M2-R5]** Feasibility frontier (warm-started continuation, grid step 0.1):
  repriced market feasible through γ** = 1.7 (29.3 MW = +89% hosting capacity
  over γ*·17.256 = 15.5 MW). At γ=1.8 the ascent fails for solver reasons (the
  inner Algorithm-2 clearing stops converging under extreme adders; the final
  iterate's physics is within limits) — γ** is a lower bound (M2-D3). Thermal
  loading 87.5% at γ=1.7 and climbing: relocation cannot reduce total current,
  so the true frontier is a thermal wall.
- **[M2-R6]** Fairness cost (M4 hook): under repricing at γ=1.1, one budget-unit
  buys 1.121 MW-units in the farthest distance quartile vs 1.200 nearest — the
  same money buys ~7% less energy at the feeder end (distance is a proxy; the
  surcharge actually lands on the congested branch). Notebook §7.

### Deviations / negative findings
- **[M2-D1]** **A single global-min-voltage constraint whack-a-moles between
  feeder branches** and never converges: suppressing branch A's end makes
  branch B's end the new argmin; the sensitivity vector flips wholesale on each
  refresh and the market equilibrium jumps (observed stall at 0.9637 with
  5-iteration crash cycles). Fix: one multiplier per load bus (the shape a real
  OPF dual has anyway). Kept in the notebook as a told negative result.
- **[M2-D2]** With constant-step subgradient, min_vm approached 0.965
  asymptotically (0.9648 after 120 iterations) — vanishing-gradient stall;
  fixed by the +0.0005 pu update margin (M2-A5).
- **[M2-D3]** γ** is a lower bound on the true repriced hosting capacity: the
  ascent is declared failed on multiplier blow-up or iteration exhaustion,
  either of which may be conservative.

## M3 — Learning the equilibrium (notebook 04)

### Scope decisions and assumptions
- **[M3-A1]** Learner: two-point bandit gradient ascent (SPSA-flavored) with
  Euclidean projection onto the household's own budget set; schedules
  η_t ∝ 1/√t, δ_t ∝ t^(-1/4); per-household scales η0 = 0.15·C_n,
  δ0 = 0.05·C_n. Deliberately NOT deep RL: the question (does decentralized
  selfish learning find the game's equilibrium?) is cleanest with the simplest
  model-free learner with convex-case guarantees.
- **[M3-A2]** Both probe points of a round are evaluated at unchanged prices
  (two half-periods of one trading round). The "played" demand of a round is
  the mean of the two probes.
- **[M3-A3]** Leaders: unchanged Algorithm-2 integral control, one UC update
  per T_uc rounds (round-robin), driven by the MEAN demand metered over the
  period — realistic (energy metering) and smooths exploration noise.
- **[M3-A4]** Information diet: households observe posted prices, own budget,
  own realized comfort (a number). They never see α_n, β_n, the log form,
  other players, supplies, or any equilibrium quantity. Error metrics use the
  known equilibria (R1 exact / M2 dual-ascent) — agents never do.
- **[M3-A5]** In §6 the congestion adders are FROZEN at the dual-ascent τ* —
  no adaptive repricing against learners. Adaptive-DLMP-vs-learners is named
  as an open question, not attempted.
- **[M3-A6]** Rounds are independent (no inter-round load coupling, storage,
  or weather); single feeder; γ=0.9 (copperplate experiments) and γ=1.1
  (network-priced experiment).

### Results (executed notebook 04)
- **[M3-R1]** Single learner vs fixed prices recovers the eq.-(21) best
  response to 2.7% in 3000 rounds (~t^(-1/2) error decay), never having seen
  the formula. §2.
- **[M3-R2]** Coupled system (96 learners + 3 adaptive leaders, 6 seeds,
  8000 rounds) converges to the exact SE: median price error 1.8%, median
  demand error 3.3 kW/household (equilibrium mean 162 kW) — ~100× slower than
  clairvoyant R2 (thousands of rounds vs ~10²). Error floors (stochastic
  steady state); equilibrium is an attractor, not a destination. §3.
- **[M3-R3]** Two-loop tuning is two-sided (final-error × tail-jitter):
  T_uc=50 too sluggish (err 0.17); (T_uc=1, σ=5) chases exploration noise
  (jitter 0.029, ~8× the quiet configs); working region T_uc≈10 with σ 10–40
  (lowest jitter 0.0035 at (10,40); lowest error 0.038 at (10,10)). §4.
- **[M3-R4]** Trajectory audits at γ=0.9: learning trajectory violates the
  voltage limit on 120 of 400 audited rounds, worst 0.8295 pu (deep collective
  exploration early; flicker across the 0.5 milli-pu margin late), vs 47/400
  for the R2 baseline from the same starting prices — and R2's violations end,
  the learners' recur. Equilibrium analysis certifies a point; learning
  occupies a neighborhood. §5.
- **[M3-R5]** Frozen DLMP-lite adders steer learners to the repriced
  equilibrium at γ=1.1 (price error → 1.5%) — but that equilibrium rides the
  limit by construction, so learning noise crosses it on 311 of 400 audited
  rounds (worst 0.8748): static efficiency leaves zero margin for learning.
  Efficiency vs robustness-to-learning named as the open tension. §6.
