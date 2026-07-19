# stackelberg-dr-real-feeder

**Status: work in progress — Milestones 1–3 of 4 complete.**

**What happens to game-theoretic demand response when the copperplate becomes a
real feeder?** Game-theoretic DR is almost always studied on a copperplate model —
energy flows from anyone to anyone, no impedance, no voltage limits, no location.
This repository starts from a faithful recreation of the canonical Stackelberg
demand-response formulation and then, milestone by milestone, confronts it with
distribution-feeder physics, learning agents, and fairness metrics.

## Milestones

| # | Question | Status |
|---|---|---|
| **M1** | Recreate the Stackelberg DR game of Maharjan *et al.* (2013): unique equilibrium, closed forms, distributed local-information algorithm | ✅ done |
| **M2** | Pin the households to buses of a SimBench MV feeder; audit every equilibrium with AC power flow; reprice with network-aware (DLMP-style) prices | ✅ done |
| **M3** | Replace clairvoyant best responses with learning agents on private data only — is the equilibrium still reachable? | ✅ done |
| **M4** | Measure location-(un)fairness of efficient network pricing; test Shapley-based cost allocation as repair and as bill explanation | ⏳ next |

## M3 — learning the equilibrium

The clairvoyant eq.-(21) households are replaced by **model-free bandit
learners** — each observes only posted prices, its own budget, and its own
realized comfort (a number, not a formula), and learns by perturb-and-observe
(the MPPT logic, as a market strategy). Leaders keep the paper's Algorithm-2
integral control on a slower timescale. Findings, multi-seed and AC-PF audited:

- **The equilibrium is learnable from comfort feedback alone:** 96 learners +
  3 adaptive leaders converge to the exact Stackelberg equilibrium (median 1.8%
  price error, 3.3 kW/household demand error over 6 seeds) — but ~100× slower
  than clairvoyant dynamics, and to a stochastic neighborhood, never a point.
- **The two nested loops must respect cascade-control rules:** sluggish leaders
  never arrive, eager leaders chase exploration noise and amplify it (~8×
  price jitter) — who updates when is a design variable, measured.
- **Learning flicker spends the equilibrium's safety margin:** at γ=0.9 the
  exact equilibrium keeps 0.5 milli-pu of voltage headroom; the learning
  trajectory violates the limit on 120 of 400 audited rounds (worst 0.83 pu)
  and keeps flickering across it after "converging." Equilibrium analysis
  certifies a point; learning occupies a neighborhood.
- **DLMP-lite prices steer learners correctly but leave zero margin:** under
  the frozen network-aware adders the learners find the repriced equilibrium,
  yet cross the voltage limit on 311 of 400 audited rounds — maximal static
  efficiency and robustness-to-learning are in direct tension (named as the
  open thesis-grade question).

## M2 — the game meets the feeder

The M1 game, unchanged, with its 96 households pinned to the buses of SimBench
`1-MV-rural--0-sw` (17.3 MW nameplate, 116 km of 20 kV line) and **every
equilibrium audited with AC power flow**. Findings, all physics-gated:

- **The market clears; the feeder doesn't.** Above supply level γ\* = 0.9
  (15.5 MW) the unique Stackelberg equilibrium violates the 0.965 pu planning
  band — the game is structurally blind to the wire. Voltage binds first,
  thermal never does (rural signature).
- **The road to equilibrium is worse than the equilibrium:** price discovery
  transits 43 iterations of voltage violation (worst 0.9155 pu) on its way to a
  perfectly feasible resting point — steady-state feasibility ≠ trajectory
  feasibility, a warning shot for learning agents (M3).
- **DLMP-lite repricing restores feasibility through self-interest alone:**
  per-bus congestion adders found by projected dual ascent against AC-PF audits
  relocate (not ration) demand. A single global voltage constraint provably
  whack-a-moles between feeder branches — per-bus multipliers are necessary
  (negative result, kept).
- **Honest prices nearly double the hosting capacity:** feasible market supply
  extends from 15.5 MW to 29.3 MW (+89%) with zero new copper, ending at a
  thermal wall no relocation can dodge.
- **The adders price responsibility, not remoteness** — the congested branch
  pays ~15× more than equally-remote healthy branches — and the same budget
  buys ~7% less energy at the feeder end: efficient ≠ fair, which is M4's
  opening question.

## M1 — the recreation

> S. Maharjan, Q. Zhu, Y. Zhang, S. Gjessing, T. Başar, "Dependable Demand
> Response Management in the Smart Grid: A Stackelberg Game Approach,"
> *IEEE Trans. Smart Grid* 4(1), 2013. DOI:
> [10.1109/TSG.2012.2223766](https://doi.org/10.1109/TSG.2012.2223766)

⚠️ Independent, unofficial, educational recreation — see
[`DISCLAIMER.md`](DISCLAIMER.md). Not affiliated with or endorsed by the authors.

**Recreated and verified:** the closed-form follower best response (checked
against a numerical optimizer to ~10⁻⁵, corner cases included), the unique
positive Stackelberg equilibrium via the paper's linear system (sell-all and
budget identities hold to machine precision), the paper's Figs. 4–8 quantitatively
(anchor tables in the notebook), and the distributed price-adjustment algorithm —
which converges to the analytical equilibrium using local information only, with
the gain-vs-stability trade-off of its σ parameter mapped.

**Honestly logged:** the paper's Figs. 9–10 are not reproducible from its stated
parameters (their converged demands are inconsistent with total supply); Figs.
11–12 are. Details in [`LEDGER.md`](LEDGER.md).

## Reading order

| Notebook | Content |
|---|---|
| [`01_from_dispatch_to_games`](notebooks/01_from_dispatch_to_games.ipynb) | Game theory from scratch for a power engineer: Nash via a 2-EV/1-transformer game, best-response dynamics as Gauss–Seidel, price of anarchy, a congestion toll that makes selfishness optimal, Stackelberg leadership, and why best-response dynamics can hunt or diverge |
| [`02_maharjan2013_recreation`](notebooks/02_maharjan2013_recreation.ipynb) | The faithful recreation: model extraction, verification, reproduction of the paper's figures, the distributed algorithm as an integral controller on excess demand, and the recreation scorecard |
| [`03_the_game_meets_the_feeder`](notebooks/03_the_game_meets_the_feeder.ipynb) | M2: the same game on a real feeder — AC-PF audits of equilibria and transients, the DLMP-lite dual ascent (with its instructive failures), the hosting-capacity frontier, and the fairness teaser |
| [`04_learning_the_equilibrium`](notebooks/04_learning_the_equilibrium.ipynb) | M3: model-free households — bandit learning to the exact SE, the cascade-tuning landscape, and the trajectory audits showing learning noise spend the voltage margin |

[`docs/model_extraction.md`](docs/model_extraction.md) holds the full equation
extraction with notation mapped to power-engineering terms;
[`LEDGER.md`](LEDGER.md) is the authoritative log of every result, assumption,
and deviation.

## Reproducing

Python 3.11; `pip install -r requirements.txt`; CPU-only, seconds to run.
Rebuild the notebooks with `python scripts/build_nb01.py && python
scripts/build_nb02.py` and execute with `jupyter nbconvert --execute --inplace
notebooks/*.ipynb`. The paper PDF is not redistributed; it is openly hosted (link
in `docs/model_extraction.md`).

## Lineage

This is the fourth study in a line: the
[`gridfm-thesis-recreation`](https://github.com/panas-bhattarai/gridfm-thesis-recreation)
(masked-reconstruction grid foundation models),
[`feeder-reconfiguration-invariance`](https://github.com/panas-bhattarai/feeder-reconfiguration-invariance)
(topology transfer under reconfiguration), and
[`physics-consistent-feeder-generation`](https://github.com/panas-bhattarai/physics-consistent-feeder-generation)
(generative models with power-flow guarantees) all asked *prediction/generation*
questions on real feeders. This repository asks the *decision* question on the
same feeders: how strategic actors share them, and at what price.

## License

MIT (code and notebooks). The recreated paper remains © IEEE / its authors; this
repository contains no text or figures from it, only an independent
implementation of its published equations.
