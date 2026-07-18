# Model extraction — Maharjan et al. (2013)

**Source:** S. Maharjan, Q. Zhu, Y. Zhang, S. Gjessing, T. Başar, "Dependable Demand
Response Management in the Smart Grid: A Stackelberg Game Approach," *IEEE
Transactions on Smart Grid*, vol. 4, no. 1, pp. 120–132, March 2013.
DOI: 10.1109/TSG.2012.2223766.

Extracted 2026-07-18 from the openly hosted PDF (University of Illinois Science of
Security Lablet mirror). Equation numbers below are the paper's. The PDF itself is
**not** committed to this repository (copyright); it lives untracked in `references/`.

## Scope of the recreation

Recreated: Sections III–V (system model, Stackelberg game, distributed algorithm)
and the corresponding numerical results (Figs. 4–8 quantitatively, Figs. 9–13
qualitatively — see "Reproducibility notes" below).

**Not** recreated (out of scope for M1): Section VI (attacker impact and
reserve-power schemes) and the wireless-communication-layer discussion.

## Actors and notation

| Paper symbol | Meaning | Power-engineering reading |
|---|---|---|
| `N`, n ∈ N | N end-users (consumers) | loads / prosumer households |
| `K`, k ∈ K | K utility companies (UCs) | competing suppliers/retailers |
| `x_{n,k}` | demand of user n from UC k | energy purchased [units] |
| `y_k` | unit price set by UC k | tariff [currency/unit] |
| `C_n > 0` | budget of user n | affordability cap [currency] |
| `P_k > 0` | available power of UC k | supply capacity [units] |
| `α_n, β_n` | user utility parameters | willingness-to-pay scale; β keeps ln finite at x=0 (typ. β=1) |
| `B` | Σ_n β_n | — |
| `C` | Σ_n C_n | total purchasing power in the system |
| `σ_k` | price-adjustment speed of UC k | integral-controller gain (larger = slower) |

## Follower (user) side — Section IV-A

User utility, eq. (1):

    U_user,n = α_n · Σ_{k∈K} ln(β_n + x_{n,k})

User optimization OP_user, eqs. (2)–(4):

    max_{x_n}  U_user,n
    s.t.       Σ_k y_k x_{n,k} ≤ C_n        (budget)
               x_{n,k} ≥ 0

Convex ⇒ unique optimum. Interior-case closed-form best response, eq. (21):

    x_{n,k} = (C_n + β_n Σ_{g∈K} y_g) / (K y_k) − β_n

Validity (non-negativity) condition, eq. (23):

    y_k ≤ (C_n + β_n Σ_{g≠k} y_g) / (β_n (K − 1))

Note: at the optimum the budget binds with equality (cases 1–3 of the paper all
satisfy both constraints as equalities/actively).

## Leader (UC) side — Section IV-B

UC revenue, eq. (24):  U_gen,k = y_k · Σ_n x_{n,k}.
Capacity constraint eq. (26) Σ_n x_{n,k} ≤ P_k taken as **equality** (each UC
prefers to sell all its power; no storage).

Substituting (21) into the sell-all condition for every k yields the linear system,
eqs. (30)–(34):

    A y = F,   with
    A = diag(P_k + D) with off-diagonal entries −E
    D = B(K−1)/K,  E = B/K,  F = (C/K) · 1_K

Theorem 1: A is strictly diagonally dominant ⇒ invertible, and the unique solution
y* is strictly positive.
Theorem 2, eq. (39): necessary condition on budgets C_n for all demands ≥ 0.
Theorem 4: the UC price-selection game has a unique Nash equilibrium ⇒ the
Stackelberg game has a unique Stackelberg equilibrium (y*, x*(y*)).

Homogeneous special case (P_k = P ∀k), eq. (35)/(46):  y = C / (K·P).

## Distributed algorithm — Section V, Table I (Algorithm 2)

Only local information: users see prices, each UC sees only its own total demand
and capacity. Sequential (user-then-UC) updates; price update eq. (44):

    y_{k,t+1} = y_{k,t} + ( Σ_n x_{n,k,t} − P_k ) / σ_k

i.e., an integral controller on excess demand (power-systems reading: the market
analogue of secondary frequency control — excess demand raises price, excess
supply lowers it; fixed point ⇔ sell-all condition Σ_n x_{n,k} = P_k).

Sufficient convergence condition, eq. (45):

    σ_k > ( (K P_k − β_n) y_{k,t} − β_n Σ_{g≠k} y_{g,t} − C_n ) / (K y_{k,t}²)

Theorem 5: with sequential updates and σ_k satisfying (45), Algorithm 2 converges
to the unique SE.

## Numerical setup — Section VII

Base case: K = 3 UCs, N = 5 users, α_n = 1, β_n = 1 ∀n,
C = [5, 10, 15, 20, 25] (C_1 swept 2→42 in Figs. 4–7), P = [10, 15, 20].

Large case (Fig. 8): K = 5, N = 100, budgets: user 1 swept 2→400, users 2–25: 10,
users 26–50: 15, users 51–75: 20, users 76–100: 25; P = [150, 150, 200, 200, 250].

Distributed algorithm (Figs. 9–12): σ_k = 40 ∀k; Fig. 13: σ_k = 10 and 50,
different initial prices y_{k,1}.

## Reproduction targets and verification anchors

Values read off the paper's figures (manual digitization, ±few %):

| Anchor | Paper value | Our check |
|---|---|---|
| Fig. 6, C_1=2: y* | ≈ (2.15, 1.62, 1.30) | linear solve gives (2.17, 1.62, 1.30) ✓ |
| Fig. 4, C_1=2: user demands | ≈ (1.4, 6.3, 9.4, 12.4, 15.5) | (21) gives (1.36, …, 15.5) ✓ |
| Fig. 5, C_1=2: user-1 utility | ≈ 1.05 | ln-sum gives 1.06 ✓ |
| Fig. 7, C_1=2: revenues | ≈ (21.5, 24.4, 26) | y_k·P_k gives (21.7, 24.4, 26.0) ✓ |
| Fig. 11: converged prices (σ=40) | ≈ (2.26, 1.70, 1.35) | matches analytical y* at C_1=5 ✓ |
| Fig. 12: converged revenues | ≈ (22.5, 25.3, 27) | y*_k·P_k at C_1=5 ✓ |

## Reproducibility notes (honest-recreation findings)

1. **Figs. 9 and 10 are not reproducible from the stated parameters.** With
   C = [5,10,15,20,25] and P = [10,15,20], total equilibrium demand must equal
   ΣP = 45 (sell-all fixed point), and per-user demands are ≈ (1.7 … 14.9).
   Fig. 9 shows per-user demands converging to ≈ (15, 24, 33, 42, 51) — total
   ≈ 165 — and Fig. 10's utilities (≈5.3–8.6) are likewise inconsistent with
   Fig. 5 (≈2.1 for user 1 at C_1 = 5). Figs. 11 and 12 *are* consistent with the
   stated parameters. Conclusion: Figs. 9–10 were produced with a different
   (unstated) P and possibly C. We therefore reproduce Figs. 9–12 **with the
   stated base parameters**, verify convergence to the analytical equilibrium
   (the property the figures exist to demonstrate), and do not chase the
   unstated parameter set.
2. Eq. (38) (closed-form y_k via determinant) is typographically ambiguous in the
   scanned PDF; we solve A y = F numerically instead and verify positivity
   (Theorem 1) numerically. This loses nothing — (38) is derived from (32).
