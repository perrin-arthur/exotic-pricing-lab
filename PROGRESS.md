# PROGRESS — exotic-pricing-lab

**XP: 1350 / 1350**  ·  Act I **closed at 220/220** · Act II **closed at
230/230** · Act III **closed at 210/210** · Act IV **closed at 340/340** ·
Act V **closed at 350/350** ✅

> **ACT V CLOSED.** Multi-asset via Cholesky → worst-of → autocall → BRC, the
> flagship product announced at the top of `CLAUDE.md`. Full BRC worst-of
> pricer, control variate reused from Act II, boss figure + sweep. Session
> notes and pitfalls below.
>
> **ACT VI 🔓 OPENED — quest 6.1 only.** Scope locked in with Arthur: Heston
> calibrated on REAL market data (yfinance, 3 commodity ETFs: GLD, SLV, USO —
> yfinance exposes no option chain on raw commodity futures, only on the
> listed ETFs that replicate them), two-step multi-asset extension (each name
> calibrated mono-asset as in Act III, THEN correlation reconstructed from
> historical returns — no joint calibration: no liquid multi-asset options
> market to calibrate one against), SVI as a per-name second baseline
> (no standard multi-asset SVI), and — Arthur's addition — a temporal
> backtest (quest 6.6) testing a falsifiable hypothesis: Heston's imposed
> structure gives more day-to-day parameter stability and better J+1
> predictive power than SVI's purely descriptive fit, even though SVI wins
> on intra-day RMSE. Pacing: one quest at a time, like every prior act — only
> 6.1 is scaffolded in code this session, the rest of the table below is a
> locked (🔒) roadmap, not yet opened.

> Green suite: **88 passed, 0 xfailed**.

Rule of the game: a quest is only earned once its validation test is
**green** against an independent reference (closed form, parity, model-free
identity). Act II's tests are written as `xfail(strict=True)`: once your code
is correct, the test flips to XPASS and fails the suite — that is the signal
to remove the marker. Removing the marker before writing the code fools no
one.

---

## ACT I — Foundations (220 / 220 XP) ✅

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 1.1 Closed-form Black-Scholes (call/put, q, delta, vega) | ✅ | 20/20 | — | `test_bs_reference_value`, `test_call_bs_dividende`, `test_put_call_parity_bs` |
| 1.2 Single-step MC pricer + 95% CI | ✅ | 20/20 | 1.1 | `test_mc_within_ci` |
| 1.3 Antithetic variates (single-step) | ✅ | 20/20 | 1.2 | measured variance ÷ 2 |
| 1.4 Delta MC under common random numbers | ✅ | 30/30 | 1.2 | `test_delta_crn` vs N(d1) |
| 1.5 Digitals (call-spread replication) | ✅ | 30/30 | 1.1 | `test_digital_call_bs_vs_mc` (CI) + `test_digital_call_replication_converge_en_h2` (ratio ~4.000) |
| 1.6 Multi-step `gbm_paths` (vectorised cumsum) | ✅ | 30/30 | 1.2 | `test_gbm_paths_shape_et_depart`, `_call_europeen`, `_loi_independante_de_n_steps` |
| 1.7 DI / DO put barriers + pathwise parity | ✅ | 40/40 | 1.6 | `test_di_do_van` (DI + DO = vanilla, 1e-12) |
| 1.8 Implied vol (Newton, Manaster-Koenig seed) | ✅ | 30/30 | 1.1 | `test_implied_vol_call` (vega-dependent tolerance) |

### Session 8 — quest 1.5

Call spread `(C(K-h) - C(K+h))/(2h)` against `digital_call_bs`: measured
convergence ratio **3.999 / 3.9998 / 4.000** each time `h` is halved
(0.4 → 0.2 → 0.1 → 0.05) — exact O(h²), to three decimal places of 4.
Against the closed form itself, `digital_call_bs` vs MC (N=200,000):
`0.53232` against `0.53256 ± 0.00207`, inside the CI.

Remaining technical debt, unscored: discrete monitoring (price vs `n_steps`,
Broadie-Glasserman-Kou) and delta near a barrier — both were on the barrier
lab's syllabus and were never done.

## ACT II — Variance reduction (230 / 230 XP) ✅

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 2.1 Generic `control_variate(Y, X, EX, c=None)` | ✅ | 30/30 | 1.2 | `pytest -k test_cv_` — 7 XPASS |
| 2.2 Degenerate case Y = X = vanilla put (tutorial boss) | ✅ | 20/20 | 2.1 | `test_cv_degenere_Y_egal_X` — exact BS price to 1e-12 |
| 2.3 Wiring into the DI put | ✅ | 40/40 | 2.1, 1.7 | `pytest -k di_put` — 4 XPASS |
| 2.4a Multi-step `gbm_paths_antithetic` | ✅ | 20/20 | 1.6 | `test_gbm_paths_antithetic_partage_Z` |
| 2.4b Antithetics × control (pair FIRST) | ✅ | 20/20 | 2.4a, 2.1 | `pytest -k antithetic` — 2 XPASS |
| 2.5 `pilot_c` — c fixed from a pilot run | ✅ | 20/20 | 2.3 | `pytest -k pilot` — 3 XPASS |
| **BOSS 2** Barrier sweep 60% → 95% + figure | ✅ | 80/80 | 2.3, 2.5 | `test_boss2_artefacts` — XPASS |

Where each TODO block lives:
`scripts/boss2_barrier_sweep.py` → BOSS 2. `src/mc_engine.py`: nothing left
for Act II.

### Session 5 — measurements for 2.5 (`pytest -k pilot`)

DI put H=90, S0=100, K=100, r=0.05, σ=0.20, T=1, n_steps=20:

    pilot c (10,000 paths) = 1.0013     full c (100,000) = 1.0003
    raw ref = 5.29734 +/- 0.03840        mean CV (fixed c) = 5.27146 +/- 0.00391

`c ≈ 1` because at H=90 the DI put is nearly the vanilla put. The pilot,
10× shorter, gives `c` to within **0.1%** — empirical demonstration that the
variance is **flat** around `c*`: a coarse `c` only costs a fraction of the
gain, never a bias. Half-width 10× tighter, and this time with a
**deterministic** `c` with respect to the final sample, so
`E[c(X̄−EX)] = 0` exactly — the O(1/N) bias from 2.1 has vanished.

Interview answer: "you estimate c on the same sample as the price, what's
the problem?" → `ĉ` and `X̄` are correlated, so `E[ĉ(X̄−EX)] ≠ 0`: a bias in
O(1/N), negligible compared to the O(1/√N) MC error — but a desk producing a
daily mark on the same generator sees a systematic shift, not noise that
averages out.

### Session 5 — measurements for 2.4 (`test_cv_antithetic_domine_chaque_technique_seule`)

Vanilla put S0=K=100, r=0.05, σ=0.20, T=1, n_steps=25, budget **2N = 50,000
paths for all four estimators** (comparing at a different budget would be
meaningless). Control = discounted forward, E[e^{-rT} S_T] = S0 exactly.

| Estimator | half-CI | **variance** gain |
|---|---:|---:|
| raw | 0.07575 | 1× |
| AV alone | 0.05783 | 1.72× |
| CV alone | 0.04939 | 2.35× |
| AV + CV | 0.02372 | **10.2×** |

Measured correlations: ρ(Y_up, Y_down) = **−0.416** → AV gain = 2/(1+ρ) = 1.71
(the factor of 2 would assume ρ = 0; it does better because the put is
monotonic in Z). ρ(put, fwd) = **−0.758** → CV gain = 1/(1−ρ²) = 2.35.

The non-trivial point: 1.72 × 2.35 = 4.0, yet the measurement is **10.2**.
After pairing, ρ(put, fwd) goes from −0.758 to **+0.912** — the per-pair
average is an **even** function of Z, so the paired put and paired forward
both grow with |Z| and become nearly collinear. AV did not just reduce
variance: it **improved the control**. Not to be generalised (specific to
this pair), but the remark to make in an interview.

An asymmetry between AV and CV worth stating: a poorly chosen CV never
degrades (c* → 0), a poorly chosen AV can **degrade** things — a
non-monotonic payoff in Z (straddle) → ρ > 0 → variance rises at an equal
budget.

### Session 4 — measurements for 2.3 (`scripts/run_s4.py`)

S0=100, K=100, T=1, r=0.02, σ=0.30, n_steps=252, N=100,000, seed=20240904:

|       H | MC price | half-CI | CV price | half-CI | rho_hat | CI gain | KI+KO−van parity |
|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-----------------:|
| 65.0000 |  5.5295 |  0.0833 |  5.4965 |  0.0500 |  0.7998 | 1.6658× |       3.5527e-15 |
| 85.0000 | 10.5477 |  0.0880 | 10.5044 |  0.0100 |  0.9936 | 8.8272× |       3.5527e-15 |

The gain equals `1/√(1−ρ²)`: 1.667 expected / 1.666 measured at ρ=0.80, 8.90
/ 8.83 at ρ=0.9936. ρ measures the fraction of paths where the DI put
**coincides** with its control: at H=85 (~0.5·σ√T below spot in log terms)
almost every in-the-money path has triggered, the control explains 99% of
the variance; at H=65 (~1.4·σ√T) the indicator decouples the two payoffs. A
**non-linear** relationship — a "correct" control barely pays off, only a
near-perfect control does. Corollary verified at H=60: a bad control never
degrades things (c*→0).

### BOSS 2 — the sweep (`scripts/boss2_barrier_sweep.py`)

S0=100, K=100, σ=0.20, r=0.05, T=1, n_steps=50, N=100,000, seed=42, **a
single set of paths for all 8 barriers** (CRN: otherwise ρ(H) would shake
with MC noise and it would be impossible to tell whether a dip is a real
effect or an artefact).

| H/S0 | ρ | half-CI ratio | c_hat | MC price | CV price |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.2989 | 0.9543 | 0.1035 | 0.2296 | 0.2346 |
| 0.70 | 0.6211 | 0.7838 | 0.4502 | 1.3391 | 1.3605 |
| 0.80 | 0.8847 | 0.4661 | 0.8669 | 3.5978 | 3.6390 |
| 0.90 | 0.9918 | 0.1279 | 1.0003 | 5.2880 | 5.3355 |
| 0.95 | 0.9995 | 0.0303 | 1.0009 | 5.5017 | 5.5493 |

A 33× gain on the half-width at the top point, i.e. **1000× in variance**.
`c_hat` climbs from 0.10 to 1.00: at H=95% the DI put **is** the vanilla
put. At H=60% the trigger is rare, `c` collapses towards 0 and the estimator
folds back on its own onto the raw MC — **which is why the ratio never
exceeds 1**. The answer to "and if your control is poorly chosen?".

`price_cv − price_mc ≈ +0.047` everywhere, **proportional to `c_hat`**: this
is `c·(X̄−EX)`, the correction measured on this set of paths. A gap that is
*not* proportional to `c` would signal a bug.

**A limit of the figure, worth stating (found by Arthur, not by the test).**
The "theoretical" curve √(1−ρ̂²) is **not** an independent validation: with
`c = Cov/Var` estimated on the sample, `Var(Z) = Var(Y)(1−ρ̂²)` holds
identically, so ratio = √(1−ρ̂²) **by algebra**. Measured gap at the 8
points: **1e-15**, machine precision. What this actually tests: internal
consistency (same `ddof` between `cov`, `var` and `std`, half-width computed
on the residual and not on Y, same `n` on both sides). A genuine external
reference would require Reiner-Rubinstein's closed form — which assumes
**continuous** monitoring, while the monitoring here happens over 50 steps
(the Broadie-Glasserman-Kou technical debt). Presenting a tautology as a
validation is the kind of thing that gets expensive if an examiner pushes
back.

## ACT III — Single-asset Heston (210 / 210 XP) ✅

🔓 **OPENED.** TODO blocks in `src/heston.py`, tests in `tests/test_heston.py`
(16 xfailed at opening), boss in `scripts/boss3_heston_barrier.py`.

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 3.1 `heston_paths` — Euler, 2 correlated Brownians | ✅ | 40/40 | 1.6 | `pytest -k paths` — 5 XPASS |
| 3.2 `heston_cf` / `heston_call` / `heston_put` — semi-analytical | ✅ | 50/50 | 3.1 | 6 XPASS: BS limit, parity, MC agreement |
| 3.3 `heston_smile` — implied vol by strike | ✅ | 40/40 | 3.2, 1.8 | `pytest -k smile` — 4 XPASS |
| **BOSS 3** DI put under Heston + CV + smile figure | ✅ | 80/80 | 3.3, Act II | `test_boss3_artefacts` — XPASS |

The act's thread: **simulate** (3.1) → build an **analytical reference**
(3.2) → **read** what the model produces (3.3) → **plug Act II back in on
top** (BOSS 3). Every validation is model-independent: the martingale
identity `E[e^{-rT}S_T]=S0`, the CIR's exact mean, the degenerate limit
ξ→0 towards Black-Scholes, call-put parity. No test compares Heston against
Heston.

### Session 6 — quest 3.1

`E[v_T]` expected 0.051157 / measured 0.050911 (0.5%), imposed `ρ` −0.7 /
measured −0.6986. Martingale and BS limit inside the CI.

**A scheme decision to own.** The returned `v` array is an *output*: a
negative variance in it would be a `nan` waiting at the first `sqrt`
downstream. Hence the split between internal state and reported output
(working vector `v_raw`, truncated value written into the array). Watch the
detail that changes the scheme: if the recursion reads back the
**truncated** value, this is no longer full truncation but the **absorbed**
scheme — full truncation keeps the memory of the negative excursion, the
absorbed variant resets it to zero. Measured gap here: 1e-5 relative,
because Feller holds (2κθ = 0.12 > 0.09 = ξ²). **On market-calibrated
parameters, Feller is almost always violated** and the gap becomes visible.
Lord, Koekkoek & van Dijk (2010): full truncation is the least biased Euler
variant.

### Session 6 — quest 3.2

Route chosen: Heston 1993 / Gatheral, `P1`/`P2` via `scipy.integrate.quad`,
Albrecher's form for the characteristic function (no branch jump).

| K | Heston | BS(20%) | reading |
|---:|---:|---:|---|
| 80 | 25.095 | 24.589 | low strikes **more expensive** |
| 100 | 10.362 | 10.451 | ATM roughly aligned |
| 120 | 2.193 | 3.247 | OTM calls **a third cheaper** |

The ρ = −0.7 skew, visible even before writing 3.3.

**Semi-analytical / MC agreement** (N=200,000, n_steps=250):
`K=90: MC 17.0984±0.0644 vs exact 17.1069` · `K=100: 10.3505±0.0523 vs 10.3619`
· `K=110: 5.3053±0.0380 vs 5.3180`. Inside the CI, but **the exact value sits
above at all three strikes, by ~0.012 each time**. The same sign three times
= an Euler discretisation bias, not noise. Verifiable: doubling `n_steps`
must halve the gap (Euler is O(dt)).

**Fixing a wrong test (my own).** `test_heston_cf_limite_gaussienne` demanded
1e-6 at ξ=1e-3: impossible, the real gap there is 3.5e-4. The departure from
the gaussian is O(ξ), not O(ξ²) — the skew term `ρ·ξ·u³` is first order,
since it is the correlation that breaks the symmetry (curvature, meanwhile,
is O(ξ²)). Measured: 3.48e-3 → 3.48e-4 → 3.48e-5 for ξ = 1e-2, 1e-3, 1e-4,
and scaling in u³ at fixed ξ. Below ξ ≈ 1e-5 the error **rises again**
(catastrophic cancellation via `κθ/ξ²`). The test now checks a
**convergence ratio** (measured factor of 10: 10.00) rather than a
threshold — a wrong formula misses the slope, not just the level.

### Session 6 — quest 3.3: ρ drives the slope, ξ drives the curvature

Smile at ρ=−0.7, K from 80 to 120 (S0=100, F=105.13, T=1):
`0.2326 0.2235 0.2147 0.2061 0.1976 0.1895 0.1817 0.1744 0.1678`
Fitted slope in log-moneyness: **−0.1613** — a realistic 1-year equity-index
skew. ATM vol 19.76% against √θ = 20% (the forward sits above spot).

Curvature (second difference in log-moneyness, ρ=0 to isolate ξ):

| ξ | curvature | ratio vs ξ=0.1 | expected ξ² |
|---:|---:|---:|---:|
| 0.1 | 0.00088 | 1 | 1 |
| 0.2 | 0.00353 | 4.01 | 4 |
| 0.3 | 0.00781 | 8.9 | 9 |
| 0.5 | 0.01952 | 22.2 | 25 (saturation) |

**Curvature ∝ ξ² to three significant figures.** Two independent
measurements agree: the characteristic function gave a first-order skew term
`ρ·ξ·u³` (session 6, 3.2), the vol surface gives a second-order curvature.

Desk translation: ρ drives the **risk reversal** (linearly), ξ drives the
**butterfly** (quadratically). And it is the risk reversal that decides a
BRC's price, because the down-and-in put sold by the investor lives in the
left wing — exactly where the skew makes vol expensive. Hence "why Heston
and not BS".

### BOSS 3 — the DI put under Heston (`scripts/boss3_heston_barrier.py`)

S0=K=100, v0=0.04, κ=1.5, θ=0.04, ξ=0.3, ρ=−0.7, r=0.05, T=1, n_steps=50,
N=100,000, seed=42, **a single simulation for all 20 barriers** (CRN).
Control = vanilla put on the same paths, `EX = heston_put` (3.2).

| H/S0 | ρ(payoff, control) | half-CI ratio | c_hat | MC price | CV price |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.6390 | 0.7692 | 0.4557 | 1.3064 | 1.2966 |
| 0.70 | 0.8267 | 0.5626 | 0.7501 | 2.7142 | 2.6980 |
| 0.80 | 0.9497 | 0.3132 | 0.9483 | 4.3431 | 4.3227 |
| 0.90 | 0.9960 | 0.0894 | 1.0003 | 5.3445 | 5.3229 |
| 0.95 | 0.9997 | 0.0229 | 1.0005 | 5.4864 | 5.4648 |

A **44×** gain on the half-width at the top point (~1900× in variance).

Three readings:

1. **`c_hat` starts at 0.46**, against 0.10 under Black-Scholes at BOSS 2.
   Under Heston the DI put resembles the vanilla put more, from H=60%
   already: the skew thickens the left wing, so paths that end up
   in-the-money more often have touched a low barrier.
2. **`price_cv − price_mc = −0.0216·c_hat`**, exactly proportional to
   `c_hat` → the control is centred, `EX` is consistent with the paths.
   The "BS `EX` on Heston paths" pitfall was avoided.
3. **`heston_put`(ATM) = 5.4848 against `put_bs`(20%) = 5.5735**, i.e.
   −0.089: what the smile costs on an ATM vanilla. Small. On a low-barrier
   DI put the model gap is much larger — this is the "why Heston and not
   BS" argument for a BRC.

**Act II's machinery reconnects without a single line of modification.**
Only the source of `EX` changes: `put_bs` → `heston_put`. Variance
reduction is a statistical technique, independent of the model.

The three pitfalls that will cost the most, flagged in advance:
1. **A forgotten `max(v,0)`** → a silent `nan` propagated across the whole
   path.
2. **The branch cut** in the characteristic function (the "little Heston
   trap"): correct price at T=1, nonsensical at T=5.
3. **`v_{t+dt}` used in the spot's step** instead of `v_t` — the first-step
   correlation test is written precisely to catch this.

Next: ACT IV — discrete hedging (see below), which tackles exactly the last
piece of technical debt (delta near a barrier, quest 4.5). Still open:
digitals (1.5, 30 XP), discrete monitoring, and — after Act IV — multi-asset
via Cholesky → worst-of → autocall → BRC.

---

## ACT IV — Discrete hedging (340 / 340 XP) ✅

TODO blocks in `src/hedging.py`, tests in `tests/test_hedging.py` (12 green,
4.1 to 4.6 and `test_boss4_artefacts`), boss in `scripts/boss4_hedging.py`
(`figures/boss4_hedging.png`, `figures/boss4_results.json`).

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 4.1 Self-financed portfolio | ✅ | 30/30 | — | `test_autofinancement_deltas_arbitraires` (1e-12, arbitrary deltas) + 2 degenerate cases |
| 4.2 Hedging error under the correct model | ✅ | 50/50 | 4.1 | `test_hedging_error_moyenne_dans_ic_et_pente_log_log` — measured slope -0.485 |
| 4.3 Vol arbitrage / wrong model | ✅ | 50/50 | 4.2 | `test_vol_arbitrage_identite_gamma` — gamma-weighted identity, MINUS sign |
| 4.4 Transaction costs | ✅ | 40/40 | 4.2 | `test_transaction_costs_frequence_optimale` — cost slope +0.44, intermediate RMS optimum |
| 4.5 Delta near a barrier | ✅ | 40/40 | 4.2 | `test_naive_barrier_hedge_degrade_pres_de_la_barriere` — std ratio 2.82, slope ≈0 (not -0.5) |
| 4.6 Delta-vega under Heston | ✅ | 50/50 | 4.2, Act III | `test_heston_delta_vega_reduit_la_variance` — variance ratio 3.05 |
| **BOSS 4** Convergence + P&L (figure) | ✅ | 80/80 | 4.2, 4.3, 4.4 | `test_boss4_artefacts` — slope -0.491, 3 distinct P&L scenarios |

The act's dependency tree:

```
4.1 (self-financing)
  └─ 4.2 (hedging error, the common brick)
       ├─ 4.3 (vol arbitrage)
       ├─ 4.4 (transaction costs)
       ├─ 4.5 (barrier)
       └─ 4.6 (Heston, also depends on Act III)
            └─ BOSS 4
```

The act's thread, three interview-defensible results: the **convergence** of
the hedging error at `n_rebal^{-1/2}` (4.2), the **vol-arbitrage identity**
(4.3), and the **degradation of delta near a barrier** (4.5) — the real
business problem, the one with no convergence to show, just a break to
document.

### Session 7 — the most expensive bugs

Unlike the earlier acts, most of this act's bugs were **not** in the
student's code — three landed in tests written by the assistant, a good
reminder that "the test is right" is not an axiom.

| Pitfall | Where | Nature | Cost |
|---|---|---|---|
| `B[-1]` instead of `B` in the `V_T` computation | 4.1 | scalar indexing of a per-path vector | invisible on 2 of the 3 tests (degenerate deltas, where `B` is identical across all paths by construction) — caught only by the ARBITRARY-deltas test |
| `deltas` received as a parameter but recomputed internally with a hard-coded `sigma=0.2` | 4.2 | ignored parameter + magic number | invisible as long as the test's `sigma` happened to match 0.20 |
| **Test reference missing `exp(-r*t_i)` in the gamma sum** | 4.3 | incomplete reference formula (a bug in the test, not the code) | stable ~2.4% gap, that did NOT SHRINK with `n_steps` — cast doubt on the student's code before an Ito derivation revealed the missing term (the hedging error itself compounds at rate r) |
| An extra `turnover += np.sum(...)` line, left over from an earlier version | 4.4 | dead code double-counting a total and re-injecting it into every column | log-log cost slope measured at 1.48 instead of ~0.5 — spotted by comparing against the share-only turnover's slope |
| Vega overlay added with the wrong sign (`+n_vega*delta` instead of `-n_vega*delta`) | 4.6 | confusion between long and short position in `bs_delta_hedge_deltas`'s convention | measured variance ratio below 1 (the overlay WORSENED the variance) instead of clearly above |

Cross-cutting lesson: a test that gives a plausible result (right sign,
right order of magnitude) can still be wrong by a missing term — the check
that unblocked 4.3 was not "the sign is right" but "the gap does not shrink
with `n_steps`, so this is not a discretisation bias, it is an incomplete
formula".

---

## ACT V — Multi-asset and structured products (350 / 350 XP) ✅

Decisions locked in with Arthur before opening: non-PD correlation repair by
**eigenvalue clipping** (not Higham nearest-correlation — simpler, the
trade-off being a repaired matrix that is close but not Frobenius-optimal);
default **2 assets** for the test fixtures; worst-of barrier convention is
`min over assets AND time` (any asset touching H at any date triggers).

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 5.1 Cholesky + non-PD correlation repair | ✅ | 40/40 | 1.6 | `pytest tests/test_multi_asset.py -k "5_1 or repair or correlated"` — 8 XPASS |
| 5.2 Worst-of DI put + degenerate cases | ✅ | 50/50 | 5.1, 1.7 | N=1 → mono-asset DI put exactly (scaled by s0); rho=1 (equal vols) → mono-asset DI put reduction (exact test dropped, Cholesky precision floor, see below) |
| 5.3 Discrete vs continuous monitoring | ✅ | 40/40 | 5.2 | `barriers.bgk_barrier_correction`/`di_put_continuous` close the Broadie-Glasserman-Kou debt open since 1.7 (mono-asset, convergence vs a quasi-continuous reference); `worst_of_di_put_continuous` reuses it per-asset as a heuristic (no exact theory for a min of correlated GBMs) |
| 5.4 BRC — coupon / short-put decomposition | ✅ | 40/40 | 5.2 | exact decomposition (1e-10) + degenerate case (barrier far away) → pure coupon + zero-coupon |
| 5.5 Autocall / Phoenix | ✅ | 50/50 | 5.2, 5.4 | single recall date → exactly `brc_price` (digital case); recall probability monotone in rho/vol |
| 5.6 Sensitivities — delta per underlying, vega, rho | ✅ | 50/50 | 5.2, Act II/IV | `worst_of_deltas` is IDENTICALLY zero (not a bug — see below); vega positive; rho-sensitivity sign checked |
| **BOSS 5** Full BRC worst-of pricing + figure | ✅ | 80/80 | 5.1–5.6 | `test_boss5_artefacts` — correlation sweep (CV reused from Act II/III) + n_assets sweep |
| *Bonus, unscored* Heston multi-asset extension | — | — | 5.1, Act III | not attempted — superseded by Arthur's Act VI scope widening (multi-asset Heston calibration), see top banner |

Core total: 350 XP.

### Session 9 — the non-obvious result: worst-of delta is EXACTLY zero

The scaffold's first version of 5.6 assumed `worst_of_deltas < 0` (a put
loses value as spot rises). Measured: `worst_of_deltas` returns `[0., 0.]`,
bit-exact, every time. Not a bug: K and H are in PERFORMANCE (convention
locked at 5.2), so the payoff only ever reads `paths/paths[:,:,0] =
exp(increments)` — S0 cancels ALGEBRAICALLY before any Monte-Carlo noise
enters. Bumping S0 can never move a 100%-performance payoff. The test was
rewritten to assert `abs(delta) < 1e-8` (a model-free identity, not a
measurement) instead of a sign. Desk reading: a quanto/performance basket has
no "spot delta" by construction — the only live Greeks are vega and the
correlation sensitivity (rho), which `worst_of_rho_sensitivity` (bump +
centered CRN difference on `rho[i,j]`) and the boss figure's sweep do cover.

### Session 9 — control variate on the BRC: small but real gain

BOSS 5's control is `X = exp(-rT)*paths[:,0,-1]` (one asset's discounted
terminal value, `EX = S0[0]`, martingale identity — weak but genuinely
model-free, no closed form needed for a worst-of). Measured at rho=0.5:
`price_mc = 0.974482 ± 0.000515`, `price_cv = 0.974424 ± 0.000476` — same
price (by construction, CV is unbiased for the same expectation), ~8% CI
gain. `c_hat ≈ 0.001`, i.e. weak correlation between a single-asset payoff
and a worst-of-on-two-assets payoff — consistent with Act II's lesson (BOSS
2): a weak control gives a weak gain, never a bias. A stronger control (an
analytical approximation of the worst-of itself) was identified as open work,
not attempted.

### Session 9 — the 5.2 test retired (Cholesky precision floor)

`test_worst_of_di_put_rho1_vols_egales_reduit_a_un_mono_actif` was dropped,
not fixed: rho=[[1,1],[1,1]] is exactly rank-1 (one eigenvalue exactly 0), so
any `tol>0` in `repair_correlation_clip` leaves a residual price gap
proportional to `tol` (measured near 1:1). Pushing `tol` below ~1e-14
approaches machine precision but destabilises the Cholesky pivot. The
degenerate identity `perf_0 == perf_1` (not the priced put) remains the valid
model-free check for this case, and is still tested.

### Session 9 — pitfalls hit (not to repeat)

| Pitfall | Where | Cost |
|---|---|---|
| `paths[:, :, 0]` (shape `(N, n_assets)`) passed where a 1D `S0` was expected | 5.2 (`worst_of_di_put`, first draft) | `n_assets = len(S0)` silently became `N` → `Z` shape `(N, n_steps, N)`, a memory blow-up that looked like an infinite loop |
| `half_width` computed on the undiscounted payoff | 5.2 | price right, CI wrong by a factor `1/exp(-rT)` — caught by the mono-asset degenerate test, not by eyeballing the price |
| `src/multi_asset.py` importing a constant (`S0`) from `tests/test_multi_asset.py` | 5.6 | circular import, crashed collection for the whole test file — the import was dead code (every function already shadows `S0` with its own parameter) |
| Barrier-gated payoff multiplying the WHOLE redemption by `knock_in` instead of just the put term | 5.5 (`autocall_price`) | paths that never breached the barrier got paid **zero** instead of full nominal+coupon — backwards from the product's actual protection |
| `control_variate(paths, lambda, lambda)` — wrong call shape entirely | BOSS 5 | `control_variate` takes two **payoff arrays** `(Y, X)` and a scalar `EX`, never paths or callables |
| `Y`/`X` built outside the `rho` sweep loop, or built from `paths` before they were regenerated for the new `rho` | BOSS 5 | `price_cv`/`half_width_cv` identical across the whole sweep, decorrelated from the actual `paths` being priced that iteration |
| A test asserting `worst_of_deltas < 0`, assuming S0 affects a performance-normalized payoff | 5.6 (my own scaffold, not Arthur's code) | fixed by recognising the identity (delta ≡ 0) instead of chasing a sign that can't exist |

### Pitfalls watched by Act II's tests

| Pitfall | Test that catches it |
|---|---|
| `np.max` instead of `np.maximum` (and `min` without `axis=1`) | `test_di_put_payoffs_valeurs_a_la_main` |
| Precedence: `cov/sd_Y*sd_X` instead of `cov/(sd_Y*sd_X)` | `test_cv_rho_invariant_par_echelle` |
| Parenthesisation: `Y - c*X - EX` instead of `Y - c*(X - EX)` | `test_cv_utilise_bien_EX` |
| A leaking global variable (the phantom `N`, twice already) | `test_cv_pas_de_N_fantome` |
| Inconsistent discounting between Y, X and EX | `test_di_put_payoffs_actualisation_coherente` — **fell into the S4 trap** |
| Half-width computed on Y instead of the residual Z | `test_cv_half_width_sur_le_residu` |
| Dividing by 2N instead of N (the number of pairs) | `test_cv_antithetic_N_est_le_nombre_de_paires` |
| Code shown without having been run | all of them — a test must run, not just be read |

### Pitfalls hit in session 4 (not to repeat)

| Pitfall | Where | Cost |
|---|---|---|
| `np.cov(ddof=1)` vs `np.var(ddof=0)` — different normalisations | 2.1 | `c_hat` wrong by a factor n/(n−1): 5e-5 against a 1e-10 tolerance |
| `np.correlate` ≠ `np.corrcoef` | 2.1 | signal cross-correlation, not Pearson |
| `np.corrcoef(...)` returns a **2×2 matrix**, not a scalar | 2.1 | `assert` on an array → `ValueError: truth value ambiguous` |
| Parameter `c` ignored (recomputed unconditionally) | 2.1 | took 4 iterations to see it; `c=0` must reproduce the raw MC |
| **Undiscounted** payoffs returned by `di_put_payoffs` | 2.3 | X̄−EX wrong by a factor `e^{rT}` **and the opposite sign**: a silent bias |
| `van_put` is a **pricer** (returns a tuple), not a payoff vector | 2.3 | a finance/stats layer mix-up |
| Spot **hard-coded** in `put_bs(100, ...)` | 2.3 | green tests (all at S0=100) but the price wrong by 11 points at S0=80, with a null CI |
| `paths[:, 0]` (array) instead of `paths[0, 0]` (scalar) | 2.3 | 100,000 identical BS computations, 800 KB, 50× slower |

### Pitfalls hit in session 5 (not to repeat)

| Pitfall | Where | Cost |
|---|---|---|
| Calling `gbm_antithetic` (single-step) to do multi-step work | 2.4a | `n_steps` never appeared in the body — the tell: an unconsumed parameter |
| 6 positional arguments for a 7-argument signature | 2.4a | `N` → `n_steps`, `rng` → `N`: an opaque `TypeError`. Name the arguments |
| Reusing `gbm_paths` (which **caches** its `Z`) to build pairs | 2.4a | impossible by construction: a function that caches its own randomness is not composable — the same reason `rng` is injected into `delta_mc` |
| Passing the same `rng` object twice believing it replays the same draws | 2.4a | a `Generator` has **state**: it advances. Two `default_rng(3)` calls would have given `up == down` |
| `S0np.exp(...)` (a missing `*`) | 2.4a | not a `SyntaxError` — a valid attribute access → a `NameError` at runtime, swallowed by `xfail` |
| Copy-paste: `X_pair = (Y_up + Y_down)/2` | 2.4b | falls back onto the degenerate case 2.2 → half-width **exactly 0** and a green test. Caught only by the one test asserting an **equality**, not an inequality |
| `np.cov(Y, X)` with `Y`/`X` absent from the signature (params: `Y_pilot`/`X_pilot`) | 2.5 | **3rd occurrence** of the phantom global. Saved by the `NameError`, for lack of a same-named global — with an `X` at module level, it would have been a silently wrong `c` |
| `[0, 1]` and `float(...)` forgotten on `np.cov` | 2.5 | already hit in 2.1, correctly rewritten at line 133 then redone 150 lines further down |

Cross-cutting lesson from the session: `xfail(strict)` **swallows any
exception** (a typo, a `NameError`, a `TypeError`) and displays it as an
expected XFAIL. As long as a test is marked, it diagnoses nothing — calling
the function by hand in a REPL is the only way to see the real error.

The thread: S4's last 4 pitfalls are **invisible without an external
reference**. An off-centre control yields a plausible price and a
half-width that narrows — the more wrong it is, the more precise it looks.
Hence the repo's rule: validate against a closed form or a parity identity,
never against "it looks about right".

---

## ACT VI — Heston calibration on real market data, multi-asset (0 / 400 XP) 🔓

🔓 **OPENED — quest 6.1 only.** TODO block in `src/market_data.py` (new),
tests in `tests/test_market_data.py` (new, 2 xfailed at opening). Quests 6.2
through BOSS 6 are locked — not scaffolded, no files, no TODO — until 6.1 is
green. Data: yfinance, 3 commodity ETFs (GLD, SLV, USO — raw commodity
futures have no option chain on yfinance, only the ETFs that replicate them
do). Scope: two-step multi-asset (mono-asset Heston per name, THEN
correlation reconstructed from historical returns — no joint calibration).
SVI: per-name second baseline. 6.6 is Arthur's addition: a temporal backtest.

| Quest | Status | XP | Depends on | Validation |
|---|---|---|---|---|
| 6.1 Market data ingestion & cleaning (yfinance, GLD/SLV/USO) | 🔓 | 0/40 | 1.1 | `test_clean_option_chain_respecte_parite_forward` — forward parity on cleaned mid-quotes, tolerance scaled to bid-ask spread |
| 6.2 Per-name Heston calibration on the market smile | 🔒 | 0/60 | 6.1, 3.2/3.3 | TBD when opened |
| 6.3 Per-name SVI calibration (second baseline) | 🔒 | 0/50 | 6.1, 1.8 | TBD when opened |
| 6.4 Correlation reconstructed from historical returns | 🔒 | 0/30 | 6.1, 5.1 | TBD when opened |
| 6.5 Heston vs SVI vs flat BS — intra-day out-of-sample (strike/expiry holdout) | 🔒 | 0/40 | 6.2, 6.3 | TBD when opened |
| 6.6 Temporal backtest — parameter stability + J+1 predictive power | 🔒 | 0/50 | 6.2, 6.3 | TBD when opened |
| 6.7 Worst-of/BRC pricing on real calibrated marginals + reconstructed correlation | 🔒 | 0/50 | 6.2, 6.4, Act V | TBD when opened |
| **BOSS 6** Fully market-calibrated BRC worst-of + figure | 🔒 | 0/80 | 6.1–6.7 | TBD when opened |

### 6.6 — the hypothesis to settle (Arthur's addition, not originally scoped)

SVI is a purely descriptive fit (5 free parameters per day, no temporal
constraint): it will fit the day's smile better than Heston almost always —
expected, not a result. Heston imposes structure (CIR + correlation).
Hypothesis under test: **Heston, calibrated on a rolling window of days,
gives day-to-day parameters that are more STABLE than SVI's, and its day-J
parameters predict the day-J+1 smile better (no recalibration) than SVI's —
even though SVI wins on intra-day RMSE.** Planned validation (once 6.6 is
scaffolded, after 6.2/6.3): day-to-day coefficient of variation per
parameter (Heston vs SVI) on a real rolling window, plus a J+1 *prediction*
RMSE (parameters frozen at day J, applied to the day-J+1 smile) compared
across Heston, SVI, and a naive "yesterday's vol" baseline. The pedagogical
trap this is built to catch: a better intra-day fit does not imply better
predictive power — that has to be shown with numbers, not assumed.
