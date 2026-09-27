# exotic-pricing-lab

A small Monte-Carlo pricing library for equity derivatives, built so that every
component is validated against an independent reference rather than against
itself: a closed-form price, a model-free identity, or an exactly known moment.

It covers European options under Black-Scholes, discretely monitored
down-and-in / down-and-out puts, two variance reduction techniques, the
single-asset Heston model with both a Monte-Carlo and a semi-analytical route
to the same prices, and discrete delta-hedging: a self-financed replicating
portfolio, its convergence rate, the P&L of hedging at the wrong volatility,
transaction costs, and what breaks near a barrier or under a model with a
smile.

## Getting started

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install numpy scipy matplotlib pytest
.venv/bin/python -m pytest -v
```

The editable install exposes the modules under flat names (`from bs import
call_bs`). `pyproject.toml` declares no dependencies, so numpy, scipy,
matplotlib and pytest are installed explicitly above.

Reproducing the figures:

```bash
.venv/bin/python scripts/boss2_barrier_sweep.py    # -> figures/boss2_*
.venv/bin/python scripts/boss3_heston_barrier.py   # -> figures/boss3_*
.venv/bin/python scripts/boss4_hedging.py          # -> figures/boss4_*
```

## Modules

| Module | Contents |
|---|---|
| `src/bs.py` | Closed-form European call and put with continuous dividend yield, finite-difference delta, vega, a cash-or-nothing digital call and its call-spread replication |
| `src/mc_engine.py` | Single-step and multi-step GBM simulation, antithetic variates, delta under common random numbers, control-variate estimators, pilot coefficient |
| `src/barriers.py` | Down-and-in / down-and-out / vanilla put payoffs on a path array, and the control-variate DI put under Black-Scholes |
| `src/implied_vol.py` | Black-Scholes implied volatility by Newton-Raphson, seeded with Manaster-Koenig |
| `src/heston.py` | Heston paths by Euler discretisation, characteristic function in Albrecher's form, semi-analytical European prices by Fourier inversion, implied volatility smile |
| `src/hedging.py` | Self-financed replicating portfolio, Black-Scholes delta hedge and its convergence rate, closed-form gamma, vol-arbitrage P&L, proportional transaction costs, a naive barrier hedge, and a static delta-vega hedge under Heston |

Two conventions hold throughout:

- the second return value of every estimator is a **95% confidence half-width**,
  `1.96 * sd / sqrt(n)`, never a raw standard error (`delta_mc` is the one
  exception and returns no interval at all);
- randomness is injected through an `rng` argument rather than seeded inside a
  function, so callers can pair samples across calls.

## Validation

62 tests. What each layer is checked against:

| Component | Reference |
|---|---|
| Black-Scholes | 10.4506 at S0=K=100, sigma=0.2, r=0.05, T=1; the dividend identity `C(S0, q) == C(S0*exp(-qT), 0)` to 1e-14 |
| Digital call | an independent Monte-Carlo estimate of the indicator payoff, within the confidence interval; the call-spread replication converges to it at O(h^2), tested as a rate (ratio ~4 when h is halved), not a threshold |
| Monte-Carlo pricers | the closed-form price, within the confidence interval |
| `gbm_paths` | the closed form; terminal law independent of `n_steps` |
| `delta_mc` | N(d1) |
| Barriers | `DI + DO = vanilla` pathwise, to 1e-12 — an algebraic identity, so the Monte-Carlo noise cancels exactly |
| Control variate | c = 0 reproduces the raw estimator; Y = X returns the closed form with a null half-width; invariance under rescaling of the control |
| Antithetic x control | exact equality against pairs-then-control, to 1e-12; and at an equal 2N path budget the combination beats each technique alone |
| Heston paths | `E[e^{-rT} S_T] = S0`; the exact CIR mean `theta + (v0-theta)exp(-kappa T)`; the degenerate limit xi=0, v0=theta returning Black-Scholes; the first-step correlation returning rho |
| Heston characteristic function | `phi(0)=1`, `abs(phi)<=1`, `phi(-u)=conj(phi(u))`; O(xi) convergence towards the gaussian limit, tested as a *rate* rather than a threshold |
| Heston prices | model-free bounds and monotonicity in K; call-put parity to 1e-10; agreement with a Monte-Carlo on the paths, within the interval |
| Smile | flat at sqrt(theta) when xi -> 0; decreasing in K when rho < 0; symmetric and convex when rho = 0; curvature growing with xi |
| Hedging | self-financing is an algebraic identity, to 1e-12, on arbitrary deltas; the discrete hedging error's standard deviation decreases as a *rate*, n_rebal^-1/2 (Boyle & Emanuel, 1980); the vol-arbitrage P&L matches a gamma-weighted integral to the confidence interval; transaction costs grow as n_rebal^1/2, giving a total-error curve with an interior minimum; a naive barrier hedge is shown to *degrade* near the barrier rather than converge; a static vega overlay under Heston is shown to cut the hedging error's variance |

## Results

**Control variate on a down-and-in put** (`figures/boss2_barrier_sweep.png`).
Barrier swept from 60% to 95% of spot, all levels priced on the same paths.
At H = 95% the DI put is nearly the vanilla put, the correlation reaches 0.9995
and the confidence interval narrows by a factor **33**, i.e. about 1000x in
variance. At H = 60% the knock-in is rare, the correlation falls to 0.30 and the
fitted coefficient collapses towards zero — which is why the ratio never exceeds
1: a poorly chosen control degrades gracefully into the raw estimator.

**The same setup under Heston** (`figures/boss3_heston.png`). Only the source of
the control's expectation changes, `put_bs` becoming `heston_put`; the variance
reduction code is reused unmodified. Gain of **44x** on the half-width at the top
of the sweep. The model produces a negative skew, implied volatility running from
23.3% at K=80 to 16.8% at K=120 for rho = -0.7.

**Discrete hedging** (`figures/boss4_hedging.png`). The hedging error's
standard deviation follows the `n_rebal^-1/2` line closely across a grid from 8
to 512 rebalancing dates (measured slope **-0.491**). At a fixed rebalancing
frequency, three P&L distributions sit side by side: hedging at the correct
volatility is centred at zero; hedging at 20% while the market realises 30%
shifts the distribution well into losses (mean P&L around -3.8, short-gamma
seller caught by a bigger move than priced in); adding proportional
transaction costs shifts it negative again, by a smaller and more predictable
amount, with a tighter spread.

## Not covered

Stated explicitly, since several of these are visible in the code:

- **Multi-asset.** No correlation between underlyings, no worst-of, no autocall,
  no barrier reverse convertible. The library stops at a single underlying.
- **Discrete monitoring bias.** Barriers are tested at the simulated dates only.
  A continuously monitored barrier knocks in more often, so these prices carry a
  discretisation bias that does not shrink as the number of paths grows. The
  Broadie-Glasserman-Kou continuity correction is not implemented, and there is
  no test of price convergence in `n_steps`.
- **The variance scheme.** `heston_paths` carries the truncated variance
  forward, which is the *absorption* variant of Lord, Koekkoek & van Dijk
  (2010), not their *full truncation*. The two agree closely when the Feller
  condition `2*kappa*theta > xi^2` holds, and diverge markedly when it does not
  — which is the case for most market-calibrated parameter sets.
- **`di_put_cv` is Black-Scholes-only.** It reads the control's expectation from
  the closed-form BS put. Pricing a DI put on Heston paths means calling
  `di_put_payoffs` and `control_variate` directly, as
  `scripts/boss3_heston_barrier.py` does.
- **Greeks beyond delta.** `delta_mc` returns no confidence interval, and delta
  near a barrier — where the finite-difference estimator degrades — is not
  addressed.
- **Calibration.** Heston parameters are given, never fitted to a quoted
  surface.
- **Degenerate inputs.** `T = 0`, `sigma = 0` and a constant control are not
  guarded and return `nan`.
- **Performance.** `heston_call` integrates with `scipy.integrate.quad`, one
  strike at a time; a COS expansion (Fang & Oosterlee) would be substantially
  faster for a whole smile.
- **Hedging.** Rebalancing happens only at the simulated dates, never in
  continuous time. There is no active gamma scalping (a trader adjusting
  rebalancing frequency or delta bands in response to realised P&L) — the
  hedges here follow a fixed schedule decided in advance. Hedging stops at a
  single underlying: a worst-of or basket payoff, and the cross-gammas that
  come with more than one asset, are out of scope. The volatility used to
  hedge is always static within a given run — no dynamic smile, no
  recalibration of the hedge ratio as the market's implied surface moves
  during the life of the trade.

## References

- Bouzoubaa & Osseiran, *Exotic Options and Hybrids*
- Grzelak & Oosterlee, *Mathematical Modeling and Computation in Finance*
- Albrecher et al., *The Little Heston Trap* (2007) — the characteristic
  function form used here, which avoids the complex-logarithm branch cut
- Lord, Koekkoek & van Dijk, *A comparison of biased simulation schemes for
  stochastic volatility models* (2010)
