"""Act IV -- discrete hedging: self-financed portfolio, replication error.

This first slice covers QUEST 4.1 and QUEST 4.2 only. Tests for 4.3 through
4.6 and BOSS 4 arrive once 4.2's output has been posted and validated -- this
is a rule from the repo's CLAUDE.md, not an oversight.

4.1 is an algebraic identity (arbitrary deltas, 1e-12): it depends on no
pricing at all. 4.2 is a log-log convergence-RATE test, never a bare
numerical threshold -- see test_hedging_error_moyenne_dans_ic_et_pente_log_log.

Run: .venv/bin/python -m pytest tests/test_hedging.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np
import pytest
from scipy.stats import norm

import barriers
import bs
import hedging
from bs import call_bs, put_bs, vega
from heston import heston_paths
from mc_engine import gbm_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Reference parameters, consistent with the earlier acts.
HP = dict(S0=100.0, K=100.0, sigma=0.20, r=0.05, T=1.0)

# Reference Heston parameters, identical to those in tests/test_heston.py.
HESTON_HP = dict(S0=100.0, v0=0.04, r=0.05, T=1.0,
                  kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


def _half_width(v: np.ndarray) -> float:
    """95% CI half-width -- the repo's convention."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))



def test_autofinancement_deltas_arbitraires():
    """The self-financing identity holds for ANY sequence of deltas.

    Deltas drawn at random in [-5, 5] (negative, huge, nothing like a real
    option delta): this test does not validate a pricing, it validates a
    bookkeeping rule. The reference recursion is rebuilt HERE, in the test,
    independently of `hedging.py` -- comparing the function to itself would
    prove nothing.
    """
    rng = np.random.default_rng(0)
    N, n_steps = 500, 12
    r, T = 0.03, 1.0
    dt = T / n_steps

    paths = gbm_paths(S0=100.0, sigma=0.25, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = rng.uniform(-5.0, 5.0, size=(N, n_steps))
    V0 = rng.uniform(1.0, 20.0, size=N)

    # Reference recursion, written independently of hedging.py.
    B = V0 - deltas[:, 0] * paths[:, 0]
    for i in range(n_steps - 1):
        B = B * np.exp(r * dt) - (deltas[:, i + 1] - deltas[:, i]) * paths[:, i + 1]
    V_T_ref = deltas[:, -1] * paths[:, -1] + B * np.exp(r * dt)

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)

    assert V_T.shape == (N,)
    assert np.max(np.abs(V_T - V_T_ref)) < 1e-12


def test_autofinancement_delta_nul_est_du_cash_pur():
    """deltas = 0 everywhere: the portfolio never touches the share.

    V_T must then be exactly V0 compounded at the risk-free rate -- no path
    should play any role in the result.
    """
    rng = np.random.default_rng(1)
    N, n_steps = 1_000, 20
    r, T = 0.04, 1.0

    paths = gbm_paths(S0=100.0, sigma=0.3, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = np.zeros((N, n_steps))
    V0 = 7.5

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)

    assert np.max(np.abs(V_T - V0 * np.exp(r * T))) < 1e-10


def test_autofinancement_delta_constant_formule_fermee():
    """deltas = c everywhere: closed identity, independent of the step loop.

    With no rebalancing, the cash placed at t=0 compounds over the whole
    period, and the result has a closed form:
        V_T = V0*exp(rT) + c*(S_T - S0*exp(rT))
    This is a SECOND reference, independent of the step-by-step recursion
    tested above -- an error that happened to cancel out in the recursion
    should still show up here.
    """
    rng = np.random.default_rng(2)
    N, n_steps = 1_000, 15
    r, T, S0 = 0.05, 1.0, 100.0
    c = 0.4

    paths = gbm_paths(S0=S0, sigma=0.2, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = np.full((N, n_steps), c)
    V0 = 10.0

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)
    V_T_ref = V0 * np.exp(r * T) + c * (paths[:, -1] - S0 * np.exp(r * T))

    assert np.max(np.abs(V_T - V_T_ref)) < 1e-9


def test_bs_delta_hedge_deltas_vs_formule_fermee():
    """At the first step (tau = T), delta must match N(d1) - 1 (put).

    Reference written directly here with scipy.stats.norm, independent of
    bs.py. Tolerance 1e-3: not an algebraic identity
    (bs_delta_hedge_deltas may go through an internal finite difference),
    but an order of magnitude that does not forgive a wrong formula.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N, n_steps = 2_000, 10

    paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(10))
    deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")

    assert deltas.shape == (N, n_steps)

    S_t0 = paths[:, 0]
    d1 = (np.log(S_t0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta_ref = norm.cdf(d1) - 1.0

    assert np.max(np.abs(deltas[:, 0] - delta_ref)) < 1e-3


def test_hedging_error_moyenne_dans_ic_et_pente_log_log():
    """4.2, the validation that closes the quest.

    (a) On a given run, the mean hedging error must sit inside its own 95%
        CI around zero: a correctly built BS hedge, under the model it
        assumes, does not bias the replicating portfolio.
    (b) On a grid of n_steps, the error's standard deviation decreases, and
        the slope of the log(std) ~ log(n_steps) regression must be close
        to -1/2 (Boyle & Emanuel, 1980 -- given as HINT 3 in hedging.py, not
        here). Loose tolerance on the slope ([-0.8, -0.2]) since this is a
        statistical fit on few points, not an identity -- what is tested is
        the DIRECTION and the ORDER OF MAGNITUDE of the convergence, never a
        numerical threshold on a single standard-deviation value.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N = 30_000
    n_steps_grid = [8, 16, 32, 64, 128, 256]

    V0 = put_bs(S0, K, sigma, r, T)
    stds = []

    for n_steps in n_steps_grid:
        paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                           N=N, rng=np.random.default_rng(1000 + n_steps))
        deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")
        errors = hedging.hedging_error(paths, deltas, K, r, T, V0, option="put")

        mean, half_width = hedging.hedging_error_stats(errors)
        assert abs(mean) < half_width * 1.5, (
            f"n_steps={n_steps}: mean {mean:.4f} outside CI (+/-{half_width:.4f})"
        )

        stds.append(errors.std(ddof=1))

    stds = np.asarray(stds)
    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(stds), 1)
    print(f"measured log-log slope = {slope:.3f} (expected ~ -0.5)")

    assert np.all(np.diff(stds) < 0.0)          # the std decreases
    assert -0.8 < slope < -0.2


def test_hedging_error_stats_convention():
    """hedging_error_stats follows the repo's convention: (mean, 95% CI half-width).

    A pure identity on an arbitrary sample, unrelated to hedging -- it only
    locks down the formula 1.96*sd/sqrt(n) on ddof=1, as everywhere else in
    the repo.
    """
    rng = np.random.default_rng(3)
    errors = rng.normal(loc=0.7, scale=2.5, size=5_000)

    mean, half_width = hedging.hedging_error_stats(errors)

    assert abs(mean - errors.mean()) < 1e-12
    assert abs(half_width - _half_width(errors)) < 1e-12


# ---------------------------------------------------------------------------
# QUEST 4.3 -- vol arbitrage / the wrong model
# ---------------------------------------------------------------------------


def test_bs_gamma_vs_difference_finie_sur_delta():
    """Gamma is the derivative of delta with respect to spot.

    Reference independent of the closed form: a centred finite difference
    on bs.delta (itself already a finite difference on the price).
    Tolerance 1e-3 -- two layers of finite differences stacked, the error
    cannot be 1e-10.
    """
    K, sigma, r = 100.0, 0.25, 0.03
    h_outer = 0.01

    for S0, tau in [(90.0, 0.5), (100.0, 1.0), (115.0, 0.25)]:
        gamma_closed = hedging.bs_gamma(S0, K, sigma, r, tau)

        delta_up = bs.delta(S0 + h_outer, 1e-4, K, sigma, r, tau)
        delta_down = bs.delta(S0 - h_outer, 1e-4, K, sigma, r, tau)
        gamma_fd = (delta_up - delta_down) / (2 * h_outer)

        print(f"S0={S0} tau={tau}  gamma_closed={gamma_closed:.6f}  gamma_fd={gamma_fd:.6f}")
        assert abs(gamma_closed - gamma_fd) < 1e-3


def test_vol_arbitrage_identite_gamma():
    """4.3's identity: mean discounted P&L = -weighted gamma integral.

    Reference rebuilt HERE, independently of vol_arbitrage_pnl: the discrete
    sum of -exp(-r*t_i)*0.5*Gamma_i*S_i^2*(sigma_real^2-sigma_impl^2)*dt
    along each path, averaged over the N paths. MINUS sign: V0 puts the
    portfolio on the SELLER's side (short gamma), which loses when realised
    vol exceeds hedge vol. The exp(-r*t_i) factor INSIDE the sum (not just
    exp(-rT) as a global factor) comes from solving the ODE
    de_t = r*e_t*dt - 0.5*Gamma*S^2*(sigma_real^2-sigma_impl^2)*dt on the
    hedging error e_t = Pi_t - V_t: the accumulated error compounds itself at
    rate r up to maturity, it is not simply discounted once at the end (see
    HINT 3 of vol_arbitrage_pnl in hedging.py -- and the PITFALL documenting
    this very mistake, hit once here). This is an IDENTITY tested on the
    MEAN (compared to its 95% CI), not a sign inequality -- a correct sign
    can come out of a wrong formula by chance, a numerical identity much
    less so.
    """
    S0, K, r, T = 100.0, 100.0, 0.05, 1.0
    sigma_impl, sigma_real = 0.20, 0.30
    N, n_steps = 20_000, 100
    dt = T / n_steps

    paths = gbm_paths(S0=S0, sigma=sigma_real, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(42))
    V0 = put_bs(S0, K, sigma_impl, r, T)

    pnl = hedging.vol_arbitrage_pnl(paths, K, sigma_impl, r, T, V0, option="put")
    mean_pnl, half_width_pnl = hedging.hedging_error_stats(pnl)

    t_i = np.arange(n_steps) * dt                 # start of each interval
    tau = T - t_i                                  # (n_steps,) -- one per rebalanced column
    S_rebal = paths[:, :-1]                        # (N, n_steps)
    gamma = hedging.bs_gamma(S_rebal, K, sigma_impl, r, tau)
    integrand = -np.exp(-r * t_i) * 0.5 * gamma * S_rebal**2 * (sigma_real**2 - sigma_impl**2) * dt
    ref_per_path = integrand.sum(axis=1)
    mean_ref = ref_per_path.mean()

    print(f"measured mean P&L={mean_pnl:.4f}+/-{half_width_pnl:.4f}  "
          f"gamma reference={mean_ref:.4f}")

    assert abs(mean_pnl - mean_ref) < half_width_pnl * 1.5


# ---------------------------------------------------------------------------
# QUEST 4.4 -- transaction costs
# ---------------------------------------------------------------------------


def test_turnover_deltas_constants_et_alternes():
    """Two closed identities on turnover, at both extremes.

    Constant deltas: a single purchase, never rebalanced afterwards --
    turnover = |delta_0| exactly. Deltas alternating sign at every step:
    nothing ever cancels, turnover = sum of all |gaps|, computable by hand
    term by term.
    """
    N, n_steps = 200, 10

    deltas_const = np.full((N, n_steps), 3.5)
    assert np.max(np.abs(hedging.turnover(deltas_const) - 3.5)) < 1e-12

    # alternates +2, -2, +2, -2, ... : |2-0| + |-2-2| + |2-(-2)| + ... = 2 + 4*(n_steps-1)
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n_steps)])
    deltas_alt = 2.0 * np.tile(signs, (N, 1))
    expected = 2.0 + 4.0 * (n_steps - 1)
    assert np.max(np.abs(hedging.turnover(deltas_alt) - expected)) < 1e-12


def test_transaction_costs_deltas_constants():
    """Constant deltas: a single trade, at the very first price S_0.

    Closed identity independent of `turnover`: cost = cost_rate * |delta| * S0,
    since no rebalancing happens after the initial purchase.
    """
    rng = np.random.default_rng(5)
    N, n_steps, S0 = 500, 8, 100.0
    paths = gbm_paths(S0=S0, sigma=0.2, r=0.05, T=1.0, n_steps=n_steps,
                       N=N, rng=rng)
    c = 1.7
    deltas = np.full((N, n_steps), c)
    cost_rate = 0.002

    costs = hedging.transaction_costs(paths, deltas, cost_rate)
    expected = cost_rate * abs(c) * S0

    assert np.max(np.abs(costs - expected)) < 1e-9


def test_transaction_costs_frequence_optimale():
    """4.4, the validation that closes the quest.

    (a) the mean transaction cost GROWS as sqrt(n_rebal) -- positive log-log
        slope, close to +1/2 (loose tolerance, same spirit as 4.2);
    (b) the hedging error's standard deviation WITHOUT costs keeps
        decreasing (a reminder of 4.2, same paths);
    (c) there is an n_steps that minimises the RMS of the error WITH costs
        -- neither the smallest nor the largest of the tested grid. A test
        that proves the existence of an intermediate optimum, not a
        convergence.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N = 20_000
    n_steps_grid = [4, 8, 16, 32, 64, 128, 256, 512]
    cost_rate = 0.005

    V0 = put_bs(S0, K, sigma, r, T)
    mean_costs, rms_with_costs = [], []

    for n_steps in n_steps_grid:
        paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                           N=N, rng=np.random.default_rng(2000 + n_steps))
        deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")

        costs = hedging.transaction_costs(paths, deltas, cost_rate)
        mean_costs.append(costs.mean())

        errors_with_costs = hedging.hedging_error_with_costs(
            paths, deltas, K, r, T, V0, cost_rate, option="put")
        rms_with_costs.append(np.sqrt(np.mean(errors_with_costs**2)))

    mean_costs = np.asarray(mean_costs)
    rms_with_costs = np.asarray(rms_with_costs)

    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(mean_costs), 1)
    print(f"mean-cost log-log slope = {slope:.3f} (expected ~ +0.5)")
    print(f"RMS error with costs = {np.round(rms_with_costs, 4)}")

    assert np.all(np.diff(mean_costs) > 0.0)      # cost grows
    assert 0.2 < slope < 0.8

    i_min = int(np.argmin(rms_with_costs))
    assert 0 < i_min < len(n_steps_grid) - 1       # intermediate optimum, not at the edges


# ---------------------------------------------------------------------------
# QUEST 4.5 -- delta near a barrier
# ---------------------------------------------------------------------------


def test_naive_barrier_hedge_degrade_pres_de_la_barriere():
    """4.5, the validation that closes the quest: no convergence, a DEGRADATION.

    Two barriers, same paths (CRN): one far from spot (H=60, knock-out is
    rare, the vanilla delta hedges close to a vanilla), one close to spot
    (H=90, paths cross the barrier often, the vanilla delta never sees the
    payoff's discontinuity).

    (a) the error's standard deviation near the barrier must be clearly
        larger than far from it (ratio > 2, measured empirically ~2.8);
    (b) near the barrier, increasing n_steps does NOT decrease the standard
        deviation as in 4.2 -- the log-log slope must stay close to 0
        (measured ~0.0, very different from 4.2's -0.5), a sign that the
        error is a structural bias of the vanilla delta, not MC noise that
        averages out.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    H_far, H_near = 60.0, 90.0
    N, n_steps = 20_000, 50

    paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(21))

    V0_far = barriers.do_put(paths, K, H_far, r, T)[0]
    V0_near = barriers.do_put(paths, K, H_near, r, T)[0]

    err_far = hedging.naive_barrier_hedge_error(
        paths, K, H_far, sigma, r, T, V0_far, barrier="do", option="put")
    err_near = hedging.naive_barrier_hedge_error(
        paths, K, H_near, sigma, r, T, V0_near, barrier="do", option="put")

    std_far, std_near = err_far.std(ddof=1), err_near.std(ddof=1)
    print(f"std(H={H_far})={std_far:.4f}  std(H={H_near})={std_near:.4f}  "
          f"ratio={std_near / std_far:.3f}")
    assert std_near / std_far > 2.0

    stds_near = []
    n_steps_grid = [16, 32, 64, 128, 256]
    for n in n_steps_grid:
        p = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n,
                       N=N, rng=np.random.default_rng(3000 + n))
        v0_n = barriers.do_put(p, K, H_near, r, T)[0]
        e = hedging.naive_barrier_hedge_error(
            p, K, H_near, sigma, r, T, v0_n, barrier="do", option="put")
        stds_near.append(e.std(ddof=1))

    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(stds_near), 1)
    print(f"log-log slope near the barrier = {slope:.3f} (expected ~ 0, NOT -0.5)")
    assert -0.15 < slope < 0.15


# ---------------------------------------------------------------------------
# QUEST 4.6 -- delta-vega under Heston
# ---------------------------------------------------------------------------


def test_heston_delta_bs_hedge_error_biaise():
    """BS-delta-only hedge under Heston: nonzero error, with notable variance.

    No closed-form identity here (that is the whole point of 4.6) -- just
    the guarantee that the function runs, returns the right shape, and a
    clearly nonzero variance (otherwise the variance-reduction test that
    follows would be meaningless: there would be nothing to reduce).
    """
    S0, K, r, T = HESTON_HP["S0"], HP["K"], HESTON_HP["r"], HESTON_HP["T"]
    sigma_hedge = np.sqrt(HESTON_HP["theta"])
    N, n_steps = 20_000, 50

    S, _ = heston_paths(**HESTON_HP, n_steps=n_steps, N=N,
                         rng=np.random.default_rng(30))
    V0 = put_bs(S0, K, sigma_hedge, r, T)

    err = hedging.heston_delta_bs_hedge_error(S, K, sigma_hedge, r, T, V0, option="put")

    assert err.shape == (N,)
    assert err.std(ddof=1) > 0.5


def test_heston_delta_vega_reduit_la_variance():
    """4.6, the validation that closes the quest and the act.

    Static vega overlay on K_vega=110, sized by the ratio of the BS vegas at
    sigma_hedge. On the SAME Heston paths (CRN), the variance ratio (delta
    alone / delta+vega) must be significantly > 1 -- measured empirically
    ~3.0, threshold set at 1.5 to leave margin for MC noise.
    """
    S0, K, K_vega, r, T = HESTON_HP["S0"], HP["K"], 110.0, HESTON_HP["r"], HESTON_HP["T"]
    sigma_hedge = np.sqrt(HESTON_HP["theta"])
    N, n_steps = 30_000, 50

    S, _ = heston_paths(**HESTON_HP, n_steps=n_steps, N=N,
                         rng=np.random.default_rng(31))
    V0 = put_bs(S0, K, sigma_hedge, r, T)

    err_delta = hedging.heston_delta_bs_hedge_error(S, K, sigma_hedge, r, T, V0, option="put")
    err_delta_vega = hedging.heston_delta_vega_hedge_error(
        S, K, K_vega, sigma_hedge, r, T, V0, option="put")

    ratio = err_delta.var(ddof=1) / err_delta_vega.var(ddof=1)
    print(f"Var(delta alone)={err_delta.var(ddof=1):.4f}  "
          f"Var(delta+vega)={err_delta_vega.var(ddof=1):.4f}  ratio={ratio:.3f}")

    assert ratio > 1.5


# ---------------------------------------------------------------------------
# BOSS 4 -- convergence and replication P&L
# ---------------------------------------------------------------------------


def test_boss4_artefacts():
    """Checks the artefacts produced by scripts/boss4_hedging.py.

    Run: .venv/bin/python scripts/boss4_hedging.py

    No new numerical identity -- everything was already validated quest by
    quest in this file. This test only checks that the script runs, that
    the output contract (documented at the top of boss4_hedging.py) is
    respected, and that the slope measured on "convergence" falls in the
    same range as 4.2's.
    """
    fig = ROOT / "figures" / "boss4_hedging.png"
    res = ROOT / "figures" / "boss4_results.json"
    assert fig.exists(), "missing figure"
    assert res.exists(), "missing results"

    data = json.loads(res.read_text())

    convergence = data["convergence"]
    assert len(convergence) >= 5
    n_steps_list = [p["n_steps"] for p in convergence]
    assert n_steps_list == sorted(n_steps_list)
    stds = [p["std_error"] for p in convergence]
    assert all(a > b for a, b in zip(stds, stds[1:]))       # decreasing

    assert -0.8 < data["slope_measured"] < -0.2

    pnl = data["pnl"]
    assert set(pnl.keys()) == {"modele_correct", "modele_faux", "avec_couts"}
    for scenario in pnl.values():
        assert set(scenario.keys()) == {"mean", "std", "p5", "p50", "p95"}
        assert scenario["p5"] < scenario["p50"] < scenario["p95"]

    # "avec_couts" is a certain loss: never centred at zero.
    assert pnl["avec_couts"]["mean"] < 0.0
