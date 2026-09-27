"""Convergence of the hedging error, and distribution of the replication P&L.

Run:       .venv/bin/python scripts/boss4_hedging.py
Validated: pytest tests/test_hedging.py::test_boss4_artefacts

Output contract, read by the test down to the key names:

  figures/boss4_hedging.png
      Two panels:
        (top)    log-log scatter of std(hedging_error) against n_rebal, on
                 4.2's grid, with the theoretical -1/2 slope line drawn on
                 top (anchored at the grid's first point);
        (bottom) three overlaid histograms (or three side-by-side
                 distributions) of the replication P&L, at fixed n_rebal:
                 correct model (4.2), wrong model -- vol arbitrage (4.3),
                 with transaction costs (4.4). Same x-axis for all three, so
                 the relative squeeze between distributions stays readable.

  figures/boss4_results.json
      {
        "params": {"S0": ..., "K": ..., "sigma": ..., "r": ..., "T": ...,
                   "sigma_impl": ..., "sigma_real": ..., "cost_rate": ...,
                   "n_steps_pnl": ..., "N": ...},
        "convergence": [
          {"n_steps": 8, "std_error": ...}, ...
        ]                                              >= 5 points, increasing
        "pnl": {
          "modele_correct":  {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...},
          "modele_faux":     {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...},
          "avec_couts":      {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...}
        },
        "slope_measured": ...             # log-log slope measured on "convergence"
      }
      Python floats only, json.dump rejects np.float64.

What the figure must show: the top panel is the same measurement as
`test_hedging_error_moyenne_dans_ic_et_pente_log_log` (4.2), simply plotted
instead of merely tested -- the -1/2 slope must hug the points. The bottom
panel sets three distinct failure modes of the same hedge side by side: pure
discretisation noise (correct model, centred at zero), model bias (vol
arbitrage, shifted and off-centre if sigma_real != sigma_impl), and the
certain cost (with costs, shifted negative, never centred at zero).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hedging
from bs import put_bs
from mc_engine import gbm_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def main() -> None:
    """Produce figures/boss4_hedging.png and figures/boss4_results.json.

    QUEST BOSS 4 -- convergence and replication P&L [80 XP]
    GOAL: assemble into a single figure the three interview-defensible
        results announced at the top of the act -- the `n_rebal^{-1/2}`
        convergence (4.2), the vol-arbitrage identity (4.3), and the effect
        of transaction costs (4.4). Nothing new to compute: this boss only
        calls what already exists in `hedging.py` and turns it into a
        picture, exactly like `boss2_barrier_sweep.py` and
        `boss3_heston_barrier.py` before it.
    UNLOCKS: nothing -- closes Act IV.
    VALIDATION: `test_boss4_artefacts` -- checks that BOTH files exist, that
        `convergence` has at least 5 increasing points in `n_steps`, that
        `slope_measured` sits in the same range as 4.2's test
        ([-0.8, -0.2]), and that all three entries of `pnl` carry their 5
        keys. No new numerical identity here -- everything was already
        validated quest by quest, this script only replays them and plots
        them together.
    HINT 1 (intuition): this script tests nothing new, it TELLS the story of
        the act -- reuse the grids and formulas already validated in
        `tests/test_hedging.py` (`n_steps_grid`, `sigma_impl`/`sigma_real`,
        `cost_rate`) as they are, no need to invent new parameters.
    HINT 2 (structure):
        - Top panel: loop over a grid of `n_steps` (as in
          `test_hedging_error_moyenne_dans_ic_et_pente_log_log`), one
          `gbm_paths` + `bs_delta_hedge_deltas` + `hedging_error` per point,
          collect `errors.std(ddof=1)`. `np.polyfit` on `log(n_steps)` vs
          `log(std)` gives `slope_measured`; the theoretical line is drawn
          with `std[0] * (n_steps_grid / n_steps_grid[0])**(-0.5)`.
        - Bottom panel: A SINGLE set of GBM paths (at `sigma_real`, for the
          wrong model), `n_steps` fixed once and for all (`n_steps_pnl`):
          `hedging_error` (correct model, on paths at `sigma_impl` -- hence
          a SECOND set of paths, at `sigma_impl` this time),
          `vol_arbitrage_pnl` (wrong model, on the paths at `sigma_real`),
          and `hedging_error_with_costs` (correct model + costs, on the
          paths at `sigma_impl`, `cost_rate` > 0).
    HINT 3 (formula): nothing new -- every call already exists
        (`bs_delta_hedge_deltas`, `hedging_error`, `hedging_error_stats`,
        `vol_arbitrage_pnl`, `hedging_error_with_costs`). The JSON's
        percentiles come from `np.percentile(array, [5, 50, 95])`.
    PITFALL: the THREE distributions in the bottom panel do not live under
        the same set of paths nor the same budget -- "correct model" is
        centred at zero by construction (4.2), "wrong model" is centred
        ONLY IF `sigma_real != sigma_impl` (otherwise it collapses back onto
        "correct model"), "with costs" is NEVER centred at zero (a certain
        loss, cf. the PITFALL of `hedging_error_with_costs` in 4.4) -- do
        not be surprised that the three histograms share neither the same
        mean nor the same spread, that is the point of the figure.
    """
    S0 = 100.0
    K = 100.0
    sigma_impl = 0.2
    sigma_real = 0.3
    r = 0.05
    T = 1.0
    cost_rate = 0.001
    n_steps_pnl = 100
    N = 10000

    S_impl = gbm_paths(S0, sigma_impl, r, T, n_steps_pnl, N)
    S_real = gbm_paths(S0, sigma_real, r, T, n_steps_pnl, N)
    
    deltas = hedging.bs_delta_hedge_deltas( paths=S_impl,K=K,sigma=sigma_impl,r=r,T=T,option="put")
    convergence = []
    for n_steps in [8, 16, 32, 64, 128, 256, 512]:
        S_impl_n = gbm_paths(S0, sigma_impl, r, T, n_steps, N)
        deltas_n = hedging.bs_delta_hedge_deltas(paths=S_impl_n,K=K,sigma=sigma_impl,r=r,T=T,option="put")
        errors = hedging.hedging_error(paths=S_impl_n,deltas=hedging.bs_delta_hedge_deltas(paths=S_impl_n,K=K,sigma=sigma_impl,r=r,T=T,option="put"),K=K,r=r,T=T,V0=put_bs(S0,K,sigma_impl,r,T),option="put")
        std_error = errors.std(ddof=1)
        convergence.append({"n_steps": n_steps, "std_error": std_error})



    
    fig, (ax_haut, ax) = plt.subplots(2, 1, figsize=(6, 8))
    n_steps_grid = np.array([c["n_steps"] for c in convergence])
    stds = np.array([c["std_error"] for c in convergence])
    droite = stds[0] * (n_steps_grid / n_steps_grid[0])**(-0.5)
    slope_measured, _ = np.polyfit(np.log(n_steps_grid), np.log(stds), 1)

    ax_haut.loglog(n_steps_grid, stds, "o", label="measured")
    ax_haut.loglog(n_steps_grid, droite, "--", label="-1/2 slope")
    ax_haut.set_xlabel("n_rebal")
    ax_haut.set_ylabel("std(hedging_error)")
    ax_haut.legend()

    ax.hist(hedging.portfolio_terminal_value(S_impl, deltas, r, T, V0=put_bs(S0,K,sigma_impl,r,T)) - np.maximum(K - S_impl[:, -1], 0),
            bins=50, density=True, alpha=0.5, label="correct model")

    ax.hist(hedging.vol_arbitrage_pnl(paths=S_real, K=K, sigma_impl=sigma_impl, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), option="put")
            ,bins=50, density=True, alpha=0.5, label="wrong model")
    ax.hist(hedging.hedging_error_with_costs(paths=S_impl,deltas=deltas, K = K, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), cost_rate = cost_rate, option="put"),
            bins=50, density=True, alpha=0.5, label="with costs")
    ax.set_xlabel("replication P&L")
    ax.set_ylabel("density")
    ax.legend()
    
  
    plt.savefig(FIGURES / "boss4_hedging.png", dpi=150, bbox_inches="tight")

    arr_correct = hedging.hedging_error(paths=S_impl,deltas=deltas,K=K,r=r,T=T,V0=put_bs(S0,K,sigma_impl,r,T),option="put")
    mean = arr_correct.mean()
    std = arr_correct.std(ddof=1)
    pnl_correct = {
        "modele_correct": {
            "mean": mean,
            "std": std,
            "p5": np.percentile(arr_correct, 5),
            "p50": np.percentile(arr_correct, 50),
            "p95": np.percentile(arr_correct, 95)
        }
      }
    arr_faux = hedging.vol_arbitrage_pnl(paths=S_real, K=K, sigma_impl=sigma_impl, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), option="put")
    mean = arr_faux.mean()
    std = arr_faux.std(ddof=1)
    pnl_faux = {
        "modele_faux": {
            "mean": mean,
            "std": std,
            "p5": np.percentile(arr_faux, 5),
            "p50": np.percentile(arr_faux, 50),
            "p95": np.percentile(arr_faux, 95)
        }
      }
    arr_couts = hedging.hedging_error_with_costs(paths=S_impl,deltas=deltas, K = K, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), cost_rate = cost_rate, option="put")
    mean = arr_couts.mean()
    std = arr_couts.std(ddof=1)
    pnl_couts = {
        "avec_couts": {
            "mean": mean,
            "std": std,
            "p5": np.percentile(arr_couts, 5),
            "p50": np.percentile(arr_couts, 50),
            "p95": np.percentile(arr_couts, 95)
        }
      }
    pnl = {**pnl_correct, **pnl_faux, **pnl_couts}
    results = {
        "params": {
            "S0": S0, "K": K, "sigma_impl": sigma_impl, "sigma_real": sigma_real,
            "r": r, "T": T, "cost_rate": cost_rate, "n_steps_pnl": n_steps_pnl,
            "N": N
        },
        "convergence": convergence,
        "pnl": pnl,
        "slope_measured": slope_measured
    }

    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "boss4_results.json").write_text(json.dumps(results))
    plt.show()

if __name__ == "__main__":
    main()
