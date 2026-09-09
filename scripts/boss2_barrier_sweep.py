"""Barrier sweep of the DI put under Black-Scholes, from 60% to 95% of spot.

Measures the gain of the control variate as a function of the barrier level.

Run:       .venv/bin/python scripts/boss2_barrier_sweep.py
Validated: pytest tests/test_variance_reduction.py::test_boss2_artefacts

Output contract, read by the test down to the key names:

  figures/boss2_barrier_sweep.png
      Two panels sharing the H/S0 axis:
        (top)    rho(H), the correlation between the DI put payoff and the
                 vanilla control payoff;
        (bottom) ratio of CI half-widths, CV over raw MC, with the horizontal
                 line y=1 above which the control would degrade the estimator.

  figures/boss2_results.json
      {
        "params": {"S0": ..., "K": ..., "sigma": ..., "r": ..., "T": ...,
                   "n_steps": ..., "N": ...},
        "sweep": [
          {"H_pct": 0.60, "H": 60.0, "rho": ..., "price_mc": ...,
           "half_width_mc": ..., "price_cv": ..., "half_width_cv": ...,
           "ratio_half_width": ..., "c_hat": ...},
          ...
          {"H_pct": 0.95, ...}
        ]
      }
      Sorted by increasing H_pct, first point at 0.60, last at 0.95, at least 6
      points. Python floats only, json.dump rejecting np.float64.

What the sweep shows: as H rises towards the spot the DI put resembles the
vanilla put more and more, rho tends to 1 and the ratio collapses. As H falls
the knock-in becomes rare and the control explains almost none of the DI put's
variance -- yet the ratio never exceeds 1, since an optimal c falls back to 0
and the estimator degrades to the raw Monte-Carlo.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib
matplotlib.use("Agg")            # backend fichier : aucun besoin d'affichage
import matplotlib.pyplot as plt
import numpy as np

import barriers
from mc_engine import gbm_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def main() -> None:
    """Produce figures/boss2_barrier_sweep.png and figures/boss2_results.json.

    All barrier levels are priced on the SAME set of paths (common random
    numbers). Simulating afresh for each H would leave the rho(H) curve
    trembling with Monte-Carlo noise, and a dip in it could no longer be told
    apart from an artefact.

    The dashed sqrt(1 - rho_hat^2) curve drawn over the measured ratio is not an
    independent validation: with c estimated in-sample, Var(Z) = Var(Y)(1 -
    rho_hat^2) identically, so the two coincide by algebra. What it does check is
    internal consistency -- matching ddof between cov, var and std, the
    half-width taken on the residual rather than on Y, the same n on both sides.
    An external reference would require the Reiner-Rubinstein closed form, which
    assumes continuous monitoring whereas the barrier is monitored over n_steps
    dates here.

    c is estimated in-sample rather than on a pilot run, which leaves an O(1/N)
    bias, negligible at N = 100 000. See mc_engine.pilot_c.
    """
    S0=100
    K=100
    sigma=0.2
    r=0.05
    T=1
    n_steps=50
    N=100_000
    rng = np.random.default_rng(42)
    H_list = np.linspace(0.60, 0.95, 8)
    paths = gbm_paths(S0,sigma,r,T,n_steps,N,rng = rng)
    sweep = []
    for H_pct in H_list:
      H = H_pct*S0
      price_mc,half_width_mc = barriers.di_put(paths,K,H,r,T)
      (price_cv, half_width_cv, c_hat, rho) = barriers.di_put_cv(paths,K,H,r,T,sigma,c=None)
      ratio_half_width = half_width_cv/half_width_mc
      sweep.append({"H_pct" : float(H_pct),
                   "H" : float(H),
                   "price_mc": float(price_mc),
                   "half_width_mc":float(half_width_mc),
                   "price_cv" : float(price_cv),
                   "half_width_cv" : float(half_width_cv),
                   "ratio_half_width":ratio_half_width,
                   "c_hat" : float(c_hat),
                   "rho": float(rho)})
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "boss2_results.json").write_text(json.dumps({"params": {"S0": S0, "K": K, "sigma": sigma,"r":r,"T":T,"n_steps":n_steps,"N":N}, "sweep": sweep}, indent=2))  
    H_pcts = [p["H_pct"] for p in sweep]
    rhos = [p["rho"] for p in sweep ]
    ratio_half_widths = [p["ratio_half_width"] for p in sweep]
    curve_theoretical = np.sqrt(1 - np.array(rhos)**2)
    
    fig, (ax_h, ax_b) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    ax_h.plot(H_pcts, rhos, marker="o")
    ax_h.set_ylabel("ρ (DI put, put vanille)")
    ax_h.set_title("Le contrôle explique d'autant mieux que la barrière est haute",
                   fontsize=10)

    ax_b.plot(H_pcts, ratio_half_widths, marker="o", label="ratio mesuré")
    ax_b.plot(H_pcts, curve_theoretical, linestyle="--", color="black",
              label="√(1−ρ̂²) — identité algébrique, cohérence interne")
    ax_b.axhline(1.0, color="grey", linestyle=":",
                 label="seuil : au-dessus, le contrôle dégraderait")
    ax_b.set_ylabel("demi-IC CV / demi-IC MC")
    ax_b.set_xlabel("H / S0")
    ax_b.legend(fontsize=9)

    ax_h.grid(True, alpha=0.3)
    ax_b.grid(True, alpha=0.3)
    fig.suptitle(
        "DI put — gain de la variable de contrôle (contrôle = put vanille)\n"
        f"S0={S0}, K={K}, σ={sigma}, r={r}, T={T}\n"
        f"n_steps={n_steps}, N={N:,}".replace("N=100,000", "N=100 000"),
        fontsize=11)
    fig.tight_layout()

    fig.savefig(FIGURES / "boss2_barrier_sweep.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()