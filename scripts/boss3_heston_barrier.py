"""DI put under Heston with a control variate, plus the model's implied smile.

Run:       .venv/bin/python scripts/boss3_heston_barrier.py
Validated: pytest tests/test_heston.py::test_boss3_artefacts

Output contract, read by the test down to the key names:

  figures/boss3_heston.png
      Two panels:
        (top)    the smile, implied volatility against log-moneyness log(K/F),
                 not against K, on which two maturities are not comparable;
        (bottom) the same barrier sweep as under Black-Scholes, transposed to
                 Heston: ratio of CV to MC half-widths against H/S0.

  figures/boss3_results.json
      {
        "params": {"S0":…, "K":…, "v0":…, "r":…, "T":…, "kappa":…, "theta":…,
                   "xi":…, "rho":…, "n_steps":…, "N":…},
        "smile": [{"K":…, "log_moneyness":…, "iv":…}, …],        >= 7 points
        "sweep": [{"H_pct":0.60, "H":60.0, "rho":…, "price_mc":…,
                   "half_width_mc":…, "price_cv":…, "half_width_cv":…,
                   "ratio_half_width":…, "c_hat":…}, …]           >= 6 points
      }
      sweep sorted by increasing H_pct, from 0.60 to 0.95. Python floats only.

Note the collision: the "rho" key inside "sweep" is the payoff-to-control
CORRELATION, whereas the "rho" inside "params" is the model's Brownian
correlation. Two different objects, one letter -- market notation.

What the run shows: the smile is no longer flat, and a barrier being a tail
event, its price is highly sensitive to the vol-of-vol and to the skew. The
variance reduction machinery, on the other hand, is reused unchanged; only the
source of EX differs, put_bs becoming heston_put. The gain curve has the same
shape as under Black-Scholes, which is the point -- variance reduction is a
statistical technique, independent of the model.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import barriers
import heston
from mc_engine import control_variate

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def main() -> None:
    """Produce figures/boss3_heston.png and figures/boss3_results.json.

    barriers.di_put and barriers.di_put_payoffs take PATHS rather than a model,
    so they work unchanged on Heston paths. barriers.di_put_cv does not: it
    calls put_bs directly, so the control is assembled here by hand from
    di_put_payoffs and mc_engine.control_variate, with EX taken from heston_put
    on the same parameters. A BS-computed EX on Heston paths would shift the
    price rather than stabilise it.

    heston_paths is much slower than gbm_paths, its time loop not being
    vectorisable, so the paths are simulated ONCE before the barrier loop --
    which also gives the sweep common random numbers.

    n_steps counts twice here: it discretises the barrier monitoring AND the
    Euler scheme. Too small a value biases the price twice, for two independent
    reasons.
    """
    S0=100
    v0=0.04
    r=0.05
    T=1
    kappa=1.5
    theta=0.04
    xi=0.3
    rho_cv=-0.7
    n_steps =50
    N = 100_000
    rng = np.random.default_rng(42)
    K = 100
    H = np.linspace(0.60,0.95,20)
    EX = heston.heston_put(S0,K,v0,r,T,kappa,theta,xi,rho_cv)
    S,v = heston.heston_paths(S0,v0,r,T,kappa,theta,xi,rho_cv,n_steps,N,rng)
    strikes = np.linspace(80, 120, 9)
    iv = heston.heston_smile(S0,strikes,v0,r,T,kappa,theta,xi,rho_cv)
    F=S0*np.exp(r*T)
    
    sweep = []
    smile =[]
    for k, ivs in zip(strikes,iv):
      smile.append({"K":float(k), "iv": float(ivs), "log_moneyness": float(np.log(k/F))})
    
    
    for H_pct in H:
      h = H_pct*S0
      Y, X = barriers.di_put_payoffs(S, K, h, r, T)
      price_mc, half_width_mc = barriers.di_put(S, K,h,r,T)
      
      (price_cv, half_width_cv, c_hat, rho_hat) = control_variate(Y, X, EX)
      ratio_half_width	= half_width_cv/half_width_mc
      sweep.append({ "H": h,"H_pct":H_pct, "rho": rho_hat, "half_width_cv": half_width_cv, "price_cv": price_cv,"ratio_half_width":ratio_half_width, "half_width_mc": half_width_mc, "c_hat": c_hat, "price_mc":price_mc})

    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "boss3_results.json").write_text(json.dumps({"params": {"S0": S0, "K": K,"r":r,"T":T,"v0":v0,"kappa": kappa,"theta": theta,"xi": xi,"rho": rho_cv,"n_steps":n_steps,"N":N}, "sweep": sweep, "smile": smile}, indent=2))  
    ratio_half_widths = [p["ratio_half_width"] for p in sweep]
   
    H_pcts =	[p["H_pct"] for p in sweep]
    rhos = 	[p["rho"] for p in sweep]
    curve_theoretical = np.sqrt(1 - np.array(rhos)**2)

    fig, (ax_h, ax_b) = plt.subplots(2, 1, figsize=(8, 7))
    ax_h.plot([p["log_moneyness"] for p in smile], [p["iv"] for p in smile],
              marker="o")
    ax_h.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax_h.set_ylabel("vol implicite BS")
    ax_h.set_xlabel("log-moneyness  log(K/F)")
    ax_h.set_title(f"Le modèle produit un skew (ρ={rho_cv}) — pente ≈ "
                   f"{np.polyfit([p['log_moneyness'] for p in smile], [p['iv'] for p in smile], 1)[0]:.3f}",
                   fontsize=10)

    ax_b.plot(H_pcts, ratio_half_widths, marker="o", label="ratio mesuré")
    ax_b.plot(H_pcts, curve_theoretical, linestyle="--", color="black",
              label="√(1−ρ̂²) — identité algébrique, cohérence interne")
    ax_b.axhline(1.0, color="grey", linestyle=":",
                 label="seuil : au-dessus, le contrôle dégraderait")
    ax_b.set_ylabel("demi-IC CV / demi-IC MC")
    ax_b.set_xlabel("H / S0")
    ax_b.legend(fontsize=9)
    ax_b.set_title("…et la réduction de variance fonctionne quand même",
                   fontsize=10)

    ax_h.grid(True, alpha=0.3)
    ax_b.grid(True, alpha=0.3)
    fig.suptitle(
        "DI put sous Heston — smile du modèle et gain du contrôle\n"
        f"S0={S0}, K={K}, r={r}, T={T}  ·  "
        f"v0={v0}, κ={kappa}, θ={theta}, ξ={xi}, ρ={rho_cv}\n"
        f"n_steps={n_steps}, N={N:,}".replace("N=100,000", "N=100 000"),
        fontsize=11)
    fig.tight_layout()

    fig.savefig(FIGURES / "boss3_heston.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
