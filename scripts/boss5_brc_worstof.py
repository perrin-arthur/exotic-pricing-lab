"""Pricing complet d'un BRC worst-of + figure -- boss de fin d'Acte V.

Run:       .venv/bin/python scripts/boss5_brc_worstof.py
Validated: pytest tests/test_multi_asset.py::test_boss5_artefacts

Output contract, lu par le test jusqu'aux noms de cles :

  figures/boss5_brc_worstof.png
      Deux panneaux :
        (haut)  prix du BRC (brc_price, 5.4) contre rho (correlation entre
                les 2 actifs de reference S0/SIGMA), a H, C, N fixes -- MC
                brut ET avec control variate reutilise de l'Acte II (meme
                esprit que BOSS 2/3 : EX model-free ou semi-analytique,
                demi-largeur CV <= demi-largeur MC a chaque point) ;
        (bas)   prix du BRC contre n_assets (2, 3, 4 actifs correles a rho
                commun), meme H/C/N -- montre comment le cout de
                l'optionalite vendue grandit avec le nombre d'actifs (plus
                d'actifs, plus le "pire" est susceptible d'etre mauvais).

  figures/boss5_results.json
      {
        "params": {"S0": [...], "sigma": [...], "r": ..., "T": ...,
                   "n_steps": ..., "N": ..., "C": ..., "H": ...},
        "sweep_rho": [
          {"rho": 0.1, "price_mc": ..., "half_width_mc": ...,
           "price_cv": ..., "half_width_cv": ..., "c_hat": ...}, ...
        ]                                                >= 5 points, rho croissant
        "sweep_n_assets": [
          {"n_assets": 2, "price": ..., "half_width": ...}, ...
        ]                                                >= 2 points, n_assets croissant
      }
      Python floats only, json.dump rejette np.float64.

Le control retenu est faible et model-free : `X = exp(-rT)*paths[:,0,-1]`
(valeur actualisee du premier actif seul, `EX = S0[0]`, identite de
martingale) -- `c_hat` reste petit (~0.001), gain modeste sur la demi-largeur
(cf. PROGRESS.md, session 9). Une grille de tirages DIFFERENTE a chaque rho
testee noierait l'effet de rho dans le bruit MC (meme piege qu'en BOSS 2) :
une seule graine fixe par point de la courbe.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import multi_asset
import barriers
import mc_engine

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def main() -> None:
    """Produce figures/boss5_brc_worstof.png and figures/boss5_results.json."""
    coupon = 0.05
    barrier = 0.7
    S0 = np.array([100.0, 100.0])
    sigma = np.array([0.2, 0.2])
    r = 0.05
    T = float(1.0)
    n_steps = 50
    N = 100_000

    rho_values = np.linspace(0.0, 1.0, 5)
    prices_mc = []
    half_widths_mc = []
    prices_cv = []
    half_widths_cv = []
    c_hats = []

    for rho in rho_values:
        rho_matrix = np.array([[1.0, rho], [rho, 1.0]])
        paths = multi_asset.correlated_gbm_paths(S0, sigma, rho_matrix, r, T, n_steps, N
                                                         , rng=np.random.default_rng(seed=42))
        perf = multi_asset.worst_of_performance(paths)
        knock_in = perf.min(axis=1) < barrier
        put_payoff = np.maximum(1.0 -perf[:,-1],0)*knock_in
        Y = np.exp(-r*T) * (1.0 + coupon - put_payoff)
        X = np.exp(-r*T) * paths[:, 0, -1]
        price_mc, half_width_mc = multi_asset.brc_price(paths,coupon, barrier, r, T)
        prices_mc.append(price_mc)
        half_widths_mc.append(half_width_mc)

        price_cv, half_width_cv, c_hat,_ = mc_engine.control_variate( Y,X,S0[0])
        prices_cv.append(price_cv)
        half_widths_cv.append(half_width_cv)
        c_hats.append(c_hat)

    # sweep_n_assets calcule AVANT la figure, pour pouvoir tracer le panel 3
    sweep_n_assets = []
    for n_assets in range(2, 21):
        S0_n = np.full(n_assets, 100.0)
        sigma_n = np.full(n_assets, 0.2)
        rho_matrix_n = np.full((n_assets, n_assets), 0.5)
        np.fill_diagonal(rho_matrix_n, 1.0)
        paths_n = multi_asset.correlated_gbm_paths(S0_n, sigma_n, rho_matrix_n, r, T, n_steps, N
                                                   , rng=np.random.default_rng(seed=42))
        price_n, half_width_n = multi_asset.brc_price(paths_n,coupon, barrier, r, T)
        sweep_n_assets.append({
            "n_assets": n_assets,
            "price": float(price_n),
            "half_width": float(half_width_n)
        })

    gains = [hw_mc / hw_cv for hw_mc, hw_cv in zip(half_widths_mc, half_widths_cv)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.errorbar(rho_values, prices_mc, yerr=half_widths_mc, label="MC brut", fmt="o-", capsize=5)
    ax1.errorbar(rho_values, prices_cv, yerr=half_widths_cv, label="MC + control variate", fmt="o-", capsize=5)
    ax1.set_xlabel("Correlation entre les 2 actifs de reference")
    ax1.set_ylabel("Prix du BRC worst-of")
    ax1.set_title("Prix du BRC worst-of contre rho -- MC brut vs control variate (Acte II)")
    ax1.legend()

    ax2.plot(rho_values, gains, "o-", color="tab:blue", label="Gain (hw_mc/hw_cv)")
    ax2.set_xlabel("Correlation entre les 2 actifs de reference")
    ax2.set_ylabel("Gain demi-largeur", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Gain du control variate, et c_hat associe")

    ax2_bis = ax2.twinx()
    ax2_bis.plot(rho_values, c_hats, "s--", color="tab:orange", label="c_hat")
    ax2_bis.set_ylabel("c_hat", color="tab:orange")
    ax2_bis.tick_params(axis="y", labelcolor="tab:orange")

    ax3.errorbar([p["n_assets"] for p in sweep_n_assets],
                [p["price"] for p in sweep_n_assets],
                yerr=[p["half_width"] for p in sweep_n_assets], fmt="o-", capsize=5)
    ax3.set_xlabel("n_assets (correles a rho=0.5 commun)")
    ax3.set_ylabel("Prix du BRC worst-of")
    ax3.set_title("Prix du BRC worst-of contre nombre d'actifs")

    plt.tight_layout()
    plt.savefig(FIGURES / "boss5_brc_worstof.png")
    results = {
        "params": {
            "S0": S0.tolist(),
            "sigma": sigma.tolist(),
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "N": N,
            "C": coupon,
            "H": barrier
        },
        "sweep_rho": [
            {
                "rho": float(rho),
                "price_mc": float(price_mc),
                "half_width_mc": float(half_width_mc),
                "price_cv": float(price_cv),
                "half_width_cv": float(half_width_cv),
                "c_hat": float(c_hat)
            }
            for rho, price_mc, half_width_mc, price_cv, half_width_cv, c_hat in zip(
                rho_values, prices_mc, half_widths_mc, prices_cv, half_widths_cv, c_hats
            )
        ],
        "sweep_n_assets": sweep_n_assets
    }
    json_file = FIGURES / "boss5_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
