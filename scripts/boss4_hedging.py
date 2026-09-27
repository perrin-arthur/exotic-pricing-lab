"""Convergence de l'erreur de hedging, et distribution du P&L de réplication.

Run:       .venv/bin/python scripts/boss4_hedging.py
Validated: pytest tests/test_hedging.py::test_boss4_artefacts

Output contract, read by the test down to the key names:

  figures/boss4_hedging.png
      Two panels:
        (top)    log-log scatter of std(hedging_error) against n_rebal, sur la
                 grille de 4.2, avec la droite théorique de pente -1/2 tracée
                 par-dessus (ancrée au premier point de la grille) ;
        (bottom) trois histogrammes superposés (ou trois distributions
                 côte à côte) du P&L de réplication, à n_rebal fixé : modèle
                 correct (4.2), modèle faux -- vol arbitrage (4.3), avec coûts
                 de transaction (4.4). Même axe des x pour les trois, pour
                 que l'écrasement relatif des distributions soit lisible.

  figures/boss4_results.json
      {
        "params": {"S0": ..., "K": ..., "sigma": ..., "r": ..., "T": ...,
                   "sigma_impl": ..., "sigma_real": ..., "cost_rate": ...,
                   "n_steps_pnl": ..., "N": ...},
        "convergence": [
          {"n_steps": 8, "std_error": ...}, ...
        ]                                              >= 5 points, croissants
        "pnl": {
          "modele_correct":  {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...},
          "modele_faux":     {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...},
          "avec_couts":      {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ...}
        },
        "slope_measured": ...             # pente log-log mesurée sur "convergence"
      }
      Python floats only, json.dump rejette np.float64.

Ce que la figure doit montrer : le panneau du haut est la même mesure que
`test_hedging_error_moyenne_dans_ic_et_pente_log_log` (4.2), simplement tracée
au lieu d'être seulement testée -- la pente -1/2 doit coller aux points. Le
panneau du bas met en regard trois défaillances distinctes d'un même hedge :
le bruit de discrétisation pur (modèle correct, centré en zéro), le biais de
modèle (vol arbitrage, décalé et non centré si sigma_real != sigma_impl), et
le coût certain (avec coûts, décalé négativement, jamais centré en zéro).
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

    QUÊTE BOSS 4 — convergence et P&L de réplication [80 XP]
    OBJECTIF : assembler en une seule figure les trois résultats défendables
        en entretien annoncés en tête d'acte -- la convergence en
        `n_rebal^{-1/2}` (4.2), l'identité de vol arbitrage (4.3), et l'effet
        des coûts de transaction (4.4). Rien de nouveau à calculer : ce boss
        ne fait qu'appeler ce qui existe déjà dans `hedging.py` et le mettre
        en image, exactement comme `boss2_barrier_sweep.py` et
        `boss3_heston_barrier.py` avant lui.
    DÉBLOQUE : rien -- ferme l'Acte IV.
    VALIDATION : `test_boss4_artefacts` -- vérifie que les DEUX fichiers
        existent, que `convergence` a au moins 5 points croissants en
        `n_steps`, que `slope_measured` est dans la même fourchette que le
        test de 4.2 ([-0.8, -0.2]), et que les trois entrées de `pnl` ont bien
        leurs 5 clés. Pas de nouvelle identité numérique ici -- tout a déjà
        été validé quête par quête, ce script ne fait que les rejouer et les
        tracer ensemble.
    INDICE 1 (intuition) : ce script ne teste rien de nouveau, il RACONTE
        l'acte -- reprends telles quelles les grilles et formules déjà
        validées dans `tests/test_hedging.py` (`n_steps_grid`,
        `sigma_impl`/`sigma_real`, `cost_rate`), pas besoin d'inventer de
        nouveaux paramètres.
    INDICE 2 (structure) :
        - Panneau haut : boucle sur une grille de `n_steps` (comme dans
          `test_hedging_error_moyenne_dans_ic_et_pente_log_log`), un
          `gbm_paths` + `bs_delta_hedge_deltas` + `hedging_error` par point,
          collecte `errors.std(ddof=1)`. `np.polyfit` sur `log(n_steps)` vs
          `log(std)` donne `slope_measured` ; la droite théorique se trace
          avec `std[0] * (n_steps_grid / n_steps_grid[0])**(-0.5)`.
        - Panneau bas : UN SEUL jeu de trajectoires GBM (à `sigma_real`, pour
          le modèle faux), `n_steps` fixé une fois pour toutes
          (`n_steps_pnl`) : `hedging_error` (modèle correct, sur des
          trajectoires à `sigma_impl` -- donc un DEUXIÈME jeu de
          trajectoires, à `sigma_impl` cette fois), `vol_arbitrage_pnl`
          (modèle faux, sur les trajectoires à `sigma_real`), et
          `hedging_error_with_costs` (modèle correct + coûts, sur les
          trajectoires à `sigma_impl`, `cost_rate` > 0).
    INDICE 3 (formule) : rien de nouveau -- tous les appels existent déjà
        (`bs_delta_hedge_deltas`, `hedging_error`, `hedging_error_stats`,
        `vol_arbitrage_pnl`, `hedging_error_with_costs`). Les percentiles du
        JSON s'obtiennent avec `np.percentile(array, [5, 50, 95])`.
    PIÈGE : les TROIS distributions du panneau du bas ne vivent pas sous le
        même jeu de trajectoires ni le même budget -- "modèle correct" est
        centré en zéro par construction (4.2), "modèle faux" ne l'est QUE si
        `sigma_real != sigma_impl` (sinon il retombe sur "modèle correct"),
        "avec coûts" n'est JAMAIS centré en zéro (une perte certaine, cf.
        PIÈGE de `hedging_error_with_costs` en 4.4) -- ne pas s'étonner que
        les trois histogrammes n'aient ni la même moyenne ni la même
        étendue, c'est le point de la figure.
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

    ax_haut.loglog(n_steps_grid, stds, "o", label="mesuré")
    ax_haut.loglog(n_steps_grid, droite, "--", label="pente -1/2")
    ax_haut.set_xlabel("n_rebal")
    ax_haut.set_ylabel("std(hedging_error)")
    ax_haut.legend()
    
    ax.hist(hedging.portfolio_terminal_value(S_impl, deltas, r, T, V0=put_bs(S0,K,sigma_impl,r,T)) - np.maximum(K - S_impl[:, -1], 0),
            bins=50, density=True, alpha=0.5, label="modèle correct")

    ax.hist(hedging.vol_arbitrage_pnl(paths=S_real, K=K, sigma_impl=sigma_impl, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), option="put")
            ,bins=50, density=True, alpha=0.5, label="modèle faux")
    ax.hist(hedging.hedging_error_with_costs(paths=S_impl,deltas=deltas, K = K, r=r, T=T,V0=put_bs(S0,K,sigma_impl,r,T), cost_rate = cost_rate, option="put"),
            bins=50, density=True, alpha=0.5, label="avec coûts")
    ax.set_xlabel("P&L de réplication")
    ax.set_ylabel("densité")
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
