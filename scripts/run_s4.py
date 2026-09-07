"""Seance 4 — tableau de synthese quete 2.3 (variable de controle sur DI put).

Script JETABLE : il ne fait que mesurer, il n'introduit aucune brique nouvelle.
Tout le calcul vient de src/barriers.py et src/mc_engine.py.

Lancer : .venv/bin/python scripts/run_s4.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import barriers
from mc_engine import gbm_paths

# --- parametres communs ----------------------------------------------------
S0, K, T = 100.0, 100.0, 1.0
R, SIGMA = 0.02, 0.30
N_STEPS, N = 252, 100_000
SEED = 20240904
BARRIERES = (65.0, 85.0)


def ligne(H, paths):
    """Une barriere -> tous les chiffres de la ligne du tableau.

    Les deux estimateurs tournent sur les MEMES trajectoires : l'ecart entre
    eux n'est donc pas du bruit d'echantillonnage independant, il vaut
    exactement c*(X_barre - EX).
    """
    mc, ci_mc = barriers.di_put(paths, K=K, H=H, r=R, T=T)
    cv, ci_cv, _, rho = barriers.di_put_cv(paths, K=K, H=H, r=R, T=T, sigma=SIGMA)

    # parite pathwise KI + KO = vanille, sur les memes chemins : identite
    # algebrique, pas une egalite statistique -> attendu au niveau machine.
    ko, _ = barriers.do_put(paths, K=K, H=H, r=R, T=T)
    van, _ = barriers.van_put(paths, K=K, r=R, T=T)

    return {
        "H": H,
        "mc": mc,
        "ci_mc": ci_mc,
        "cv": cv,
        "ci_cv": ci_cv,
        "rho": rho,
        "gain": ci_mc / ci_cv,          # >1 = l'IC a ete divise par ce facteur
        "parite": mc + ko - van,
    }


def main():
    paths = gbm_paths(S0, SIGMA, R, T, n_steps=N_STEPS, N=N,
                      rng=np.random.default_rng(SEED))
    res = [ligne(H, paths) for H in BARRIERES]

    print(f"\n**Parametres** : S0={S0:.0f}, K={K:.0f}, T={T:.0f}, r={R}, "
          f"sigma={SIGMA}, n_steps={N_STEPS}, N={N:,}, seed={SEED}\n")

    head = ("|       H | Prix MC | demi-IC | Prix CV | demi-IC | rho_hat "
            "| gain IC | parite KI+KO-van |")
    sep = ("|--------:|--------:|--------:|--------:|--------:|--------:"
           "|--------:|-----------------:|")
    print(head)
    print(sep)
    for d in res:
        print(f"| {d['H']:7.4f} | {d['mc']:7.4f} | {d['ci_mc']:7.4f} "
              f"| {d['cv']:7.4f} | {d['ci_cv']:7.4f} | {d['rho']:7.4f} "
              f"| {d['gain']:6.4f}x | {d['parite']:16.4e} |")

    print("\n(gain IC = demi-IC brut / demi-IC controle : facteur de reduction "
          "de la largeur d'intervalle, a budget de trajectoires identique.)")


if __name__ == "__main__":
    main()
