"""BOSS 2 — balayage de la barriere du DI put, de 60% a 95% du spot.

Lance :  .venv/bin/python scripts/boss2_barrier_sweep.py
Valide par : pytest tests/test_variance_reduction.py::test_boss2_artefacts

CE QUE LE SCRIPT DOIT PRODUIRE (contrat lu par le test — a respecter au nom
de cle pres) :

  figures/boss2_barrier_sweep.png
      Deux panneaux partageant l'axe des abscisses H/S0 :
        (haut) rho(H) — la correlation entre le payoff DI put et le payoff du
               controle vanille ;
        (bas)  ratio des demi-largeurs d'IC, CV / MC brut, avec la ligne
               horizontale y=1 (la frontiere "le controle ne sert plus a rien").

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
      Trie par H_pct croissant, premier point a 0.60, dernier a 0.95,
      au moins 6 points. Que des floats PYTHON (json.dump refuse np.float64 —
      d'ou le cast impose dans control_variate).

LA LECTURE ATTENDUE (c'est ca, le boss — pas le code) :
  quand H remonte vers le spot, le DI put ressemble de plus en plus au put
  vanille, rho monte vers 1 et le ratio s'effondre. Quand H descend, le
  declenchement devient rare, le controle n'explique presque plus rien de la
  variance du DI put — et pourtant le ratio ne depasse jamais 1. Savoir dire
  pourquoi en une phrase vaut plus que le script.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib
matplotlib.use("Agg")            # backend fichier : aucun besoin d'affichage
import matplotlib.pyplot as plt
import numpy as np

import barriers
from mc_engine import gbm_paths, pilot_c

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


# ╔═══════════════════════════════════════════════════════════════╗
# ║ BOSS 2 — Balayage de barriere + figure                [80 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : mesurer le gain de la variable de controle en fonction du niveau
#              de barriere, et en produire une figure defendable en entretien.
# DEBLOQUE   : ACTE III (Heston mono-actif)
# VALIDATION : pytest tests/test_variance_reduction.py::test_boss2_artefacts
#
# INDICE 1 (intuition)  : tu as deja tout. Une boucle sur H, et a chaque tour
#     les deux pricers que tu viens d'ecrire. La seule vraie decision est :
#     est-ce que chaque H a droit a ses propres trajectoires, ou est-ce que
#     tout le balayage tourne sur les MEMES ? Tranche, et sache justifier.
# INDICE 2 (structure)  : reutilise les MEMES trajectoires pour tous les H
#     (common random numbers) — sinon la courbe rho(H) tremble du bruit MC et
#     tu ne sais plus si le creux que tu vois est un effet ou un artefact.
#     C'est exactement l'argument de delta_mc. Une seule simulation, une boucle,
#     une liste de dicts, un json.dump, deux subplots.
# INDICE 3 (formule)    : ratio = half_width_cv / half_width_mc ; le gain
#     theorique vaut sqrt(1 - rho^2) — trace-le en pointilles par-dessus le
#     ratio mesure, les deux courbes doivent se superposer. C'est ce
#     recouvrement qui prouve que ton estimateur fait ce que la theorie dit.
#
# PIEGE : FIGURES.mkdir(parents=True, exist_ok=True) avant de sauver, sinon
#         plt.savefig plante sur un dossier absent (figures/ est vide au depart).
# PIEGE : np.float64 n'est pas serialisable par json.dump. Caste.
# PIEGE : H_pct = 0.60 doit donner H = 60.0 pour S0 = 100 — mais ecris
#         H = H_pct * S0, pas 60.0 en dur : le jour ou S0 change, la courbe doit
#         suivre.
# PIEGE : plt.show() dans un script lance en batch bloque le terminal. Le
#         backend Agg ci-dessus est la pour ca ; ne le retire pas.
def main() -> None:
    """Produit figures/boss2_barrier_sweep.png et figures/boss2_results.json.

    Parametres de reference du repo : S0=100, K=100, sigma=0.2, r=0.05, T=1,
    n_steps=50. Prends N assez grand pour que la courbe rho(H) soit lisse
    (100 000 chemins est un bon depart) et au moins 8 niveaux de barriere
    entre 0.60 et 0.95 inclus.
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
    ax_h.plot(H_pcts,rhos,marker = "o")
    ax_h.set_ylabel("ρ(DI put, put vanille)")

    ax_b.plot(H_pcts,ratio_half_widths, marker="o", label= "ratio mesuré")
    ax_b.plot(H_pcts,curve_theoretical,linestyle="--",label="courbe théorique")
    ax_b.axhline(1.0,label="")
    ax_b.set_ylabel("demi-IC CV / demi-IC MC")
    ax_b.set_xlabel("H / S0 ")
    ax_b.legend()
    
    ax_h.grid(True, alpha=0.3)
    ax_b.grid(True, alpha=0.3)
    fig.suptitle("DI put: gain de la variable de contrôle (contrôle = put vanille) \n S0=100, K=100, σ=0.20, r=0.05, T=1, n_steps=50, N=100 000")
    fig.tight_layout()

    fig.savefig(FIGURES / "boss2_barrier_sweep.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()