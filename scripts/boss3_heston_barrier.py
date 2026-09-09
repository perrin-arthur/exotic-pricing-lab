"""BOSS 3 — DI put sous Heston, avec variable de controle. Et le smile.

Lance :  .venv/bin/python scripts/boss3_heston_barrier.py
Valide par : pytest tests/test_heston.py::test_boss3_artefacts

CE QUE LE SCRIPT DOIT PRODUIRE (contrat lu par le test — au nom de cle pres) :

  figures/boss3_heston.png
      Deux panneaux :
        (haut) le smile : vol implicite contre log-moneyness log(K/F) — pas
               contre K, sinon deux maturites ne sont pas comparables ;
        (bas)  le meme balayage de barriere que le BOSS 2, mais sous Heston :
               ratio des demi-largeurs CV / MC en fonction de H/S0.

  figures/boss3_results.json
      {
        "params": {"S0":…, "K":…, "v0":…, "r":…, "T":…, "kappa":…, "theta":…,
                   "xi":…, "rho":…, "n_steps":…, "N":…},
        "smile": [{"K":…, "log_moneyness":…, "iv":…}, …],        >= 7 points
        "sweep": [{"H_pct":0.60, "H":60.0, "rho":…, "price_mc":…,
                   "half_width_mc":…, "price_cv":…, "half_width_cv":…,
                   "ratio_half_width":…, "c_hat":…}, …]           >= 6 points
      }
      sweep trie par H_pct croissant, de 0.60 a 0.95. Que des floats PYTHON.

  ATTENTION : la cle "rho" du sweep est la CORRELATION payoff/controle (comme au
  BOSS 2), pas le rho du modele. Le rho du modele vit dans "params". Deux objets
  differents, meme lettre — c'est la notation du marche, on fait avec.

LA LECTURE ATTENDUE (c'est ca, le boss) :
  1. Le smile n'est plus plat. Sous BS, le DI put a un prix ; sous Heston il en
     a un autre, et l'ecart n'est pas du bruit — la barriere est un evenement de
     QUEUE, donc extremement sensible a la vol des vols et au skew. Sache dire
     de quel cote ca bouge et pourquoi.
  2. Le dispositif de reduction de variance, lui, ne change pas d'une ligne.
     Seul EX change de source : put_bs devient heston_put. La courbe de gain a
     la meme forme qu'au BOSS 2. Message : la reduction de variance est une
     technique STATISTIQUE, independante du modele.
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


# ╔═══════════════════════════════════════════════════════════════╗
# ║ BOSS 3 — DI put sous Heston + smile                    [80 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : rebrancher tout l'Acte II sur le modele de l'Acte III, et
#              montrer que rien ne casse.
# DEBLOQUE   : ACTE IV (multi-actif : Cholesky, worst-of, BRC)
# VALIDATION : pytest tests/test_heston.py::test_boss3_artefacts
#
# INDICE 1 (intuition) : tu as deja ecrit ce script. C'est boss2_barrier_sweep,
#     avec deux substitutions : gbm_paths -> heston_paths, et put_bs -> heston_put.
#     Si tu te retrouves a reecrire la logique du controle, arrete-toi : c'est le
#     signe que tes couches sont mal separees.
# INDICE 2 (structure) : barriers.di_put et barriers.di_put_payoffs prennent des
#     TRAJECTOIRES, pas un modele — elles marchent telles quelles sur les
#     trajectoires Heston. En revanche barriers.di_put_cv appelle put_bs en dur
#     (regarde la ligne), donc elle n'est PAS reutilisable ici. Deux options :
#     (a) appeler di_put_payoffs puis control_variate a la main dans ce script,
#     (b) generaliser di_put_cv pour qu'elle accepte EX en argument.
#     (b) est la bonne reponse d'ingenieur, (a) va plus vite. Tranche, et sache
#     dire pourquoi — c'est une vraie question de design d'entretien.
# INDICE 3 (formule) : aucune formule nouvelle dans tout le boss.
#
# PIEGE : le controle doit etre le put vanille SUR LES MEMES TRAJECTOIRES
#         Heston, et son EX doit etre heston_put avec les MEMES parametres. Un
#         EX calcule en BS sur des trajectoires Heston decale le prix : c'est le
#         piege "actualisation incoherente" de la seance 4, version modele.
# PIEGE : heston_paths est BEAUCOUP plus lent que gbm_paths (boucle en temps).
#         Simule UNE fois avant la boucle sur H, comme au BOSS 2 — et cette
#         fois-ci ce n'est plus seulement pour la qualite de la courbe, c'est
#         aussi ce qui rend le script tenable.
# PIEGE : n_steps compte double ici. Il discretise la barriere (monitoring) ET
#         le schema d'Euler. Un n_steps trop petit biaise le prix deux fois,
#         pour deux raisons independantes.
# PIEGE : le smile se trace contre log(K/F) avec F = S0*exp(r*T), pas contre K.
def main() -> None:
    """Produit figures/boss3_heston.png et figures/boss3_results.json.

    Parametres suggeres : ceux de HP dans tests/test_heston.py
    (S0=100, v0=0.04, r=0.05, T=1, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7),
    n_steps=100, N=100_000, strikes de 80 a 120, barrieres de 0.60 a 0.95.
    """
    #params
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
