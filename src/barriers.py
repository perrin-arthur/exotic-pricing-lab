import bs
import numpy as np

from mc_engine import gbm, gbm_paths, pricer_mc_put
from bs import put_bs
import mc_engine

def di_put(paths,K,H,r,T):
    """Pricer un down-and-in put sur trajectoires GBM.

    paths : shape (N, n_steps+1) : trajectoires GBM, S0 en colonne 0.
    K : strike du put
    H : barriere
    r : taux sans risque
    T : maturite

    Retourne (prix, demi-largeur IC 95%) du pricer MC.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) < H}
    ST = paths[:, -1]   
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def do_put(paths,K,H,r,T):
    """Pricer un down-and-out put sur trajectoires GBM.

    paths : shape (N, n_steps+1) : trajectoires GBM, S0 en colonne 0.
    K : strike du put
    H : barriere
    r : taux sans risque
    T : maturite

    Retourne (prix, demi-largeur IC 95%) du pricer MC.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) >=  H}
    ST = paths[:, -1]     
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths >= H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def van_put(paths, K, r, T):
    """Pricer un put vanille sur trajectoires GBM.

    paths : shape (N, n_steps+1) : trajectoires GBM, S0 en colonne 0.
    K : strike du put
    r : taux sans risque
    T : maturite

    Retourne (prix, demi-largeur IC 95%) du pricer MC.
    """
    N = paths.shape[0]
    ST = paths[:, -1]     
    payoff = np.maximum(K - ST, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 2.3 — Contrôle branché sur le DI put            [40 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : pricer le down-and-in put avec le put vanille comme variable de
#              contrôle, sur les mêmes trajectoires.
# DÉBLOQUE   : quête 2.4
# VALIDATION : pytest tests/test_variance_reduction.py -k di_put
#
# INDICE 1 (intuition)  : quel autre payoff, calculable sur les MÊMES chemins,
#     a un prix que tu connais déjà exactement ET bouge dans le même sens que le
#     DI put ? Regarde la parité que tu as prouvée dans test_di_do_van.
# INDICE 2 (structure)  : deux fonctions séparées. La première ne price rien :
#     elle renvoie les deux vecteurs de payoffs PAR CHEMIN (rien n'est moyenné).
#     La seconde ne fait aucune statistique : elle récupère les deux vecteurs,
#     calcule l'espérance exacte du contrôle et délègue à control_variate.
#     Aucune formule de covariance ne doit réapparaître ici.
# INDICE 3 (formule)    : Y_i = e^{-rT} (K - S_T^i)^+ 1{min_t S_t^i <= H},
#     X_i = e^{-rT} (K - S_T^i)^+, EX = put_bs(S0, K, sigma, r, T).
#
# PIÈGE : np.max / np.min renvoient un SCALAIRE (le max de tout le tableau).
#         Le max terme à terme, c'est np.maximum ; le min par chemin, c'est
#         .min(axis=1). Les deux erreurs passent silencieusement et donnent un
#         vecteur de la mauvaise forme (ou constant).
# PIÈGE : actualisation. EX = put_bs est DÉJÀ un prix actualisé. Si tu renvoies
#         X non actualisé, X̄ et EX ne vivent pas dans la même unité et le
#         contrôle décale le prix au lieu de le stabiliser.
# PIÈGE : paths[:, 0] est un ARRAY (déjà rencontré dans test_gbm_paths_shape_et_depart).
#         Le spot scalaire, c'est paths[0, 0].
def di_put_payoffs(paths: np.ndarray,
                   K: float,
                   H: float,
                   r: float,
                   T: float) -> tuple[np.ndarray, np.ndarray]:
    """Payoffs actualisés PAR CHEMIN du DI put et de son contrôle vanille.

    Rien n'est moyenné ici : cette fonction ne price pas, elle fabrique les deux
    échantillons appariés que control_variate attend.

    Params
    ------
    paths : (N, n_steps+1) trajectoires GBM, S0 en colonne 0.
    K, H, r, T : strike, barrière, taux, maturité.

    Returns
    -------
    (Y, X) : deux arrays de shape (N,), tous deux ACTUALISÉS.
        Y = payoff down-and-in put du chemin.
        X = payoff put vanille du MÊME chemin, même K, même T.

    Garanties attendues
    -------------------
    * Y.mean() == di_put(paths, K, H, r, T)[0] à la précision machine.
    * X.mean() == van_put(paths, K, r, T)[0] à la précision machine.
      (Si l'une des deux casse, c'est la cohérence d'actualisation ou la forme
      du vecteur qui est en cause, pas le hasard.)
    """
    
    ST = paths[:, -1]   
    min_paths = paths.min(axis=1)
    # payoff vaut ca pour le VAN put X = np.maximum(K - ST, 0.0)

    X = np.maximum(K - ST, 0.0)*np.exp(-r*T)
    
    # payoff vaut ca pour le DI put Y = (K − S_T)⁺ · 1{ min₀≤t≤T S_t ≤ H }
    
    Y = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)*np.exp(-r*T)
    
    return (Y,X)
    

def di_put_cv(paths: np.ndarray,
              K: float,
              H: float,
              r: float,
              T: float,
              sigma: float,
              c: float | None = None) -> tuple[float, float, float, float]:
    """DI put pricé par variable de contrôle (contrôle = put vanille).

    Params
    ------
    paths : (N, n_steps+1) trajectoires GBM, S0 en colonne 0.
    K, H, r, T : strike, barrière, taux, maturité.
    sigma : volatilité utilisée pour SIMULER paths — elle sert à calculer
        l'espérance exacte du contrôle. Une incohérence entre ce sigma et celui
        des trajectoires biaise le prix : ce n'est plus une réduction de variance,
        c'est une erreur de modèle.
    c : coefficient imposé (typiquement issu de pilot_c), ou None pour l'estimer.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat), même convention que control_variate.
        estimate doit être compatible avec di_put(paths, ...)[0] à l'IC près,
        avec une demi-largeur strictement plus petite dès que |rho_hat| est élevé.
    """
    Y,X= di_put_payoffs(paths,K,H,r,T)
    EX = bs.put_bs(paths[0, 0],K,sigma,r,T,q=0.0)
    (estimate, half_width, c_hat, rho_hat) = mc_engine.control_variate(Y,X,EX,c)
    return (estimate, half_width, c_hat, rho_hat)


if __name__ == "__main__":
    print(di_put(mc_engine.gbm_paths(100, 0.1, 0.05, 1, 252, N=100_000), 100, 85, 0.05, 1))
