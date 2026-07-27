import bs
import numpy as np

from mc_engine import gbm, gbm_paths, pricer_mc_put
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
#test roi

