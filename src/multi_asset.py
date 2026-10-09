"""ACTE V -- MULTI-ACTIF ET PRODUITS STRUCTURES.

Simule des sous-jacents correles (Cholesky), en derive le worst-of et ses
produits structures (DI put, BRC, autocall), et leurs sensibilites.

Convention du repo : la 2e valeur de retour d'un estimateur est TOUJOURS une
demi-largeur d'IC 95% (1.96*sd/sqrt(n)). Les fonctions de ce module qui
renvoient des trajectoires (pas des prix) n'ont pas cette contrainte -- meme
convention que gbm_paths / heston_paths.

Convention de shape retenue pour tout l'acte : un tableau de trajectoires
multi-actifs a la forme (N, n_assets, n_steps+1) -- l'axe des actifs entre les
chemins et le temps. paths[:, i, :] est alors exactement ce que gbm_paths
aurait renvoye pour l'actif i seul ; paths.min(axis=2) donne le minimum
temporel par actif, paths.min(axis=(1, 2)) le minimum sur actifs ET temps.

Convention de cotation (worst-of) : K et H sont des PERFORMANCES, pas des
niveaux absolus -- K=1.0 est ATM, H=0.6 une barriere a 60% du depart. C'est la
convention standard d'un term sheet autocall/BRC, et c'est ce qui permet au
cas degenere n_assets=1 de retomber sur barriers.di_put (a un facteur
d'echelle pres, cf. worst_of_di_put).
"""

import numpy as np
import barriers
from bs import vega


# ---------------------------------------------------------------------------
# Cholesky et reparation de correlation
# ---------------------------------------------------------------------------


def repair_correlation_clip(rho: np.ndarray, tol: float = 6.6e-16) -> np.ndarray:
    """Plus proche matrice de correlation valide, par clipping des valeurs propres.

    Params
    ------
    rho : (n, n) matrice symetrique a diagonale unite, PAS forcement definie
          positive (ex : correlations par paires incoherentes entre elles).
    tol : plancher applique aux valeurs propres avant reconstruction. Doit
          rester strictement positif pour que la Cholesky en aval ne tombe
          pas sur un pivot nul.

    Returns
    -------
    (n, n) matrice de correlation : symetrique, valeurs propres >= 0,
    diagonale exactement 1.

    Garanties attendues
    --------------------
    * Si rho est deja une matrice de correlation valide (symetrique,
      semi-definie positive, diagonale unite), la sortie est identique a
      l'entree a la tolerance numerique pres (idempotence).
    * La sortie est toujours utilisable telle quelle par np.linalg.cholesky.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues_clipped = np.maximum(eigenvalues, tol)
    rho_hat = np.dot(eigenvectors, np.dot(np.diag(eigenvalues_clipped), eigenvectors.T))
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(rho_hat)))
    rho_repaired = np.dot(D_inv_sqrt, np.dot(rho_hat, D_inv_sqrt))
    return rho_repaired


def correlated_gbm_paths(S0: np.ndarray,
                         sigma: np.ndarray,
                         rho: np.ndarray,
                         r: float,
                         T: float,
                         n_steps: int,
                         N: int = 100_000,
                         rng: np.random.Generator | None = None,
                         ) -> np.ndarray:
    """Trajectoires GBM multi-actifs correlees, par Cholesky.

    Params
    ------
    S0    : (n_assets,) spots initiaux.
    sigma : (n_assets,) vols constantes, une par actif.
    rho   : (n_assets, n_assets) matrice de correlation CIBLE, pas forcement
            definie positive -- reparee en interne via
            repair_correlation_clip avant la Cholesky.
    r     : taux sans risque, commun a tous les actifs (marche risque-neutre
            unique).
    T     : maturite.
    n_steps : nombre de pas de discretisation.
    N     : nombre de trajectoires.
    rng   : generateur injecte (jamais de graine capturee a l'interieur).

    Returns
    -------
    (N, n_assets, n_steps+1) array. paths[:, i, 0] == S0[i] exactement pour
    chaque actif i.

    Garanties attendues
    --------------------
    * n_assets = 1 : la distribution (moyenne, variance) de paths[:, 0, -1]
      coincide avec celle d'un gbm_paths(S0[0], sigma[0], r, T, n_steps, N)
      -- teste statistiquement (dans l'IC), pas bit a bit.
    * E[e^{-rT} S_i(T)] = S0[i] pour chaque actif (martingale, model-free).
    * la correlation empirique des log-rendements entre actifs i et j
      converge vers repair_correlation_clip(rho)[i, j] quand N grandit.
    * rho = identite -> les actifs sont simules independamment (correlation
      empirique proche de 0 hors diagonale).
    """
    if rng is None:
        rng = np.random.default_rng()
    rho_repaired = repair_correlation_clip(rho)
    L = np.linalg.cholesky(rho_repaired)
    n_assets = len(S0)
    dt = T / n_steps
    Z = rng.standard_normal(size=(N, n_steps, n_assets))
    dW = np.dot(Z, L.T) * np.sqrt(dt)  # (N, n_steps, n_assets)
    paths = np.zeros((N, n_assets, n_steps + 1))
    paths[:, :, 0] = S0
    for i in range(n_assets):
        paths[:, i, 1:] = S0[i] * np.exp(np.cumsum((r - 0.5 * sigma[i] ** 2) * dt + sigma[i] * dW[:, :, i], axis=1))
    return paths


# ---------------------------------------------------------------------------
# Worst-of : performance et DI put
# ---------------------------------------------------------------------------


def worst_of_performance(paths: np.ndarray) -> np.ndarray:
    """Performance du pire actif, a chaque date.

    Params
    ------
    paths : (N, n_assets, n_steps+1) trajectoires multi-actifs, meme
        convention de forme que `correlated_gbm_paths`.

    Returns
    -------
    (N, n_steps+1) array. perf[n, t] = min_i paths[n, i, t] / paths[n, i, 0].

    Garanties attendues
    --------------------
    * perf[:, 0] vaut exactement 1.0 pour chaque trajectoire (chaque actif
      part a une performance de 1 par construction).
    * perf[n, t] <= paths[n, i, t] / paths[n, i, 0] pour CHAQUE actif i --
      c'est un minimum, identite model-free, vraie quel que soit le modele
      derriere les trajectoires (GBM, Heston, n'importe quoi).
    * n_assets=1 : perf[n, t] == paths[n, 0, t] / paths[n, 0, 0] exactement.
    """
    perf = np.min(paths / paths[:, :, 0][:, :, np.newaxis], axis=1)
    return perf


def worst_of_di_put(paths: np.ndarray,
                    K: float,
                    H: float,
                    r: float,
                    T: float) -> tuple[float, float]:
    """Prix Monte-Carlo d'un put down-and-in worst-of, sur la performance.

    Convention de declenchement : minimum sur les actifs ET le temps --
    n'importe quel actif qui touche H a n'importe quelle date declenche,
    meme si ce n'est pas lui qui finit le pire a maturite.

    Params
    ------
    paths : (N, n_assets, n_steps+1) trajectoires multi-actifs.
    K : strike, en PERFORMANCE (K=1.0 = ATM).
    H : barriere, en PERFORMANCE (H=0.6 = 60% du depart de chaque actif).
    r, T : taux sans risque, maturite.

    Returns
    -------
    (price, half_width), meme convention IC 95% que tout le repo.

    Garanties attendues
    --------------------
    * n_assets=1 : egal (apres mise a l'echelle par S0, le payoff ici est en
      PERFORMANCE) a `barriers.di_put` applique a `paths[:, 0, :]` avec un
      strike et une barriere remis a l'echelle du prix (K*S0, H*S0).
    * rho=1 avec des sigma identiques entre actifs : `worst_of_performance`
      est alors IDENTIQUE, trajectoire par trajectoire, a la performance de
      n'importe quel actif pris seul -- le prix se reduit exactement a un DI
      put mono-actif sur cette performance (depart a 1).
    """
    perf = worst_of_performance(paths)
    knock_in = perf.min(axis=1) < H
    payoff = np.maximum(K - perf[:, -1], 0) * knock_in
    price = np.mean(np.exp(-r*T) * payoff)
    half_width = 1.96 * np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price, half_width


# ---------------------------------------------------------------------------
# Correction de continuite (BGK), appliquee au worst-of
# ---------------------------------------------------------------------------
# Heuristique : worst_of_performance n'est pas log-normale (minimum de GBM
# correles), donc la formule BGK (barriers.bgk_barrier_correction) n'a pas de
# justification theorique exacte ici -- chaque actif est corrige
# individuellement, barriere commune appliquee ensuite sur la performance du
# pire. Documente comme approximation (PROGRESS.md), pas comme resultat
# rigoureux.


def worst_of_di_put_continuous(paths: np.ndarray,
                               K: float,
                               H: float,
                               r: float,
                               T: float,
                               sigma: np.ndarray) -> tuple[float, float]:
    """Worst-of DI put, monitoring discret corrige par actif (BGK, heuristique).

    Params
    ------
    paths : (N, n_assets, n_steps+1).
    K, H : strike/barriere en PERFORMANCE (convention worst-of).
    r, T : taux, maturite.
    sigma : (n_assets,) vols -- une correction BGK par actif (direction
        "down", seul sens utile pour un knock-in worst-of), barriere
        ajustee independamment pour chacun avant de reprendre le worst-of.

    Returns
    -------
    (price, half_width), meme convention que worst_of_di_put.

    Garanties attendues
    --------------------
    * sigma uniformement nul : H_ajustee = H pour chaque actif (cf.
      bgk_barrier_correction), le resultat doit alors etre identique a
      worst_of_di_put -- pas de correction sans volatilite.
    * rho proche de 1, vols et spots egaux entre actifs : prix dans l'IC du
      di_put_continuous mono-actif applique a la performance commune.
    """
    n_steps = paths.shape[2] - 1
    perf = worst_of_performance(paths)
    H_ajustee = barriers.bgk_barrier_correction(H, sigma[0], T, n_steps, direction="down")
    knock_in = perf.min(axis=1) < H_ajustee
    payoff = np.maximum(K - perf[:, -1], 0) * knock_in
    price = np.mean(np.exp(-r*T) * payoff)
    half_width = 1.96 * np.exp(-r*T) * np.std(np.maximum(K - perf[:, -1], 0) * (perf.min(axis=1) < H), ddof=1) / np.sqrt(len(perf))
    return price, half_width


# ---------------------------------------------------------------------------
# BRC : decomposition coupon / put worst-of vendu
# ---------------------------------------------------------------------------
# Un Barrier Reverse Convertible, c'est un zero-coupon (nominal + coupon)
# MOINS un put down-and-in worst-of vendu par l'investisseur a la banque --
# decomposition exacte, pas une approximation.


def brc_price(paths: np.ndarray,
             C: float,
             H: float,
             r: float,
             T: float) -> tuple[float, float]:
    """Prix MC d'un Barrier Reverse Convertible worst-of, coupon fixe unique.

    Params
    ------
    paths : (N, n_assets, n_steps+1) trajectoires multi-actifs.
    C : coupon fixe, en fraction du nominal (nominal = 1.0), paye a T que la
        barriere ait ete touchee ou non.
    H : barriere de declenchement, en PERFORMANCE.
    r, T : taux sans risque, maturite.

    Returns
    -------
    (price, half_width), meme convention IC 95% que tout le repo. Prix pour
    un nominal de 1.0 -- multiplie par le nominal reel pour un term sheet.

    Garanties attendues
    --------------------
    * Decomposition EXACTE (memes tirages) : price == C*exp(-r*T) +
      exp(-r*T) - worst_of_di_put(paths, 1.0, H, r, T)[0], a 1e-12.
    * H -> 0 (barriere jamais touchee) : price -> (C + 1.0) * exp(-r*T), le
      produit redevient un pur zero-coupon + coupon, aucune optionalite
      vendue.
    """
    price = C * np.exp(-r*T) + 1.0 * np.exp(-r*T) - worst_of_di_put(paths, 1.0, H, r, T)[0]
    half_width = 1.96 * np.exp(-r*T) * np.std(np.maximum(1.0 - worst_of_performance(paths)[:, -1], 0) * (worst_of_performance(paths).min(axis=1) < H), ddof=1) / np.sqrt(len(paths))
    return price, half_width


# ---------------------------------------------------------------------------
# Autocall / Phoenix worst-of
# ---------------------------------------------------------------------------
# Generalise le BRC a plusieurs dates de rappel : des que la performance du
# pire actif depasse un niveau de rappel a une date d'observation, le
# produit rembourse nominal + coupons cumules par anticipation, et s'arrete
# la. Jamais rappele -> le payoff a maturite retombe sur le BRC. Le knock-in
# (barriere H) continue de s'observer meme apres une date d'autocall manquee
# -- seul un rappel arrete le monitoring.


def recall_index(perf_obs: np.ndarray, autocall_level: float) -> np.ndarray:
    """Indice (dans perf_obs) de la premiere date de rappel par trajectoire.

    Params
    ------
    perf_obs : (N, n_obs) performance du pire actif, restreinte aux SEULES
        dates d'observation (pas toutes les dates simulees).
    autocall_level : niveau de performance au-dela duquel le produit
        rappelle (typiquement 1.0, ATM).

    Returns
    -------
    (N,) tableau d'entiers : indice (0-based, dans perf_obs) de la premiere
    date ou perf_obs >= autocall_level, ou -1 si jamais rappele.
    """
    mask = perf_obs >= autocall_level
    first_occurrence = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
    return first_occurrence


def autocall_price(paths: np.ndarray,
                   obs_steps: np.ndarray,
                   autocall_level: float,
                   C: float,
                   H: float,
                   r: float,
                   T: float) -> tuple[float, float]:
    """Prix MC d'un autocall/Phoenix worst-of.

    Params
    ------
    paths : (N, n_assets, n_steps+1).
    obs_steps : (n_obs,) indices (dans l'axe temps de paths) des dates
        d'observation du rappel, strictement croissants, le dernier etant
        n_steps (la maturite).
    autocall_level : niveau de rappel, en performance.
    C : coupon par periode ecoulee, verse (cumule) en cas de rappel.
    H : barriere de knock-in, en performance, monitoree sur TOUTE la
        trajectoire.
    r, T : taux, maturite.

    Returns
    -------
    (price, half_width), convention IC 95% du repo.

    Garanties attendues
    --------------------
    * obs_steps = [n_steps] (une seule date, la maturite) : autocall_price
      egale exactement brc_price(paths, C, H, r, T) -- un cas particulier
      "digital" connu, identite model-free, memes tirages.
    * la probabilite de rappel (fraction de trajectoires avec
      recall_index != -1) croit quand rho augmente et decroit quand sigma
      augmente -- verifie la MONOTONIE, pas une valeur precise.
    """
    perf = worst_of_performance(paths)
    first_occurrence = recall_index(perf[:, obs_steps], autocall_level)
    payoff = np.zeros(len(paths))
    for n in range(len(paths)):
        if first_occurrence[n] != -1:
            t_rappel = obs_steps[first_occurrence[n]] * (T / (paths.shape[2] - 1))
            payoff[n] = (1 + C * (first_occurrence[n] + 1)) * np.exp(-r * t_rappel)
        else:
            knock_in = perf[n, :].min() < H
            perte = np.maximum(1.0 - perf[n, -1], 0) * knock_in
            payoff[n] = (1.0 + C * len(obs_steps) - perte) * np.exp(-r * T)

    price = np.mean(payoff)
    half_width = 1.96 * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price, half_width


# ---------------------------------------------------------------------------
# Sensibilites : delta par actif, vega, rho
# ---------------------------------------------------------------------------
# Meme technique qu'en 1.4 (delta_mc, bump + common random numbers)
# generalisee a n_assets : bumper UN SEUL actif a la fois, les autres et les
# tirages inchanges, pour isoler sa sensibilite propre.


def worst_of_deltas(S0: np.ndarray,
                    sigma: np.ndarray,
                    rho: np.ndarray,
                    r: float,
                    T: float,
                    n_steps: int,
                    N: int,
                    K: float,
                    H: float,
                    h: float = 1.0,
                    seed: int = 42) -> np.ndarray:
    """Delta par actif du worst-of DI put, bump + CRN.

    Params
    ------
    S0, sigma, rho, r, T, n_steps, N : parametres de simulation, memes
        conventions que correlated_gbm_paths.
    K, H : strike/barriere en performance.
    h : taille du bump sur S0[i], en NIVEAU (pas en performance).
    seed : graine fixe, rejouee a l'identique pour chaque bump (CRN).

    Returns
    -------
    (n_assets,) delta_i = d(prix)/d(S0_i), un par actif.

    Garanties attendues
    --------------------
    * IDENTIQUEMENT nul : K et H etant en PERFORMANCE, le payoff ne depend
      que de paths/paths[:,:,0] = exp(increments) -- S0 se simplifie
      algebriquement, avant toute simulation. Un bump de S0 ne peut donc
      jamais deplacer le prix d'un produit 100% performance (identite
      model-free, pas une mesure).
    * pres de la barriere H, le delta (et tout bump en general) devient
      instable (grande variance d'un seed a l'autre) -- meme phenomene que
      4.5, a documenter, pas a corriger silencieusement.
    """
    delta = np.zeros(len(S0))
    for i in range(len(S0)):
        S0_up, S0_down = S0.copy(), S0.copy()
        S0_up[i] += h
        S0_down[i] -= h
        price_up = worst_of_di_put(correlated_gbm_paths(S0_up, sigma, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
        price_down = worst_of_di_put(correlated_gbm_paths(S0_down, sigma, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
        delta[i] = (price_up - price_down) / (2 * h)
    return delta


def worst_of_vega(S0: np.ndarray,
                  sigma: np.ndarray,
                  rho: np.ndarray,
                  r: float,
                  T: float,
                  n_steps: int,
                  N: int,
                  K: float,
                  H: float,
                  h: float = 0.01,
                  seed: int = 42) -> np.ndarray:
    """Vega par actif du worst-of DI put, bump + CRN sur sigma[i].

    Meme technique que worst_of_deltas, bump sur sigma au lieu de S0.

    Returns
    -------
    (n_assets,) vega_i = d(prix)/d(sigma_i).
    """
    vega = np.zeros(len(sigma))
    for i in range(len(sigma)):
       sigma_up, sigma_down = sigma.copy(), sigma.copy()
       sigma_up[i] += h
       sigma_down[i] -= h
       price_up = worst_of_di_put(correlated_gbm_paths(S0, sigma_up, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
       price_down = worst_of_di_put(correlated_gbm_paths(S0, sigma_down, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
       vega[i] = (price_up - price_down) / (2 * h)
    return vega


def worst_of_rho_sensitivity(S0: np.ndarray,
                             sigma: np.ndarray,
                             rho: np.ndarray,
                             r: float,
                             T: float,
                             n_steps: int,
                             N: int,
                             K: float,
                             H: float,
                             i: int,
                             j: int,
                             h: float = 0.01,
                             seed: int = 42) -> float:
    """Sensibilite du prix a la correlation CIBLE rho[i, j] (bump + CRN).

    Nom choisi pour eviter la confusion avec le taux sans risque `r`. Bump
    un SEUL coefficient hors-diagonal de `rho` (et son symetrique rho[j,i]),
    les autres inchanges, meme graine pour les deux pricings.

    Returns
    -------
    float : d(prix)/d(rho_ij), pour la paire d'actifs (i, j) passee.
    """
    rate = np.zeros((len(S0), len(S0)))
    rate[i, j] = h
    rate[j, i] = h
    price_up = worst_of_di_put(correlated_gbm_paths(S0, sigma, rho + rate, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
    price_down = worst_of_di_put(correlated_gbm_paths(S0, sigma, rho - rate, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
    return float(price_up - price_down) / (2 * h)
