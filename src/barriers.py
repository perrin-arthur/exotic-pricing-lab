"""Down-and-in / down-and-out put payoffs, evaluated on simulated paths.

These pricers take a path array rather than a model, so they work unchanged on
Black-Scholes paths (mc_engine.gbm_paths) and on Heston paths
(heston.heston_paths).

Monitoring is DISCRETE: the barrier is tested at the n_steps+1 simulated dates
only. A continuously monitored barrier is knocked in more often, so these prices
carry a discretisation bias that does not vanish as N grows; the Broadie-
Glasserman-Kou continuity correction is not implemented.

Barrier convention: the knock-in condition is strict, min_t S_t < H. Since the
knock-out condition is min_t S_t >= H, the two are exact complements and the
identity DI + DO = vanilla holds pathwise, to machine precision.
"""

import numpy as np

import bs
import mc_engine

def di_put(paths,K,H,r,T):
    """Monte-Carlo price of a down-and-in put.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    H : barrier level.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) < H}
    ST = paths[:, -1]
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def do_put(paths,K,H,r,T):
    """Monte-Carlo price of a down-and-out put.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    H : barrier level.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) >=  H}
    ST = paths[:, -1]
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths >= H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def van_put(paths, K, r, T):
    """Monte-Carlo price of a vanilla put, on the same path array.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    ST = paths[:, -1]
    payoff = np.maximum(K - ST, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def di_put_payoffs(paths: np.ndarray,
                   K: float,
                   H: float,
                   r: float,
                   T: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-path discounted payoffs of the DI put and of its vanilla control.

    Nothing is averaged here: this function does not price, it builds the two
    paired samples that `mc_engine.control_variate` expects.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K, H, r, T : strike, barrier, rate, maturity.

    Returns
    -------
    (Y, X) : two (N,) arrays, both DISCOUNTED.
        Y = down-and-in put payoff of the path.
        X = vanilla put payoff of the SAME path, same K, same T.

    Notes
    -----
    Y and X must live in the same units as the control's expectation EX, which
    is a discounted price. Returning them undiscounted shifts the estimate
    instead of stabilising it.

    Y.mean() equals di_put(paths, K, H, r, T)[0] to machine precision, and
    X.mean() equals van_put(paths, K, r, T)[0].
    """

    ST = paths[:, -1]
    min_paths = paths.min(axis=1)

    X = np.maximum(K - ST, 0.0)*np.exp(-r*T)

    Y = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)*np.exp(-r*T)

    return (Y,X)


def di_put_cv(paths: np.ndarray,
              K: float,
              H: float,
              r: float,
              T: float,
              sigma: float,
              c: float | None = None) -> tuple[float, float, float, float]:
    """DI put priced with the vanilla put as control variate, under Black-Scholes.

    The control's expectation is taken from the closed-form BS put, so this
    helper is tied to the Black-Scholes model. Pricing a DI put on Heston paths
    means calling `di_put_payoffs` and `mc_engine.control_variate` directly with
    a Heston EX, as scripts/boss3_heston_barrier.py does.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0. The spot is read
        back from paths[0, 0].
    K, H, r, T : strike, barrier, rate, maturity.
    sigma : volatility used to SIMULATE paths, needed for the control's exact
        expectation. A mismatch between this sigma and the one behind the paths
        biases the price: that is a model error, not variance reduction.
    c : imposed coefficient, typically from `mc_engine.pilot_c`, or None to
        estimate it in-sample.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat), same convention as
    `mc_engine.control_variate`. Run on the same paths as di_put, the gap
    between the two estimates is exactly c*(X_bar - EX), of the order of the
    control's own CI; a larger gap means the control is not centred.
    """
    Y,X= di_put_payoffs(paths,K,H,r,T)
    EX = bs.put_bs(paths[0, 0],K,sigma,r,T,q=0.0)
    (estimate, half_width, c_hat, rho_hat) = mc_engine.control_variate(Y,X,EX,c)
    return (estimate, half_width, c_hat, rho_hat)


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.3 — Correction de continuité (Broadie-Glasserman-Kou) [40 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : la dette technique ouverte depuis 1.7 (voir le header de ce
#              fichier) -- di_put/do_put ne monitorent la barriere qu'aux
#              n_steps+1 dates simulees, jamais en continu. Un chemin qui
#              plonge sous H ENTRE deux dates et remonte n'est jamais
#              detecte : le prix MC discret porte un biais STRUCTUREL par
#              rapport au prix d'un term sheet (barriere observee en
#              continu), un biais qui ne disparait pas quand N grandit --
#              seulement quand n_steps grandit.
# DÉBLOQUE   : multi_asset.worst_of_di_put_continuous (Acte V, 5.3)
# VALIDATION : pytest tests/test_pricers.py -k "bgk or continuous"
#
# INDICE 1 (intuition)  : Broadie, Glasserman et Kou (1997) montrent qu'un
#     prix a monitoring DISCRET avec une barriere H est, en premiere
#     approximation, egal au prix a monitoring CONTINU avec une barriere
#     legerement DEPLACEE -- vers S0 pour un cas down (plus facile a
#     toucher, ca compense les franchissements rates), a l'oppose de S0 pour
#     un cas up. Le decalage depend de sigma et du pas dt = T/n_steps : plus
#     dt est petit (n_steps grand), moins il y a de place pour un exces de
#     parcours manque entre deux dates, moins la correction pese. Corollaire
#     : n_steps -> l'infini, barriere ajustee -> H exactement.
# INDICE 2 (structure)  : une fonction PURE (la formule du decalage, aucune
#     simulation dedans -- teste-la independamment, sans Monte-Carlo) et un
#     wrapper qui rappelle di_put en lui passant la barriere ajustee a la
#     place de H. Le reste (payoff, actualisation, IC) ne change pas : toute
#     la correction tient dans UNE substitution de H.
# INDICE 3 (formule)    :
#     beta = 0.5826   # = -zeta(1/2)/sqrt(2*pi), constante universelle BGK
#     dt = T / n_steps
#     H_ajustee = H * exp(+beta * sigma * sqrt(dt))   si direction == "down"
#     H_ajustee = H * exp(-beta * sigma * sqrt(dt))   si direction == "up"
#
# PIÈGE : inverser le signe selon down/up -- verifie a la main sur un
#         exemple ou tu sais ce qui doit se passer (H_ajustee plus proche de
#         S0 pour un down, donc le knock-in doit devenir PLUS probable, donc
#         le prix DI doit AUGMENTER par rapport au prix non corrige).
# PIÈGE : sqrt(dt) et pas sqrt(T) ni sqrt(n_steps) -- dt DIMINUE quand
#         n_steps grandit, c'est bien dt qu'on veut petit a la limite ;
#         sqrt(n_steps) irait dans le mauvais sens.
# PIÈGE : appliquer la correction a H mais garder le MEME H non corrige pour
#         le payoff a maturite -- seule la condition de DECLENCHEMENT bouge,
#         pas le strike ni le reste du payoff.


def bgk_barrier_correction(H: float,
                           sigma: float,
                           T: float,
                           n_steps: int,
                           direction: str) -> float:
    """Barriere ajustee pour approcher un monitoring continu (Broadie-Glasserman-Kou).

    Params
    ------
    H : barriere cible (celle du term sheet, monitoree en continu).
    sigma : volatilite de l'actif.
    T : maturite.
    n_steps : nombre de dates de monitoring discret.
    direction : "down" ou "up" -- sens de la barriere.

    Returns
    -------
    H_ajustee : a utiliser EN PLACE de H dans un pricer a monitoring discret
        (di_put, do_put, ...) pour que son prix approche le prix a
        monitoring continu avec la vraie barriere H.

    Garanties attendues
    --------------------
    * n_steps -> l'infini (dt -> 0) : H_ajustee -> H.
    * direction="down" : H_ajustee > H (barriere ajustee plus proche de S0,
      donc plus facile a toucher).
    * direction="up" : H_ajustee < H (symetrique, plus loin de S0).
    """
    dt = T / n_steps
    beta = 0.5826  # = -zeta(1/2)/sqrt(2*pi), constante universelle BGK
    if direction == "down":
        H_ajustee = H * np.exp(beta * sigma * np.sqrt(dt))
    elif direction == "up":
        H_ajustee = H * np.exp(-beta * sigma * np.sqrt(dt))
    else:
        raise ValueError("direction must be 'down' or 'up'")
    return H_ajustee


def di_put_continuous(paths: np.ndarray,
                      K: float,
                      H: float,
                      r: float,
                      T: float,
                      sigma: float) -> tuple[float, float]:
    """DI put, monitoring discret corrige vers un equivalent continu (BGK).

    Params
    ------
    paths : (N, n_steps+1) trajectoires, meme convention que di_put.
    K, H, r, T : strike, barriere CIBLE (continue), taux, maturite.
    sigma : volatilite de l'actif simule -- necessaire a la correction, pas
        au payoff lui-meme.

    Returns
    -------
    (price, half_width), meme convention que di_put.

    Garanties attendues
    --------------------
    * n_steps croissant (a sigma, K, H, r, T fixes) : l'ecart entre
      di_put_continuous(paths) et di_put(paths) RETRECIT -- la correction
      n'a plus de raison d'etre quand le monitoring devient fin.
    * Convergence : |di_put_continuous(n_steps) - prix_quasi_continu| decroit
      plus vite que |di_put(n_steps) - prix_quasi_continu|, ou
      prix_quasi_continu est mesure a tres grand n_steps (CRN, meme tirages
      de base) -- c'est la validation independante de la quete, dans le
      meme esprit que "doubler n_steps doit diviser le biais par deux" note
      en Acte III (pas de formule fermee a monitoring continu ici, cf. la
      remarque Reiner-Rubinstein en Acte II).
    """
    n_steps = paths.shape[1] - 1
    H_ajustee = bgk_barrier_correction(H, sigma, T, n_steps, direction="down")
    return di_put(paths, K, H_ajustee, r, T)
