"""Closed-form Black-Scholes prices and Greeks.

Everything is priced at t = 0 under the risk-neutral measure, with a continuous
dividend yield q.

Parameter convention, shared by the whole library:
    S0     spot at t = 0
    K      strike
    sigma  annualised volatility
    r      risk-free rate, continuously compounded
    T      maturity, in years
    q      continuous dividend yield (default 0.0)
"""

import numpy as np

from scipy.stats import norm


def call_bs(S0,K,sigma,r,T,q=0.0):
    """European call price.

    Returns
    -------
    float : discounted price.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) -np.exp(-r*(T))*K*norm.cdf(d2)


def put_bs(S0,K,sigma,r,T,q=0.0):
    """European put price, derived from call-put parity.

    The put being *defined* by the parity relation, checking parity against this
    function is not an independent test of `call_bs`. The dividend identity
    C(S0, q) == C(S0*exp(-qT), 0) is, and is what tests/test_pricers.py uses.

    Returns
    -------
    float : discounted price.
    """
    return call_bs(S0,K,sigma,r,T,q)+K*np.exp(-r*T) - S0*np.exp(-q*T)


def delta(S0,h,K,sigma,r,T,q=0.0):
    """Call delta, by central finite difference on the closed-form price.

    Params
    ------
    h : spot bump. Note the argument position: h comes second, before K.

    Returns
    -------
    float : (C(S0+h) - C(S0-h)) / (2h), an O(h^2) approximation of
        exp(-qT) * N(d1).
    """
    return (call_bs(S0+h,K,sigma,r,T,q) - call_bs(S0-h,K,sigma,r,T,q))/(2*h)


def vega(S0,K,sigma,r,T,q=0.0):
    """Call vega, dC/dsigma, in price units per unit of volatility.

    Returns
    -------
    float : collapses towards zero away from the money, which is what bounds the
        accuracy of the Newton solver in implied_vol.py.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S0*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)


def digital_call_bs(S0,K,sigma,r,T,q=0.0):
    """Cash-or-nothing digital call, closed form: exp(-rT) * N(d2).

    Pays 1 (in cash) if S_T > K at maturity, 0 otherwise.

    Returns
    -------
    float : discounted price.

    QUÊTE 1.5 — digitales (réplication call spread) [30 XP]
    OBJECTIF : pricer une digitale cash-or-nothing par DEUX routes
        indépendantes -- une formule fermée ici, une réplication statique par
        call spread juste en dessous (`digital_call_replication`) -- et
        vérifier qu'elles convergent l'une vers l'autre.
    DÉBLOQUE : rien d'autre dans l'Acte I ; ferme la dernière dette technique
        du parcours.
    VALIDATION : `tests/test_pricers.py::test_digital_call_bs_vs_mc` --
        comparaison à une estimation Monte-Carlo de l'indicatrice
        `1{S_T > K}` actualisée, dans l'IC 95% ; et
        `test_digital_call_replication_converge_en_h2` -- l'écart entre cette
        fonction et `digital_call_replication(h)` doit décroître en `O(h^2)`
        quand `h` diminue, testé comme une PENTE (ratio de convergence), pas
        comme un seuil sur une seule valeur de `h`.
    INDICE 1 (intuition) : le payoff digital est une fonction en escalier
        (0 puis 1, saut en `K`) -- exactement la dérivée, au signe près, du
        payoff d'un call par rapport à SON STRIKE. `N(d2)` est cette dérivée,
        actualisée.
    INDICE 2 (structure) : `d2` est le MÊME `d2` que dans `call_bs`
        (`d1 - sigma*sqrt(T)`) -- ne réécris pas `d1` à la main, on a déjà
        toute la mécanique dans `call_bs`, cette fonction n'a besoin que du
        dernier morceau.
    INDICE 3 (formule) : `digital_call_bs = exp(-r*T) * N(d2)`, avec
        `d1 = (log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))` et
        `d2 = d1 - sigma*sqrt(T)` -- identiques à ceux de `call_bs`.
    PIÈGE : ne pas actualiser deux fois -- `N(d2)` est déjà une PROBABILITÉ
        (risque-neutre), pas un prix ; c'est `exp(-r*T)` qui la transforme en
        prix, une seule fois, comme partout ailleurs dans ce module.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(d2)


def digital_call_replication(S0,h,K,sigma,r,T,q=0.0):
    """Digital call approximated by a tight call spread, in strike space.

    Params
    ------
    h : half-width of the spread, IN STRIKE SPACE -- note the argument
        position, mirrors `delta`'s `h` (spot bump), second positional slot,
        before `K`.

    Returns
    -------
    float : (call_bs(K-h) - call_bs(K+h)) / (2h), converging to
        `digital_call_bs` as h -> 0, error in O(h^2).

    QUÊTE 1.5 (suite) — la réplication statique
    OBJECTIF : construire le prix digital comme le construirait un desk sans
        contrat digital coté -- un call spread très serré, normalisé par sa
        largeur, plutôt qu'une formule fermée.
    VALIDATION : voir `digital_call_bs` ci-dessus.
    INDICE 1 (intuition) : être long un call K-h et court un call K+h, ça
        reproduit une rampe qui monte de 0 à 1 entre K-h et K+h -- diviser
        par la largeur `2h` normalise cette rampe à une hauteur de 1, comme le
        vrai payoff digital, et la rampe se resserre en escalier quand
        `h -> 0`.
    INDICE 2 (structure) : exactement `delta`, mais le bump est sur `K`, pas
        sur `S0` -- même différence finie centrée, même formule, un autre axe.
    INDICE 3 (formule) : `(call_bs(S0,K-h,sigma,r,T,q) - call_bs(S0,K+h,sigma,r,T,q)) / (2*h)`.
    PIÈGE : le signe -- le prix d'un call DÉCROÎT avec le strike (`dC/dK < 0`),
        donc `C(K-h) > C(K+h)`, et c'est bien `(C(K-h) - C(K+h))/(2h)`, PAS
        l'inverse, qui donne un nombre POSITIF. Deuxième piège, différent de
        celui de `delta` : ici pas de bruit Monte-Carlo, mais un `h` trop
        petit fait quand même dégénérer le calcul -- différence de deux
        `call_bs` presque égaux divisée par un `h` minuscule, arrondi flottant
        qui explose. Un `h` trop grand, à l'inverse, biaise (le spread n'est
        plus assez raide pour approcher un escalier). Le bon ordre de
        grandeur se mesure, il ne se devine pas.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return (call_bs(S0,K-h,sigma,r,T,q) - call_bs(S0,K+h,sigma,r,T,q)) / (2*h)
