"""ACTE III — HESTON MONO-ACTIF.

Debloque par le BOSS 2. Le fil : simuler correctement (3.1), se donner une
REFERENCE analytique (3.2), lire ce que le modele produit (3.3), puis rebrancher
toute la machinerie de l'Acte II dessus (BOSS 3).

Le point d'arrivee du repo est le BRC worst-of sous Heston multi-actif. Ici on
fait le mono-actif : tout ce qui suit (Cholesky multi-actif, worst-of, autocall)
se pose SUR ces briques. Une erreur de schema non detectee ici deviendra
indetectable a 3 sous-jacents.

Convention du repo : la 2e valeur de retour d'un estimateur est TOUJOURS une
demi-largeur d'IC 95% (1.96*sd/sqrt(n)).

Notation, fixee une fois pour toutes (celle de Grzelak et de Bouzoubaa) :
    v0     variance INITIALE (pas la vol : v0 = sigma0^2)
    kappa  vitesse de retour a la moyenne
    theta  variance de long terme (pas la vol non plus)
    xi     vol-of-vol (souvent note sigma ou eta ailleurs — ici xi, pour ne
           jamais le confondre avec la vol BS)
    rho    correlation entre le brownien du spot et celui de la variance
"""

import numpy as np
from scipy.integrate import quad
from implied_vol import implied_vol_call
# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 3.1 — Trajectoires Heston (Euler full truncation) [40 XP]║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : simuler conjointement (S_t, v_t) avec deux browniens correles,
#              en gardant la variance positive.
# DÉBLOQUE   : 3.2, 3.3
# VALIDATION : pytest tests/test_heston.py -k paths
#
# INDICE 1 (intuition)  : c'est gbm_paths avec deux differences. Un, la vol
#     n'est plus constante : a chaque pas tu lis la variance courante. Deux, il
#     te faut DEUX gaussiennes par pas et par chemin, correlees a rho — le spot
#     et la variance ne bougent pas independamment (rho < 0 : quand ca baisse,
#     ca s'agite ; c'est le levier, et c'est ce qui cree le skew).
# INDICE 2 (structure)  : tire une matrice (N, n_steps, 2) de gaussiennes
#     INDEPENDANTES, puis fabrique le couple correle par Cholesky 2x2 (une
#     combinaison lineaire, pas un appel a np.linalg — en dimension 2 elle
#     s'ecrit a la main). Boucle sur le temps : impossible de vectoriser comme
#     dans gbm_paths, puisque le pas t+1 depend de v_t. La boucle porte sur
#     n_steps (quelques centaines), pas sur N : chaque iteration traite les N
#     chemins d'un coup.
# INDICE 3 (formule)    :
#     Cholesky 2D : W_v = Z1 ; W_S = rho*Z1 + sqrt(1-rho^2)*Z2.
#     v_{t+dt} = v_t + kappa*(theta - v_t^+)*dt + xi*sqrt(v_t^+ * dt)*W_v
#     log S_{t+dt} = log S_t + (r - 0.5*v_t^+)*dt + sqrt(v_t^+ * dt)*W_S
#     avec v^+ = max(v, 0) — c'est ca, "full truncation".
#
# PIÈGE : oublier max(v, 0) -> sqrt d'un negatif -> nan silencieux qui se
#         propage a TOUTE la trajectoire. Euler sur un CIR passe sous zero des
#         que la condition de Feller (2*kappa*theta > xi^2) est violee, et meme
#         quand elle tient si dt est grossier.
# PIÈGE : utiliser v_{t+dt} (deja mis a jour) dans le pas du spot. Le schema est
#         EXPLICITE : les deux pas lisent le MEME v_t. Le test de correlation du
#         premier pas attrape l'inversion.
# PIÈGE : simuler S directement au lieu de log S. Tu perds la positivite exacte
#         et tu ajoutes un biais que le test de martingale voit.
# PIÈGE : deux appels separes a standard_normal pour Z1 et Z2 — meme faute qu'en
#         2.4a, mais ici elle ne casse pas la correlation entre chemins, elle
#         casse rho. Un seul tirage, deux tranches.
# PIÈGE : tronquer la variance STOCKEE au lieu de la variance UTILISEE. La
#         convention full truncation garde v_t tel quel dans le tableau (il peut
#         etre negatif) et n'applique max(v,0) que dans le drift et sous les
#         racines. Le test verifie v_min >= 0 : choisis l'autre convention
#         (stocker le max) si tu preferes, mais alors sois coherent partout.
def heston_paths(S0: float,
                 v0: float,
                 r: float,
                 T: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho: float,
                 n_steps: int,
                 N: int = 100_000,
                 rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Trajectoires (spot, variance) du modele de Heston, schema d'Euler.

    Params
    ------
    S0    : spot initial.
    v0    : VARIANCE initiale (v0 = sigma_0^2 ; pour une vol de 20%, v0 = 0.04).
    r     : taux sans risque.
    T     : maturite.
    kappa : vitesse de retour a la moyenne de la variance.
    theta : VARIANCE de long terme.
    xi    : vol-of-vol.
    rho   : correlation des deux browniens, dans [-1, 1]. Negatif sur actions.
    n_steps : nombre de pas de discretisation.
    N     : nombre de trajectoires.
    rng   : generateur injecte (jamais de graine capturee a l'interieur).

    Returns
    -------
    (S, v) : deux arrays (N, n_steps+1).
        S[:, 0] == S0 et v[:, 0] == v0 EXACTEMENT.
        v est positive ou nulle partout (full truncation).

    Garanties attendues
    -------------------
    * xi = 0 et v0 = theta  ->  v constante = theta, et S est un GBM de vol
      sqrt(theta) : le pricer de l'Acte I doit redonner le prix BS.
    * E[e^{-rT} S_T] = S0 (identite de martingale, model-free).
    * E[v_T] -> theta + (v0 - theta)*exp(-kappa*T) (moyenne exacte du CIR).
    * la correlation empirique des deux increments du PREMIER pas vaut rho.
    """
    dt = T/n_steps
    S = np.zeros((N,n_steps+1))
    v = np.zeros((N,n_steps+1))
    S[:,0] = S0
    v[:,0] = v0
    if rng is None: 
      rng = np.random.default_rng()
    Z = rng.standard_normal((N,n_steps,2))
    W_v = Z[:, :, 0]
    W_S = rho*Z[:, :, 0] + np.sqrt(1-rho**2)*Z[:, :, 1]
    v_brut = np.full(N, v0)
    for t in range(n_steps):
      v_plus = np.maximum(v_brut, 0.0)
      v_brut = v[:, t] + kappa*(theta - v_plus)*dt + xi*np.sqrt(v_plus*dt)*W_v[:,t]
      v[:, t+1] = np.maximum(v_brut, 0.0)
      incr = (r - 0.5*v_plus)*dt + np.sqrt(v_plus*dt)*W_S[:, t]
      S[:,t+1] = S[:,t]*np.exp(incr)
    return (S, v)
    


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 3.2 — Vanille semi-analytique (fonction car.)    [50 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : un prix de call europeen Heston SANS Monte-Carlo, pour servir de
#              reference externe a tout le reste de l'acte.
# DÉBLOQUE   : 3.3, BOSS 3
# VALIDATION : pytest tests/test_heston.py -k semi
#
# INDICE 1 (intuition)  : Heston n'a pas de densite fermee, mais il a une
#     fonction caracteristique fermee — la transformee de Fourier de la loi de
#     log S_T. Or un prix d'option est une integrale contre cette densite : on
#     peut donc l'ecrire comme une integrale contre la fonction caracteristique,
#     que l'on calcule numeriquement. "Semi-analytique" = formule exacte,
#     quadrature numerique.
# INDICE 2 (structure)  : deux routes, choisis-en une et sache dire pourquoi.
#     (a) Heston 1993 / Gatheral : prix = S0*P1 - K*exp(-rT)*P2, ou P1 et P2
#         sont deux integrales sur u de 0 a l'infini d'une partie reelle.
#         Quadrature : scipy.integrate.quad, ou une somme de Gauss-Legendre.
#     (b) COS (Fang-Oosterlee, lecture de Grzelak) : developpement en cosinus
#         sur un intervalle tronque [a, b] deduit des cumulants. Plus rapide,
#         plus technique, et c'est celle qu'on te demandera si l'entretien est
#         a Zurich chez quelqu'un qui a lu le meme cours que toi.
#     Dans les deux cas, isole la fonction caracteristique dans sa PROPRE
#     fonction (voir heston_cf ci-dessous) : c'est elle que tu debugueras.
# INDICE 3 (formule)    : fonction caracteristique de x_T = log S_T, dans la
#     forme dite "little trap" de Albrecher (celle qui ne saute pas de branche) :
#         d = sqrt((rho*xi*i*u - kappa)^2 + xi^2*(i*u + u^2))
#         g = (kappa - rho*xi*i*u - d) / (kappa - rho*xi*i*u + d)
#         C = r*i*u*T + (kappa*theta/xi^2)*((kappa - rho*xi*i*u - d)*T
#                       - 2*log((1 - g*exp(-d*T))/(1 - g)))
#         D = ((kappa - rho*xi*i*u - d)/xi^2)*((1 - exp(-d*T))/(1 - g*exp(-d*T)))
#         phi(u) = exp(C + D*v0 + i*u*log(S0))
#     Puis Gil-Pelaez / Heston : P_j = 1/2 + (1/pi) * integrale de la partie
#     reelle de exp(-i*u*log K)*phi_j(u)/(i*u).
#
# PIÈGE : LA COUPURE DE BRANCHE. La forme "naive" (celle de l'article de 1993,
#         avec g = (b - d)/(b + d) au lieu de (b - d)/(b + d) ci-dessus, signes
#         inverses) fait sauter le logarithme complexe d'une feuille a l'autre
#         pour T grand : le prix devient discontinu en T. C'est LE piege
#         classique, il a sa litterature ("the little Heston trap"). Si ton prix
#         est bon a T=1 et delire a T=5, c'est ca.
# PIÈGE : l'integrande est singuliere en u = 0 (division par i*u). Elle admet
#         une limite finie, mais la quadrature peut y planter : borne
#         l'integrale a partir d'un epsilon, ou utilise une regle qui evite le
#         bord.
# PIÈGE : borne superieure d'integration. "Infini" en pratique = 100 a 200 selon
#         les parametres. Trop court : prix faux de facon lisse (donc invisible).
#         Verifie que doubler la borne ne change plus le prix a 1e-10.
# PIÈGE : xi^2 au denominateur — le cas xi = 0 est une division par zero. Le
#         test degenere utilise xi = 1e-6, pas 0, et c'est deliberé.
def heston_cf(u: np.ndarray | complex,
              S0: float,
              v0: float,
              r: float,
              T: float,
              kappa: float,
              theta: float,
              xi: float,
              rho: float) -> np.ndarray | complex:
    """Fonction caracteristique de log(S_T) sous Heston : E[exp(i*u*log S_T)].

    Params
    ------
    u : point (ou vecteur de points) d'evaluation, reel.
    (les autres) : voir heston_paths.

    Returns
    -------
    phi(u), complexe (ou array complexe si u est un array).

    Garanties attendues
    -------------------
    * phi(0) = 1 exactement (c'est une fonction caracteristique).
    * |phi(u)| <= 1 pour tout u reel.
    * phi(-u) = conj(phi(u)) : la loi sous-jacente est reelle.
    * xi -> 0 et v0 = theta : phi doit tendre vers la fonction caracteristique
      d'une gaussienne de variance theta*T (le cas Black-Scholes).
    """
    b = kappa - rho*xi*1j*u
    d = np.sqrt(b**2 + (xi**2)*(1j*u + u**2))
    g = (b-d)/(b+d)

    Duration = ((b-d)/(xi**2))*((1-np.exp(-d*T))/(1-g*np.exp(-d*T)))
    
    C = (r*1j*u*T) + (kappa*theta/(xi**2))*((b-d)*T - 2*np.log((1-g*np.exp(-d*T))/(1-g)))


    phi = np.exp(1j*u*np.log(S0) + Duration*v0 + C)
    
    return phi


def heston_call(S0: float,
                K: float,
                v0: float,
                r: float,
                T: float,
                kappa: float,
                theta: float,
                xi: float,
                rho: float) -> float:
    """Prix d'un call europeen sous Heston, par integration de heston_cf.

    Returns
    -------
    prix : float Python (pas np.float64 — le BOSS 3 serialise en JSON).

    Garanties attendues
    -------------------
    * bornes model-free : max(S0 - K*exp(-rT), 0) <= C <= S0.
    * xi -> 0 avec v0 = theta : redonne call_bs(S0, K, sqrt(theta), r, T).
    * coherent avec un Monte-Carlo sur heston_paths, a l'IC pres.
    * decroissant en K, croissant en T.
    """
    
    val,erreur2 = quad(lambda u:(np.real(np.exp(-1j*u*np.log(K))*heston_cf(u,S0,v0,r,T,kappa,theta,xi,rho)/(1j*u))),1e-8,np.inf) 
    P2 = 0.5 + val/np.pi
    valo,erreur1 = quad(lambda u:(np.real(np.exp(-1j*u*np.log(K))*heston_cf(u-1j,S0,v0,r,T,kappa,theta,xi,rho)/(1j*u*S0*np.exp(r*T)))),1e-8,np.inf)
    P1 = 0.5 + valo/np.pi
    return float(S0*P1 -K*np.exp(-r*T)*P2)


def heston_put(S0: float,
               K: float,
               v0: float,
               r: float,
               T: float,
               kappa: float,
               theta: float,
               xi: float,
               rho: float) -> float:
    """Prix d'un put europeen sous Heston.

    Aucune integrale nouvelle : la parite call-put est model-free, elle ne
    depend que de l'absence d'arbitrage. C'est exactement ce que fait put_bs
    dans src/bs.py — relis-le avant d'ecrire quoi que ce soit.

    Returns
    -------
    prix : float Python.
    """

    #parité call-put
    C = heston_call(S0,K,v0,r,T,kappa,theta,xi,rho)

    P = C + K*np.exp(-r*T) - S0
    return float(P)


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 3.3 — Le smile, et ce que chaque parametre y fait [40 XP]║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : convertir des prix Heston en vols implicites BS, et savoir lire
#              la surface : qui fait la pente, qui fait la courbure.
# DÉBLOQUE   : BOSS 3
# VALIDATION : pytest tests/test_heston.py -k smile
#
# INDICE 1 (intuition)  : la vol implicite n'est pas un parametre de modele,
#     c'est une UNITE DE PRIX — le sigma qu'il faut mettre dans BS pour
#     retrouver le prix observe. Tu as deja l'inverseur (quête 1.8). Ici tu
#     l'appliques a des prix Heston : le smile qui en sort est la signature du
#     modele, et c'est sous cette forme que le marche cote.
# INDICE 2 (structure)  : une boucle sur les strikes. Pour chaque K, un prix
#     Heston (3.2), puis implied_vol_call dessus. Rien d'autre.
# INDICE 3 (formule)    : aucune formule nouvelle. Ce qu'il faut savoir DIRE :
#     rho gouverne la PENTE (rho < 0 -> skew negatif, les puts OTM cotent plus
#     cher : c'est la demande de protection, et c'est ce qui fait vivre les
#     produits structures) ; xi gouverne la COURBURE (xi = 0 -> smile plat) ;
#     kappa et theta gouvernent la structure par terme (le smile s'aplatit
#     quand T grandit, a la vitesse de la reversion).
#
# PIÈGE : implied_vol_call attend un prix de CALL. Si tu lui passes un put,
#         Newton diverge ou converge vers n'importe quoi. Passe par la parite.
# PIÈGE : loin de la monnaie, le vega s'effondre -> Newton devient instable.
#         C'est la limite du solveur de la quête 1.8, pas un bug de Heston.
#         Reste dans une plage de strikes raisonnable (70% - 130% du spot).
# PIÈGE : tracer la vol implicite contre K et non contre la log-moneyness
#         log(K/F). Contre K, deux maturites ne sont pas comparables — c'est la
#         premiere chose qu'un trader de vol te fera remarquer.
def heston_smile(S0: float,
                 strikes: np.ndarray,
                 v0: float,
                 r: float,
                 T: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho: float) -> np.ndarray:
    """Vols implicites BS des calls Heston, strike par strike.

    Params
    ------
    strikes : (m,) grille de strikes.
    (les autres) : voir heston_paths.

    Returns
    -------
    iv : (m,) vols implicites, meme ordre que `strikes`.

    Garanties attendues
    -------------------
    * xi -> 0 et v0 = theta : smile PLAT a sqrt(theta) (a 1e-4 pres).
    * rho < 0 : iv decroissante en K (skew negatif).
    * rho = 0 : smile symetrique en log-moneyness, et convexe.
    * xi plus grand a rho fixe : courbure plus forte.
    """
    iv = []
    for K in strikes:
      market_price = heston_call(S0,K,v0,r,T,kappa,theta,xi,rho)
      iv.append(implied_vol_call(S0, K, r, T, market_price, q=0.0, tol=1e-6, max_iter=100))
    return np.asarray(iv)
