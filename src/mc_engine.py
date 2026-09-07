import bs
import numpy as np

from scipy.stats import norm

#gbm
def gbm(S0,sigma,r,T, N=100_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return S0*np.exp((r-0.5*sigma**2)*(T)+sigma*np.sqrt(T)*rng.standard_normal(N))

#monte-carlo mtn pour avoir un pricer
def pricer_mc_call(S0, K, sigma, r, T, N=100_000, rng=None):
    ST = gbm(S0, sigma, r, T, N=N, rng=rng)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def pricer_mc_put(S0,K,sigma,r,T,N=100_000, rng=None):
    ST = gbm(S0,sigma,r,T, N=N, rng=rng)
    payoff= np.maximum(K-ST,0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96 * disc.std(ddof=1) / np.sqrt(N))

#antithetique functions
def gbm_antithetic(S0, sigma, r, T, N=50_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(N)
    drift = (r - 0.5*sigma**2)*T
    up   = S0*np.exp(drift + sigma*np.sqrt(T)*Z)
    down = S0*np.exp(drift - sigma*np.sqrt(T)*Z)
    return up, down

def pricer_mc_call_av(S0, K, sigma, r, T, N=50_000, rng=None):
    up,down = gbm_antithetic(S0, sigma, r, T, N=N, rng=rng)
    pair = 0.5*(np.maximum(up-K, 0.0) + np.maximum(down-K, 0.0))  # moyenne PAR PAIRE
    disc = np.exp(-r*T)*pair
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def delta_mc(S0,h,K,sigma,r,T,N=100_000,seed = 42):
    # Common random numbers : DEUX generateurs distincts, MEME graine -> les
    # deux pricings voient exactement les memes Z. Sans ca, la difference
    # up-down est dominee par le bruit MC (variance en 1/h^2) et le delta est
    # inutilisable. C'est le seul endroit ou une graine fixe est legitime :
    # elle fait partie de l'estimateur, ce n'est pas de la reproductibilite.
    up   = pricer_mc_call(S0+h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    down = pricer_mc_call(S0-h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    return (up-down)/(2*h)

def gbm_paths(S0, sigma, r, T, n_steps, N=100_000, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()
    dt = T/n_steps
    Z = rng.standard_normal((N, n_steps))
    log_incr = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_paths = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(log_incr, axis=1)],
        axis=1,
    )
    return S0*np.exp(log_paths)  


# =====================================================================
#                  ACTE II — REDUCTION DE VARIANCE
#   Convention du repo : la 2e valeur de retour est TOUJOURS une
#   demi-largeur d'IC 95% (1.96 * ecart-type / sqrt(n)), jamais un
#   stderr brut. Tout l'Acte I la respecte, l'Acte II aussi.
# =====================================================================
# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 2.1 — Estimateur à variable de contrôle        [30 XP]  ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : combiner un estimateur bruité Y avec un second estimateur X
#              dont on connaît l'espérance EXACTEMENT, pour réduire la variance.
# DÉBLOQUE   : quêtes 2.2, 2.3
# VALIDATION : pytest tests/test_variance_reduction.py -k "test_cv_"
#
# INDICE 1 (intuition)  : sur ton échantillon, X a commis une erreur que tu peux
#     MESURER (tu connais sa vraie moyenne). Si Y se trompe "dans le même sens"
#     que X, cette erreur mesurée est une information sur l'erreur de Y. Tu la
#     retranches. Le coefficient c dose combien tu en retranches.
# INDICE 2 (structure)  : construis un NOUVEAU vecteur Z, de même longueur que Y,
#     qui a la même espérance que Y mais moins de variance. Ensuite, tout
#     (moyenne, IC) se calcule sur Z et uniquement sur Z. Le c optimal minimise
#     Var(Z) : dérive une parabole en c, elle fait intervenir une covariance
#     et une variance. rho est le coefficient de corrélation usuel, et la
#     variance résiduelle vaut Var(Y)*(1-rho^2) — d'où vient tout le gain.
# INDICE 3 (formule)    : c* = Cov(Y,X)/Var(X) ; Z = Y - c*(X - EX).
#
# PIÈGE : Z = Y - c*X - EX  (parenthésage) donne un estimateur biaisé de -EX*(1-c)
#         — le test test_cv_utilise_bien_EX est écrit pour ça.
# PIÈGE : rho = cov/(sd_Y*sd_X). Écrit cov/sd_Y*sd_X, Python lit (cov/sd_Y)*sd_X
#         et rho sort de [-1, 1] dès que les échelles diffèrent.
# PIÈGE : la demi-largeur se calcule sur Z, PAS sur Y. Si tu la calcules sur Y,
#         tout marche "sauf" que le gain de variance est invisible.
# PIÈGE : n = len(Y). Ne va pas chercher un N global du module.
def control_variate(Y: np.ndarray,
                    X: np.ndarray,
                    EX: float,
                    c: float | None = None) -> tuple[float, float, float, float]:
    """Estimateur à variable de contrôle, agnostique à la finance.

    Params
    ------
    Y  : (n,) échantillon dont on veut estimer l'espérance.
    X  : (n,) échantillon de contrôle, apparié à Y (X[i] et Y[i] viennent du
         MÊME tirage aléatoire — sinon la corrélation est nulle et tout le
         dispositif ne sert à rien).
    EX : espérance EXACTE de X (valeur analytique, pas une estimation MC).
    c  : coefficient imposé. Si None, il est estimé sur l'échantillon.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat) : 4 floats Python.
        estimate   : estimateur à variable de contrôle de E[Y].
        half_width : demi-largeur de l'IC 95% de CET estimateur.
        c_hat      : le c utilisé (celui estimé, ou celui imposé si c donné).
        rho_hat    : corrélation empirique entre Y et X, dans [-1, 1].

    Garanties attendues
    -------------------
    * c = 0 doit reproduire EXACTEMENT le Monte-Carlo brut sur Y (moyenne et
      demi-largeur) — c'est le cas de base qui prouve que rien n'est cassé.
    * Y = X et EX exact doivent redonner EX à la précision machine, avec une
      demi-largeur numériquement nulle (quête 2.2).
    * rho_hat est invariant si on multiplie X par une constante ; c_hat, lui,
      est divisé par cette constante.
    """
    n = len(Y)
    if c is None:
        c_hat = float((np.cov(Y,X,ddof=1)[0,1])/(np.var(X,ddof=1)))
    else:
        c_hat=c
    Z = Y - c_hat*(X - EX)
    estimate = float(Z.mean())
    half_width = float(1.96*Z.std(ddof=1)/np.sqrt(n))
    rho_hat = float(np.corrcoef(X,Y)[0,1])


    return estimate,half_width,c_hat,rho_hat
    


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 2.4a — Trajectoires antithétiques                [20 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : version multi-pas de gbm_antithetic — deux jeux de trajectoires
#              construits sur le MÊME tirage gaussien, l'un avec +Z, l'autre -Z.
# DÉBLOQUE   : quête 2.4b
# VALIDATION : pytest tests/test_variance_reduction.py::test_gbm_paths_antithetic_partage_Z
#
# INDICE 1 (intuition)  : gbm_antithetic (ligne ~26) fait déjà exactement ça en
#     un pas. Ici le tirage est une MATRICE (N, n_steps) au lieu d'un vecteur.
# INDICE 2 (structure)  : un seul appel à rng.standard_normal, deux exponentielles.
#     Le reste est copié de gbm_paths : même drift, même cumsum, même colonne de
#     zéros en tête. Ne retire PAS le rng du tirage entre les deux (sinon les
#     paires ne sont plus appariées).
# INDICE 3 (formule)    : log(S_up) et log(S_down) ne diffèrent que par le signe
#     du terme en sigma ; leur somme est déterministe.
#
# PIÈGE : deux appels séparés à standard_normal = deux tirages indépendants,
#         zéro antithétique. Le test le détecte à 1e-12.
# PIÈGE : N ici est le nombre de PAIRES, donc 2N trajectoires au total. C'est ce
#         que fait déjà gbm_antithetic : reste cohérent.
def gbm_paths_antithetic(S0: float,
                         sigma: float,
                         r: float,
                         T: float,
                         n_steps: int,
                         N: int = 50_000,
                         rng: np.random.Generator | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Paires antithétiques de trajectoires GBM.

    Params
    ------
    N : nombre de PAIRES (donc 2N trajectoires simulées).

    Returns
    -------
    (up, down) : deux arrays (N, n_steps+1), up[i] et down[i] construits sur le
        même tirage gaussien au signe près. up[:, 0] == down[:, 0] == S0 exactement.
    """
    
    if rng is None:
        rng = np.random.default_rng()
    dt = T/n_steps
    Z = rng.standard_normal((N, n_steps))
    log_incr = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_incr2 = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-Z)
    log_paths = np.concatenate([np.zeros((N, 1)), np.cumsum(log_incr, axis=1)],axis=1,)
    log_paths2= np.concatenate([np.zeros((N, 1)), np.cumsum(log_incr2, axis=1)],axis=1,)
    return S0*np.exp(log_paths),S0*np.exp(log_paths2)
    


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 2.4b — Antithétiques × variable de contrôle     [20 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : combiner les deux techniques dans le bon ORDRE.
# DÉBLOQUE   : quête 2.5
# VALIDATION : pytest tests/test_variance_reduction.py -k antithetic
#
# INDICE 1 (intuition)  : les 2N tirages ne sont pas 2N observations
#     indépendantes — ils sont appariés par construction. Un IC calculé sur 2N
#     points suppose l'indépendance : il serait faux (trop optimiste).
# INDICE 2 (structure)  : l'ordre des opérations n'est pas négociable. Tu formes
#     d'abord les paires, ce qui te donne un échantillon de N observations
#     i.i.d., PUIS tu appliques le contrôle sur ces N observations. Une fois les
#     paires formées, tu n'as plus rien à écrire : control_variate fait le reste.
# INDICE 3 (formule)    : Y_pair = (Y_up + Y_down)/2, idem pour X. E[X_pair] = EX
#     (la moyenne d'une paire a la même espérance que chaque membre).
#
# PIÈGE : appliquer le contrôle sur les 2N payoffs concaténés puis "corriger"
#         l'IC après coup. Non : les paires D'ABORD.
# PIÈGE : diviser par 2N au lieu de N dans la demi-largeur → facteur sqrt(2)
#         d'erreur. test_cv_antithetic_N_est_le_nombre_de_paires est écrit pour ça.
# PIÈGE : EX ne change PAS quand on passe aux paires. Ne le divise pas par 2.
def control_variate_antithetic(Y_up: np.ndarray,
                               Y_down: np.ndarray,
                               X_up: np.ndarray,
                               X_down: np.ndarray,
                               EX: float,
                               c: float | None = None
                               ) -> tuple[float, float, float, float]:
    """Variable de contrôle appliquée à un échantillon antithétique.

    Params
    ------
    Y_up, Y_down : (N,) payoffs de la quantité d'intérêt sur les deux branches
        d'une même paire (Y_up[i] et Y_down[i] partagent le tirage).
    X_up, X_down : (N,) payoffs de la variable de contrôle, mêmes paires.
    EX : espérance exacte de la variable de contrôle (inchangée par le pairage).
    c  : coefficient imposé, ou None pour l'estimer.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat), même convention que control_variate.
    La demi-largeur porte sur N observations (le nombre de PAIRES), pas 2N.
    """
    Y_pair = (Y_up +Y_down)/2
    X_pair = (X_up +X_down)/2
    (estimate, half_width, c_hat, rho_hat)= control_variate(Y_pair,X_pair,EX,c)

    return (estimate, half_width, c_hat, rho_hat)
    


# ╔═══════════════════════════════════════════════════════════════╗
# ║ QUÊTE 2.5 — c figé sur run pilote                     [20 XP] ║
# ╚═══════════════════════════════════════════════════════════════╝
# OBJECTIF   : estimer c sur un petit échantillon SÉPARÉ, pour que l'estimateur
#              final soit rigoureusement sans biais.
# DÉBLOQUE   : BOSS 2
# VALIDATION : pytest tests/test_variance_reduction.py -k pilot
#
# INDICE 1 (intuition)  : dans la quête 2.1, c_hat est calculé sur les MÊMES
#     tirages que l'estimateur. c_hat et (X - EX) ne sont donc pas indépendants,
#     et E[c_hat * (X̄ - EX)] != 0 : il reste un biais. Il est en O(1/N), invisible
#     à 100 000 chemins, mais il est là — et c'est une question d'entretien.
# INDICE 2 (structure)  : le remède est trivial une fois le problème compris :
#     rends c indépendant de l'échantillon final. Un run court, jeté après usage,
#     dont on ne garde QUE le nombre c. Ensuite tu passes ce c à control_variate.
# INDICE 3 (formule)    : même c* = Cov(Y,X)/Var(X), calculé sur l'échantillon
#     pilote. Aucune formule nouvelle.
#
# PIÈGE : réutiliser les chemins pilotes dans le run final annule tout l'intérêt.
#         Deux rng, ou un seul rng consommé séquentiellement — jamais deux
#         default_rng(42) (ce serait le même échantillon deux fois : c'est utile
#         pour delta_mc en CRN, c'est une faute ici).
# PIÈGE : pilot_c ne connaît ni EX ni l'actualisation. C'est un ratio de moments,
#         il ne renvoie qu'un scalaire.
def pilot_c(Y_pilot: np.ndarray, X_pilot: np.ndarray) -> float:
    """Coefficient de contrôle estimé sur un échantillon pilote indépendant.

    Params
    ------
    Y_pilot, X_pilot : (n_pilot,) observations appariées d'un run court
        (typiquement 10 000 chemins), tirées indépendamment du run final.

    Returns
    -------
    c : float. Destiné à être passé tel quel en argument `c` de control_variate
        sur l'échantillon final, ce qui rend l'estimateur exactement sans biais.
    """
    c = float(np.cov(Y_pilot,X_pilot,ddof=1)[0,1]/np.var(X_pilot,ddof=1))
    return c


if __name__ == "__main__":
    paths = gbm_paths(100, 0.2, 0.05, 1.0, n_steps=50,N=100_000)
    ST=paths[:,-1]
    payoff = np.maximum(ST-100,0)
    prix = np.exp(-0.05*1)*payoff.mean()
    ic   = 1.96 * (np.exp(-0.05)*payoff).std(ddof=1)/np.sqrt(len(ST))
    print(prix, "+/-", ic)
