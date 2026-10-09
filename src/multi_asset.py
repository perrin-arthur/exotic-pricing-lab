"""ACTE V -- MULTI-ACTIF ET PRODUITS STRUCTURES.

Debloque par l'Acte IV (clos a 340/340). Le point d'arrivee annonce en tete de
CLAUDE.md : le BRC worst-of sous Heston multi-actif. Ce module pose la brique
de depart -- simuler des sous-jacents CORRELES -- sur laquelle tout le reste
de l'acte (worst-of, monitoring discret, BRC, autocall, sensibilites) se
pose. Une erreur ici (une correlation mal reconstruite, une racine qui plante
sur une matrice non definie positive) devient indetectable une fois melangee
au payoff worst-of.

Convention du repo : la 2e valeur de retour d'un estimateur est TOUJOURS une
demi-largeur d'IC 95% (1.96*sd/sqrt(n)). Les fonctions de ce module qui
renvoient des trajectoires (pas des prix) n'ont pas cette contrainte -- meme
convention que gbm_paths / heston_paths.

Convention de shape retenue pour tout l'acte : un tableau de trajectoires
multi-actifs a la forme (N, n_assets, n_steps+1) -- l'axe des actifs entre les
chemins et le temps. paths[:, i, :] est alors exactement ce que gbm_paths
aurait renvoye pour l'actif i seul ; paths.min(axis=2) donne le minimum
temporel par actif, paths.min(axis=(1, 2)) le minimum sur actifs ET temps (la
brique dont 5.2, le worst-of DI put, aura besoin).
"""

import numpy as np
import barriers
from bs import vega


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.1 — Cholesky et matrices non définies positives  [40 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : simuler N trajectoires de n_assets GBM correles, sans jamais
#              planter sur une matrice de correlation cible non definie
#              positive (rho12=0.9, rho13=0.9, rho23=-0.9 : incoherente, et
#              pourtant c'est exactement le genre de chiffres qu'un desk tape
#              a la main).
# DÉBLOQUE   : 5.2 (worst-of DI put)
# VALIDATION : pytest tests/test_multi_asset.py -k "5_1 or repair or correlated"
#
# INDICE 1 (intuition)  : tirer n_assets gaussiennes independantes par pas de
#     temps ne donne PAS la correlation voulue entre actifs -- il faut les
#     MELANGER. Cholesky est la "racine carree" d'une matrice : si L L^T =
#     rho, alors L @ Z (Z independant, covariance identite) a exactement rho
#     comme matrice de covariance. Mais Cholesky exige rho definie positive
#     (au sens strict) ; une matrice de correlations "tapee a la main" ne
#     l'est pas forcement, et numpy leve LinAlgError -- ou pire, passe avec
#     des nan silencieux selon la methode.
# INDICE 2 (structure)  : deux fonctions. Un reparateur qui prend n'importe
#     quelle matrice symetrique a diagonale unite et renvoie la matrice de
#     correlation valide la plus proche (diagonalisation, on plafonne les
#     valeurs propres negatives, on reconstruit, on RENORMALISE la diagonale
#     -- plafonner seul casse la diagonale=1). Un simulateur qui reutilise
#     gbm_paths comme modele : meme vectorisation par cumsum en log, sauf que
#     l'increment gaussien de chaque pas est maintenant un VECTEUR de taille
#     n_assets, transforme par L avant d'etre mis a l'echelle par sigma.
# INDICE 3 (formule)    :
#     Reparation (rho = Q diag(lambda) Q^T, decomposition symetrique) :
#         lambda_clip = np.maximum(lambda, tol)
#         rho_tilde   = Q @ diag(lambda_clip) @ Q.T
#         D           = diag(1/sqrt(diag(rho_tilde)))
#         rho_repaired = D @ rho_tilde @ D          # diagonale = 1 exactement
#     Cholesky : L = np.linalg.cholesky(rho_repaired), L L^T = rho_repaired.
#     Par pas de temps : Z ~ N(0, I_{n_assets}), dW = Z @ L.T (dW_i a
#     variance rho_repaired[i,i] = 1, et corr(dW_i, dW_j) = rho_repaired[i,j]).
#     log S_i(t+dt) = log S_i(t) + (r - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*dW_i
#
# PIÈGE : plafonner les valeurs propres SANS renormaliser la diagonale -- tu
#         obtiens une matrice semi-definie positive, mais ce n'est plus une
#         matrice de CORRELATION (diagonale != 1), et la variance simulee de
#         chaque actif devient fausse.
# PIÈGE : tol = 0 exactement au lieu d'un petit epsilon positif -- Cholesky
#         peut echouer sur un pivot nul (matrice seulement semi-definie).
# PIÈGE : broadcasting. paths a la forme (N, n_assets, n_steps+1) mais sigma
#         est (n_assets,) et dW issu de "Z @ L.T" est (N, n_steps, n_assets)
#         si tu vectorises le tirage d'un coup -- verifie sur QUEL axe tu
#         multiplies avant de transposer/reshaper, une erreur d'axe donne un
#         resultat qui tourne (pas de crash, juste des correlations fausses).
# PIÈGE : un seul appel a standard_normal pour TOUS les pas et TOUS les
#         actifs (meme faute qu'en 2.4a et 3.1 si tu boucles dessus) -- tire
#         Z une fois, forme (N, n_steps, n_assets), pas un appel par pas.
# PIÈGE : le cas n_assets=1 doit redonner un GBM simple -- si ta reparation
#         ou ta multiplication par L change quoi que ce soit quand rho=[[1]],
#         c'est que l'un des deux plante silencieusement dans le cas
#         degenere (L=[[1]] doit etre un no-op).


def repair_correlation_clip(rho: np.ndarray, tol: float =6.6e-16) -> np.ndarray:
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
    # Calculer les valeurs propres et les vecteurs propres de rho
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    #clippping des valeurs propres pour garantir la positivité
    eigenvalues_clipped = np.maximum(eigenvalues, tol)
    # Reconstruction de la matrice avec les valeurs propres modifiées
    rho_hat = np.dot(eigenvectors, np.dot(np.diag(eigenvalues_clipped), eigenvectors.T))
    # Normalisation de la diagonale pour obtenir une matrice de corrélation
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
    # Tirage gaussien pour tous les pas et tous les actifs
    Z = rng.standard_normal(size=(N, n_steps, n_assets))
    # Mélange des tirages pour obtenir les incréments corrélés
    dW = np.dot(Z, L.T) * np.sqrt(dt)  # (N, n_steps, n_assets)
    # Initialisation des trajectoires
    paths = np.zeros((N, n_assets, n_steps + 1))
    paths[:, :, 0] = S0
    # Simulation des trajectoires par cumul des incréments
    for i in range(n_assets):
        paths[:, i, 1:] = S0[i] * np.exp(np.cumsum((r - 0.5 * sigma[i] ** 2) * dt + sigma[i] * dW[:, :, i], axis=1))
    return paths


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.2 — Worst-of DI put et cas dégénérés              [50 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : etendre le DI put mono-actif (1.7) au cas worst-of : la pire
#              performance parmi les actifs pilote a la fois le declenchement
#              de la barriere (dans le temps) et le payoff a maturite (entre
#              actifs).
# DÉBLOQUE   : 5.3 (monitoring discret vs continu), 5.4 (BRC coupon/put)
# VALIDATION : pytest tests/test_multi_asset.py -k "worst_of"
#
# Convention actee avec Arthur (comme 5.1) : K et H sont des PERFORMANCES,
# pas des niveaux absolus -- K=1.0 est ATM, H=0.6 une barriere a 60% du
# depart. C'est la convention de cotation standard d'un term sheet autocall/
# BRC (le desk ne cote jamais un worst-of en niveau brut, un actif a 80 et un
# a 100 ne sont pas comparables), et c'est ce qui permet au cas degenere
# n_assets=1 de retomber EXACTEMENT sur barriers.di_put.
#
# INDICE 1 (intuition)  : un "worst-of" ne regarde jamais le prix brut d'un
#     actif. On ramene chaque actif a sa propre PERFORMANCE S_i(t)/S_i(0)
#     (qui demarre a 1 pour tous, quel que soit le spot de depart), et c'est
#     sur cette performance commune qu'on prend un minimum -- deux minimums
#     distincts en realite : dans le TEMPS (la barriere peut se declencher
#     n'importe quand) et entre ACTIFS (le payoff final ne regarde que le pire
#     a maturite, pas une moyenne).
# INDICE 2 (structure)  : deux fonctions. Une qui reduit (N, n_assets,
#     n_steps+1) a (N, n_steps+1) -- chaque actif divise par son PROPRE
#     depart, puis minimum sur l'axe des actifs. C'est la brique que 5.3, 5.4
#     et 5.5 reutiliseront toutes telle quelle, ne pas la fusionner avec le
#     pricer. Une deuxieme qui en fait un pricer : la barriere se teste sur le
#     minimum DANS LE TEMPS de cette performance (comme 1.7, mais sur la
#     performance plutot que le prix), le payoff ne lit que la valeur
#     terminale de cette meme performance.
# INDICE 3 (formule)    :
#     perf[n, t]   = min_i  paths[n, i, t] / paths[n, i, 0]
#     knock_in[n]  = perf[n, :].min() < H                      # dans le temps
#     payoff[n]    = max(K - perf[n, -1], 0) * knock_in[n]      # entre actifs
#     price        = mean(exp(-r*T) * payoff)
#     half_width   = 1.96 * std(ddof=1) / sqrt(N)               # convention repo
#
# PIÈGE : diviser par `paths[:, :, 0]` sans reflechir a l'axe -- selon comment
#         tu broadcastes, tu peux diviser CHAQUE actif par le S0 d'un SEUL
#         actif au lieu du sien. Le test n_assets=1 passe quand meme (un seul
#         actif, l'erreur n'a nulle part ou se cacher) ; le test rho=1 a
#         plusieurs actifs, lui, echoue silencieusement -- sans lever
#         d'exception, juste un mauvais chiffre.
# PIÈGE : lire S0 comme `paths[0, :, 0]` (un vecteur) la ou le cas n_assets=1
#         attend un scalaire -- broadcasting silencieux, pas de crash, resultat
#         faux.
# PIÈGE : K et H sont des PERFORMANCES (proches de 1), pas des niveaux comme
#         dans barriers.py -- copier-coller une intuition "H doit etre sous
#         S0" de l'Acte I ne veut plus rien dire ici.
# PIÈGE : minimum sur l'axe des actifs PUIS sur le temps donne ici le meme
#         resultat que l'inverse (le min est associatif), mais seulement parce
#         qu'on prend LE MEME minimum aux deux endroits -- ne pas en deduire
#         que l'ordre des axes est indifferent en general.


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
    #définition de r,T,H,K
    r = 0.05
    T = 1.0
    H = 0.6
    K = 1.0
    N, n_assets, n_steps_plus_1 = paths.shape
    perf = np.zeros((N, n_steps_plus_1))
    knock_in = np.zeros(N)
    payoff = np.zeros(N)
    #vectorisation du calcul de la performance, du knock-in et du payoff
    perf = np.min(paths / paths[:, :, 0][:, :, np.newaxis], axis=1)
    knock_in = np.min(perf, axis=1) < H
    payoff = np.maximum(K - perf[:, -1], 0) * knock_in
    price = np.mean(np.exp(-r*T) * payoff)
    half_width  = 1.96 * np.std(payoff,ddof=1) / np.sqrt(N)
    return perf


def worst_of_di_put(paths: np.ndarray,
                    K: float,
                    H: float,
                    r: float,
                    T: float) -> tuple[float, float]:
    """Prix Monte-Carlo d'un put down-and-in worst-of, sur la performance.

    Convention de declenchement (actee avec Arthur) : minimum sur les actifs
    ET le temps -- n'importe quel actif qui touche H a n'importe quelle date
    declenche, meme si ce n'est pas lui qui finit le pire a maturite.

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
    * n_assets=1 : egal a `barriers.di_put` applique a `paths[:, 0, :]` avec
      un strike et une barriere remis a l'echelle du prix (K*S0, H*S0) -- a
      la precision machine, memes tirages.
    * rho=1 avec des sigma identiques entre actifs : `worst_of_performance`
      est alors IDENTIQUE, trajectoire par trajectoire, a la performance de
      n'importe quel actif pris seul (aucun actif n'est jamais pire qu'un
      autre) -- le prix se reduit exactement a un DI put mono-actif sur
      cette performance (depart a 1).
    """
    perf = worst_of_performance(paths)
    knock_in = perf.min(axis=1) < H
    payoff = np.maximum(K - perf[:, -1], 0) * knock_in
    price = np.mean(np.exp(-r*T) * payoff)
    half_width = 1.96 * np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price, half_width


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.3 — BGK applique au worst-of (heuristique assumee)         ║
# ╚══════════════════════════════════════════════════════════════════╝
# Pas de XP separe ici -- le gros de la quete 5.3 (les 40 XP) est dans
# barriers.bgk_barrier_correction / di_put_continuous, qui ferme la dette
# ouverte depuis 1.7 sur le cas MONO-actif. Cette fonction reutilise la
# meme correction, actif par actif, mais ATTENTION : worst_of_performance
# n'est PAS log-normale (c'est un minimum de GBM correles), la formule BGK
# n'a donc PAS de justification theorique exacte ici -- seulement une
# heuristique (chaque actif corrige individuellement, barriere commune
# appliquee ensuite sur la performance du pire). A documenter comme une
# approximation dans PROGRESS.md, pas comme un resultat rigoureux.
#
# VALIDATION : pytest tests/test_multi_asset.py -k "continuous"
#   * sigma uniformement nul -- cas degenere, aucune correction a appliquer.
#   * rho proche de 1 (pas exactement 1, cf. le test retire en 5.2 --
#     Cholesky devient instable pile a rho=1) : le prix doit tomber DANS
#     L'IC de barriers.di_put_continuous applique a la performance commune.
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
    K, H : strike/barriere en PERFORMANCE (convention 5.2).
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
    n_assets = paths.shape[1]
    n_steps = paths.shape[2] - 1
    perf = worst_of_performance(paths)
    H_ajustee = barriers.bgk_barrier_correction(H, sigma[0], T, n_steps, direction="down")
    knock_in = perf.min(axis=1) < H_ajustee
    payoff = np.maximum(K - perf[:, -1], 0) * knock_in
    price = np.mean(np.exp(-r*T) * payoff)
    half_width = 1.96 * np.exp(-r*T) * np.std(np.maximum(K - perf[:, -1], 0) * (perf.min(axis=1) < H), ddof=1) / np.sqrt(len(perf))
    return price,half_width


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.4 — BRC : decomposition coupon / put worst-of vendu  [40 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : un Barrier Reverse Convertible, c'est un zero-coupon (nominal
#              + coupon) MOINS un put down-and-in worst-of vendu par
#              l'investisseur a la banque. Pas de nouvelle brique Monte-
#              Carlo -- juste recombiner ce qui existe deja (5.2) avec une
#              actualisation simple.
# DÉBLOQUE   : 5.5 (autocall), BOSS 5
# VALIDATION : pytest tests/test_multi_asset.py -k "brc"
#
# INDICE 1 (intuition)  : au pire (knock-in puis performance sous K a
#     maturite), l'investisseur ne recupere PAS son nominal entier -- il
#     recupere la performance du pire actif. C'est exactement l'inverse du
#     payoff d'un put : l'investisseur a implicitement VENDU un put worst-of
#     a la banque, et recoit en echange un coupon plus genereux qu'un taux
#     sans risque. D'ou : valeur du produit = obligation (nominal + coupon,
#     actualises) MOINS le prix du put vendu.
# INDICE 2 (structure)  : une seule fonction, qui ne fait QUE combiner
#     worst_of_di_put (5.2, reutilisee telle quelle, ne pas la reecrire) et
#     deux actualisations. La decomposition est EXACTE (pas une
#     approximation) : verifie-la a la main sur un chemin avant de coder.
# INDICE 3 (formule)    :
#     put_price, put_hw = worst_of_di_put(paths, 1.0, H, r, T)
#     price = C * exp(-r*T) + 1.0 * exp(-r*T) - put_price
#     half_width = put_hw          # seul terme stochastique de la somme
#
# PIÈGE : coupon ET nominal actualises au MEME exp(-r*T) ici (paiement
#         unique a T -- les dates intermediaires arrivent en 5.5).
# PIÈGE : K doit rester 1.0 (ATM, nominal entier en cas de non-declenchement)
#         -- un K different n'est plus un BRC standard.
# PIÈGE : half_width de la somme n'est PAS une somme de half_width -- ici la
#         seule partie aleatoire est le put (coupon et nominal actualises
#         sont deterministes), donc half_width_BRC = half_width_put
#         EXACTEMENT, pas 2x ni une combinaison quadratique.


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
    H : barriere de declenchement, en PERFORMANCE (convention 5.2).
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
    put_price, put_hw = worst_of_di_put(paths, 1.0, H, r, T)
    price = C * np.exp(-r*T) + 1.0 * np.exp(-r*T) - worst_of_di_put(paths, 1.0, H, r, T)[0]
    half_width = 1.96 * np.exp(-r*T) * np.std(np.maximum(1.0 - worst_of_performance(paths)[:, -1], 0) * (worst_of_performance(paths).min(axis=1) < H), ddof=1) / np.sqrt(len(paths))
    return price,half_width


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.5 — Autocall / Phoenix worst-of                  [50 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : generaliser le BRC (5.4, une seule date) a plusieurs dates de
#              rappel : des que la performance du pire actif depasse un
#              niveau de rappel a une date d'observation, le produit
#              rembourse NOMINAL + coupons cumules par anticipation, et
#              s'arrete la. S'il n'est jamais rappele, le payoff a maturite
#              retombe sur le BRC (5.4).
# DÉBLOQUE   : BOSS 5
# VALIDATION : pytest tests/test_multi_asset.py -k "autocall"
#
# INDICE 1 (intuition)  : chaque trajectoire a UNE SEULE date de rappel
#     effective (la premiere ou perf >= niveau de rappel), ou aucune. Ce
#     n'est pas une somme sur toutes les dates futures -- des qu'un chemin
#     est rappele, les dates suivantes n'existent plus pour lui. Premiere
#     occurrence vraie d'un booleen, pas une somme ni une moyenne.
# INDICE 2 (structure)  : pour chaque trajectoire, trouve l'INDICE (dans
#     obs_steps) de la premiere date ou perf >= niveau de rappel -- ou
#     "jamais" si aucune. Les chemins rappeles sont payes a LEUR propre
#     date (actualises chacun a leur propre T_rappel, pas tous a la meme
#     date) ; les chemins jamais rappeles tombent sur le payoff BRC de 5.4,
#     a maturite, barriere H monitoree sur TOUTE la trajectoire (pas
#     seulement aux dates d'observation -- le knock-in et le rappel sont
#     deux mecanismes independants).
# INDICE 3 (formule)    : pour chaque n,
#     i_rappel[n] = premier i tel que perf[n, obs_steps[i]] >= niveau_rappel
#                   (ou -1 si jamais)
#     si rappele (i_rappel[n] != -1) :
#         t_rappel = obs_steps[i_rappel[n]] * (T / n_steps)
#         payoff[n] = (1 + C*(i_rappel[n]+1)) * exp(-r * t_rappel)
#     sinon : payoff[n] = payoff BRC par-chemin (comme 5.4), actualise a T
#     price = mean(payoff) ; half_width = 1.96*std(payoff,ddof=1)/sqrt(N)
#
# PIÈGE : le KNOCK-IN (barriere H) continue de s'observer MEME apres une
#         date d'autocall manquee -- seul un RAPPEL arrete le monitoring, un
#         knock-in ne l'arrete jamais.
# PIÈGE : actualiser tous les chemins rappeles a la MEME date (ex: toujours
#         T) au lieu de LEUR date de rappel -- biaise le prix.
# PIÈGE : "premiere date >= niveau" implemente avec un argmax naif sur un
#         tableau de booleens ou AUCUNE date ne depasse le niveau -- argmax
#         renvoie alors l'indice 0 silencieusement (pas d'erreur), le meme
#         piege qu'en 1.8 (Manaster-Koenig) sous une autre forme.


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
    # Créer un masque booléen indiquant où la performance dépasse le niveau de rappel
    mask = perf_obs >= autocall_level
    # Utilisation de np.where pour trouver les indices de la première occurrence
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
    n_assets =len(paths[1])
    
    perf = worst_of_performance(paths)
    first_occurrence = recall_index(perf[:, obs_steps], autocall_level)
    payoff = np.zeros(len(paths))
    for n in range(len(paths)):
        #multiplier la perte par knock_in(0 ou1) pour ne pas payer le put si barriere jamais touchee
        if first_occurrence[n] != -1:
            t_rappel = obs_steps[first_occurrence[n]] * (T / (paths.shape[2] - 1))
            payoff[n] = (1 + C * (first_occurrence[n] + 1)) * np.exp(-r * t_rappel)
        else:
          knock_in = perf[n, :].min() < H
          perte = np.maximum(1.0 - perf[n, -1], 0) * knock_in
          payoff[n] = (1.0 + C * len(obs_steps) - perte) * np.exp(-r * T)

    price = np.mean(payoff)
    half_width = 1.96 * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price,half_width


# ╔══════════════════════════════════════════════════════════════════╗
# ║ QUÊTE 5.6 — Sensibilites : delta par actif, vega, rho       [50 XP] ║
# ╚══════════════════════════════════════════════════════════════════╝
# OBJECTIF   : meme technique qu'en 1.4 (delta_mc, bump + common random
#              numbers) mais generalisee a n_assets -- bumper UN SEUL actif
#              a la fois, les autres et les tirages inchanges, pour isoler
#              sa sensibilite propre.
# DÉBLOQUE   : BOSS 5
# VALIDATION : pytest tests/test_multi_asset.py -k "delta or vega or rho_sens"
#
# INDICE 1 (intuition)  : le delta par actif repond a "si SEUL cet actif
#     bouge, combien vaut le produit en plus/moins ?" -- les tirages (le Z
#     gaussien) doivent etre IDENTIQUES entre le prix bumpe et le prix de
#     base, sinon la difference est noyee dans le bruit MC (meme piege
#     qu'en 1.4, variance en 1/h^2 sans CRN).
# INDICE 2 (structure)  : reutilise worst_of_di_put tel quel, appelle-le
#     deux fois (S0 bumpe sur l'actif i, S0 non bumpe), avec le MEME rng
#     reconstruit a partir de la MEME graine (comme delta_mc :
#     np.random.default_rng(seed) appele deux fois, pas le meme objet
#     reutilise -- un Generator avance, piege deja rencontre en 2.4a). Vega
#     et sensibilite-rho : meme principe, bump sur sigma[i] ou sur rho[i,j]
#     au lieu de S0[i].
# INDICE 3 (formule)    :
#     S0_up, S0_down = S0.copy(), S0.copy()
#     S0_up[i] += h ; S0_down[i] -= h
#     price_up   = worst_of_di_put(correlated_gbm_paths(S0_up,  ..., rng=np.random.default_rng(seed)), K, H, r, T)[0]
#     price_down = worst_of_di_put(correlated_gbm_paths(S0_down, ..., rng=np.random.default_rng(seed)), K, H, r, T)[0]
#     delta_i = (price_up - price_down) / (2*h)
#
# PIÈGE : bumper S0[i] et oublier que le PAYOFF (K, H en PERFORMANCE) ne
#         change pas de sens pour autant -- performance = S_t/S0, bumper S0
#         affecte la trajectoire ET son propre point de depart des le
#         premier pas (contrairement a un bump de strike qui ne toucherait
#         que le payoff final).
# PIÈGE : meme h pour vega que pour delta sans y penser -- vega bump sigma
#         (ordre de grandeur 0.20), delta bump S0 (ordre de grandeur 100) :
#         le MEME h numerique n'a pas le meme sens relatif dans les deux cas.
# PIÈGE : rappeler repair_correlation_clip directement ici avant de bumper
#         rho[i,j] -- correlated_gbm_paths le refait deja en interne, un
#         appel ici serait du code mort, pas un bug de resultat mais un
#         signe que tu dupliques une responsabilite.


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
    K, H : strike/barriere en performance (convention 5.2).
    h : taille du bump sur S0[i], en NIVEAU (pas en performance).
    seed : graine fixe, rejouee a l'identique pour chaque bump (CRN).

    Returns
    -------
    (n_assets,) delta_i = d(prix)/d(S0_i), un par actif.

    Garanties attendues
    --------------------
    * rho proche de 1, vols et spots identiques entre actifs : la SOMME des
      deltas converge vers le delta d'un unique actif bumpe (delta basket)
      -- mesure, pas une identite exacte (bruit MC + approximation 5.1 a
      rho=1).
    * pres de la barriere H, le delta devient instable (grande variance d'un
      seed a l'autre) -- meme phenomene que 4.5, a documenter, pas a
      "corriger" silencieusement.
    """
    perf = worst_of_performance(correlated_gbm_paths(S0, sigma, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)))
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
    perf = worst_of_performance(correlated_gbm_paths(S0, sigma, rho, r, T, n_steps, N, rng=np.random.default_rng(seed)))
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
    price_up = worst_of_di_put(correlated_gbm_paths(S0, sigma, rho+rate, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
    price_down = worst_of_di_put(correlated_gbm_paths(S0, sigma, rho - rate, r, T, n_steps, N, rng=np.random.default_rng(seed)), K, H, r, T)[0]
    return float(price_up - price_down) / (2 * h)
