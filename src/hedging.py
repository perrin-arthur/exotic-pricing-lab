"""Discrete hedging and replication P&L.

A price without a replicating portfolio is not validated: this module builds
the self-financed delta-hedge simulation that closes the loop opened by
bs.py / mc_engine.py / heston.py, and measures what breaks it (discretisation
noise, a wrong volatility, transaction costs, a barrier, an unhedged vega).

Library-wide conventions carried over: the second return value of an
estimator is a 95% confidence half-width, 1.96 * sd / sqrt(n). This module
takes already-simulated paths as input rather than drawing its own randomness,
so no function here owns an `rng` argument -- the paths passed in were built
with one, upstream, in gbm_paths / heston_paths.
"""

import numpy as np
import scipy

import bs

def portfolio_terminal_value(paths: np.ndarray,
                             deltas: np.ndarray,
                             r: float,
                             T: float,
                             V0: np.ndarray | float) -> np.ndarray:
    """Terminal value of a self-financed replicating portfolio.

    Params
    ------
    paths  : (N, n_steps+1) simulated price paths, S0 in column 0.
    deltas : (N, n_steps) shares held over [t_i, t_{i+1}); deltas[:, i] is
        decided at t_i and held until the next rebalancing date.
    r, T   : risk-free rate, maturity. dt = T / n_steps is read off
        paths.shape[1] - 1.
    V0     : initial portfolio value, (N,) or a scalar broadcastable to it --
        typically the option premium, so that a perfect hedge reproduces the
        payoff exactly.

    Returns
    -------
    (N,) terminal portfolio value V_T, one per path.

    QUÊTE 4.1 — le portefeuille auto-financé [30 XP]
    OBJECTIF : simuler le compte cash + actions d'une stratégie de réplication,
        sans supposer que `deltas` est le bon delta -- c'est une mécanique
        comptable, pas encore une question de pricing.
    DÉBLOQUE : 4.2, et par transitivité tout le reste de l'acte -- c'est le
        socle sur lequel `hedging_error` (4.2) est bâti.
    VALIDATION : `tests/test_hedging.py::test_autofinancement_deltas_arbitraires`
        -- avec des deltas TIRÉS AU HASARD (positifs, négatifs, énormes), la
        récursion d'auto-financement doit être vérifiée à 1e-12. Aucune
        référence de pricing n'intervient : c'est une identité algébrique,
        vraie pour n'importe quelle suite de deltas.
    INDICE 1 (intuition) : à chaque date de rebalancement tu achètes ou vends
        des actions ; cet argent ne vient de nulle part, il sort d'un compte
        cash qui, lui, a fructifié au taux sans risque depuis la dernière
        date. Le portefeuille (actions + cash) ne reçoit ni ne perd jamais
        d'argent "de l'extérieur" entre deux dates -- c'est ÇA,
        l'auto-financement.
    INDICE 2 (structure) : fais vivre deux quantités en parallèle sur la
        boucle des dates -- le nombre d'actions détenues (donné,
        `deltas[:, i]`) et un compte cash `B` (à calculer). `B` démarre à
        `V0 - deltas[:, 0]*S0`. Entre deux dates, `B` capitalise au taux `r`
        sur `dt`, PUIS absorbe le coût (positif ou négatif) du changement de
        position en actions à la date suivante. À la toute dernière date, il
        n'y a plus de rebalancement : la position `deltas[:, -1]` est
        conservée jusqu'à `S[:, -1]`.
    INDICE 3 (formule) : pour i = 0 .. n_steps-2,
        `B_{i+1} = B_i*exp(r*dt) - (deltas[:,i+1]-deltas[:,i])*S[:,i+1]` ;
        `V_T = deltas[:,-1]*S[:,-1] + B_{n_steps-1}*exp(r*dt)`.
    PIÈGE : `paths` a `n_steps+1` colonnes, `deltas` en a `n_steps` --
        `deltas[:, i]` est décidé À `t_i` et vaut pour `[t_i, t_{i+1})`, donc
        `S[:, i+1]` est le prix auquel ce changement de position se règle, pas
        `S[:, i]`. Un décalage d'une colonne entre `paths` et `deltas` reste
        invisible avec un delta "raisonnable" (l'erreur se noie dans le bruit
        de couverture) ; il ne saute aux yeux qu'avec des deltas aberrants,
        exactement ce que teste 4.1.
    """
    B = V0 - deltas[:, 0] * paths[:, 0]
    dt = T / (paths.shape[1] - 1)
    V_T = np.empty(paths.shape[0])
    for i in range(deltas.shape[1] - 1): 
        B = B * np.exp(r * dt) - (deltas[:, i + 1] - deltas[:, i]) * paths[:, i + 1]
    V_T = deltas[:, -1] * paths[:, -1] + B * np.exp(r * dt)
    return V_T


def bs_delta_hedge_deltas(paths: np.ndarray,
                          K: float,
                          sigma: float,
                          r: float,
                          T: float,
                          option: str = "put") -> np.ndarray:
    """Black-Scholes delta at each rebalancing date, on shrinking maturity.

    Params
    ------
    paths  : (N, n_steps+1) simulated paths.
    K, sigma, r, T : as in bs.py.
    option : "call" or "put".

    Returns
    -------
    (N, n_steps) deltas, deltas[:, i] evaluated at t_i with remaining maturity
    tau = T - t_i.

    QUÊTE 4.2 — erreur de hedging sous le bon modèle [50 XP]
    OBJECTIF : produire, pour chaque path et chaque date de rebalancement, le
        delta BS calculé sur la maturité RÉSIDUELLE -- pas la maturité
        initiale T, qui ne varierait pas d'une colonne à l'autre.
    DÉBLOQUE : `hedging_error` puis `hedging_error_stats`, plus loin 4.3, 4.4,
        4.5 (qui réutilise le principe pour le put vanille), 4.6.
    VALIDATION : `test_bs_delta_hedge_deltas_vs_formule_fermee` -- au premier
        pas (tau = T), compare deltas[:, 0] à N(d1) - 1 (put) calculé par une
        formule fermée écrite directement dans le test, indépendante de
        `bs.py`. Tolérance justifiée par l'ordre de grandeur d'une différence
        finie bien réglée (1e-3), pas 1e-12 -- ce n'est pas une identité
        algébrique.
    INDICE 1 (intuition) : le delta d'une option n'est pas un nombre fixe, il
        dépend du temps qu'il reste avant l'échéance. À la date t_i, il reste
        T - t_i avant maturité -- c'est CE temps-là qu'il faut passer à la
        formule du delta, pas T.
    INDICE 2 (structure) : `bs.delta(S0, h, K, sigma, r, T, q=0.0)` calcule un
        delta de CALL par différence finie -- attention à l'ordre des
        arguments positionnels (h avant K, piège déjà documenté dans bs.py).
        Le delta de PUT s'obtient par parité : delta_put = delta_call - 1
        (dérivée de C - P = S - K*exp(-rT) par rapport à S). Boucle (ou
        vectorise) sur les n_steps colonnes de `paths[:, :-1]`, avec à chaque
        colonne i une maturité résiduelle tau_i = T - i*dt.
    INDICE 3 (formule) : delta_put(S, tau) = N(d1(S, tau)) - 1, avec
        d1(S, tau) = (log(S/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau)).
    PIÈGE : à la DERNIÈRE date de rebalancement (i = n_steps - 1), tau = dt,
        pas 0 -- le portefeuille n'atteint tau = 0 qu'à la toute fin, une fois
        la position finale déjà figée (cf. 4.1, INDICE 3). Passer tau = 0 ici
        donnerait un delta indicateur (0 ou -1 selon K vs S), pas N(d1) - 1.
    """
    dt = T / (paths.shape[1] - 1)
    tau = T - np.arange(paths.shape[1] - 1) * dt
    if option == "put":
        return bs.delta(paths[:, :-1], 1e-4, K, sigma, r, tau) - 1
    elif option == "call":
        return  bs.delta(paths[:, :-1], 1e-4, K, sigma, r, tau)
    return np.full_like(paths[:, :-1], np.nan)  # pragma: defensive


def hedging_error(paths: np.ndarray,
                  deltas: np.ndarray,
                  K: float,
                  r: float,
                  T: float,
                  V0: float,
                  option: str = "put") -> np.ndarray:
    """Per-path replication error: portfolio minus payoff, at maturity.

    Params
    ------
    paths, deltas : as in portfolio_terminal_value.
    K, r, T, option : payoff parameters.
    V0 : initial portfolio value, passed through to portfolio_terminal_value.

    Returns
    -------
    (N,) array, V_T - payoff(S_T), undiscounted.

    QUÊTE 4.2 (suite) — assembler la couverture
    OBJECTIF : brancher `bs_delta_hedge_deltas` sur `portfolio_terminal_value`
        (4.1) puis soustraire le payoff réellement dû à maturité.
    VALIDATION : `test_hedging_error_moyenne_dans_ic_et_pente_log_log` (voir le
        bloc principal sur `bs_delta_hedge_deltas`).
    INDICE 1 (intuition) : V_T (4.1) est ce que vaut TON portefeuille de
        réplication à l'échéance ; le payoff est ce que tu DOIS au détenteur
        de l'option. L'écart entre les deux est ton erreur de couverture --
        nul en temps continu, pas en rebalancement discret.
    INDICE 2 (structure) : `deltas = bs_delta_hedge_deltas(...)`,
        `V_T = portfolio_terminal_value(paths, deltas, r, T, V0)`, puis
        `payoff = maximum(K - S_T, 0)` pour un put (`maximum(S_T - K, 0)` pour
        un call) avec `S_T = paths[:, -1]`.
    INDICE 3 (formule) : `hedging_error = V_T - payoff`.
    PIÈGE : ne PAS actualiser ici -- V_T et le payoff vivent tous les deux à
        maturité, l'actualisation n'a de sens que si tu veux comparer à un
        prix à t=0 (ce que fait `vol_arbitrage_pnl` en 4.3, explicitement).
    """
    portofolio = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return portofolio - payoff




def hedging_error_stats(errors: np.ndarray) -> tuple[float, float]:
    """Mean and 95% CI half-width of a hedging error sample.

    Params
    ------
    errors : (N,) array, e.g. the output of hedging_error.

    Returns
    -------
    (mean, half_width) : same convention as every estimator in the library.

    QUÊTE 4.2 (suite) — statistiques et taux de convergence
    OBJECTIF : produire (moyenne, demi-largeur IC 95%) sur un échantillon
        d'erreurs, EXACTEMENT comme `pricer_mc_put` ou `control_variate` --
        aucune formule nouvelle, juste la convention du repo appliquée à un
        nouvel objet.
    VALIDATION (l'identité qui ferme la 4.2) :
        `test_hedging_error_moyenne_dans_ic_et_pente_log_log` --
        (a) sur un run, la moyenne de l'erreur doit être dans son IC 95%
            autour de zéro (un hedge BS bien construit ne biaise pas) ;
        (b) sur une grille de `n_steps` (ex. 8, 16, 32, ..., 256), l'écart-type
            de l'erreur DÉCROÎT, et la régression log-log de cet écart-type
            contre `n_steps` a une pente proche de -1/2. Testé comme une
            PENTE (tolérance sur la régression), jamais comme un seuil
            numérique sur une seule valeur -- une pente fausse trahit une
            erreur de discrétisation, un seuil raté peut n'être qu'un
            mauvais choix de N.
    INDICE 1 (intuition) : plus tu rebalances souvent, plus ton portefeuille
        colle à l'option -- mais jamais parfaitement, parce qu'entre deux
        dates le delta que tu tiens est déjà légèrement faux (le spot a
        bougé). Doubler la fréquence ne divise pas l'erreur par deux, il la
        divise par racine de deux -- c'est une erreur de discrétisation d'un
        processus continu, pas un bruit MC qu'on ferait fondre en augmentant N.
    INDICE 2 (structure) : pour chaque `n_steps` de la grille, simule un
        NOUVEAU jeu de chemins (`gbm_paths`), calcule l'erreur de hedging sur
        ce jeu, prends son écart-type (`ddof=1`), et fais une régression
        linéaire de `log(std)` contre `log(n_steps)` (`np.polyfit(..., 1)`) :
        le premier coefficient est la pente cherchée.
    INDICE 3 (formule, référence indépendante) : Boyle & Emanuel (1980) donnent
        la variance limite de l'erreur de couverture discrète pour un
        portefeuille delta-neutre : elle est de l'ordre de
        `(pi/4) * (1/n_rebal) * E[ (Gamma(S_t,t) * S_t^2 * sigma^2 * dt)^2 ]`
        sommée sur les pas, ce qui donne un écart-type en `O(1/sqrt(n_rebal))`
        -- la pente -1/2 à tester. Ne sert QUE de justification théorique de
        la pente attendue ; le test ne calcule pas cette formule, il mesure la
        pente empirique.
    PIÈGE : ne pas réutiliser le MÊME `rng` pour tous les `n_steps` de la
        grille en croyant faire du CRN -- `gbm_paths` consomme
        `N * n_steps` tirages, donc deux appels avec des `n_steps` différents
        et le même générateur ne portent PAS sur les mêmes trajectoires
        sous-jacentes. Ici ce n'est pas un problème (on ne compare pas les
        runs entre eux terme à terme, seulement leurs écarts-types), mais s'en
        souvenir pour ne pas le supposer ailleurs dans l'acte.
    """
    mean = np.mean(errors)
    half_width = 1.96 * np.std(errors, ddof=1) / np.sqrt(errors.size)
    return mean, half_width


def bs_gamma(S: np.ndarray,
            K: float,
            sigma: float,
            r: float,
            tau: np.ndarray | float,
            q: float = 0.0) -> np.ndarray:
    """Closed-form Black-Scholes gamma, d2(price)/dS2.

    Identical for call and put (gamma is parity-invariant: d2C/dS2 = d2P/dS2
    since C - P is affine in S). Not in bs.py, which stays untouched; needed
    only for the vol-arbitrage identity of QUÊTE 4.3.

    Params
    ------
    S   : spot, scalar or array.
    K, sigma, r : as in bs.py.
    tau : time to maturity remaining, scalar or array broadcastable with S.
    q   : continuous dividend yield.

    Returns
    -------
    Gamma, same shape as the broadcast of S and tau.

    QUÊTE 4.3 — vol arbitrage / modèle faux [50 XP]
    OBJECTIF : écrire le gamma fermé de Black-Scholes -- la seule brique
        manquante pour boucler l'identité de 4.3, puisque `bs.py` n'a que
        delta et vega.
    DÉBLOQUE : `vol_arbitrage_pnl`, ci-dessous.
    VALIDATION : `test_bs_gamma_vs_difference_finie_sur_delta` -- gamma est la
        dérivée du delta par rapport au spot ; comparé à une différence finie
        de `bs.delta` (donc indépendant de la formule fermée elle-même),
        tolérance 1e-4 (erreur en O(h^2) d'une différence finie centrée bien
        réglée).
    INDICE 1 (intuition) : le delta bouge quand le spot bouge -- le gamma
        mesure À QUELLE VITESSE. Un gamma élevé veut dire un delta instable :
        c'est justement ce qui rend une couverture discrète imparfaite (4.2),
        et c'est ce paramètre-là que 4.3 va pondérer par l'écart de variance
        entre la vol de hedge et la vol réalisée.
    INDICE 2 (structure) : le gamma ne dépend que de d1 (pas de d2, pas de
        N(d1) lui-même) -- c'est la DENSITÉ gaussienne évaluée en d1, mise à
        l'échelle par S, sigma et la racine de tau. Une densité, pas un
        tirage aléatoire : `scipy.stats.norm.pdf`, pas `np.random.*`.
    INDICE 3 (formule) : gamma = exp(-q*tau)*phi(d1) / (S*sigma*sqrt(tau)), où
        phi est la densité N(0,1) et
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*tau) / (sigma*sqrt(tau)) --
        même d1 que dans bs.py, à tau près (qui remplace T).
    PIÈGE : gamma explose quand tau -> 0 pour une option proche de la monnaie
        (division par sqrt(tau) qui tend vers 0) -- attendu, pas un bug ; ça
        annonce déjà le problème structurel de 4.5 (delta près d'une
        barrière), où un gamma qui explose est précisément ce qui casse la
        couverture discrète.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-q * tau) * scipy.stats.norm.pdf(d1) / (S * sigma * np.sqrt(tau))



def vol_arbitrage_pnl(paths: np.ndarray,
                      K: float,
                      sigma_impl: float,
                      r: float,
                      T: float,
                      V0: float,
                      option: str = "put") -> np.ndarray:
    """Discounted P&L of hedging at sigma_impl while paths realise a different vol.

    Params
    ------
    paths : (N, n_steps+1) simulated paths -- however they were generated
        (any realised volatility; this function only hedges, it does not
        simulate).
    K, r, T, option : payoff parameters.
    sigma_impl : volatility used both to compute the delta (bs_delta_hedge_deltas)
        and to price V0 (typically bs.put_bs(S0, K, sigma_impl, r, T)).
    V0 : initial portfolio value.

    Returns
    -------
    (N,) discounted P&L, exp(-r*T) * (V_T - payoff(S_T)).

    QUÊTE 4.3 — vol arbitrage / modèle faux [50 XP]
    OBJECTIF : mesurer le P&L de réplication quand la vol utilisée pour
        hedger n'est pas celle réalisée par le marché -- la question
        d'entretien "tu marks à quelle vol, et le marché en réalise une
        autre, qu'est-ce qui se passe ?".
    DÉBLOQUE : rien en aval dans l'acte (branche indépendante de 4.4/4.5/4.6),
        mais c'est la quête la plus citée en entretien structuration.
    VALIDATION : `test_vol_arbitrage_identite_gamma` -- l'espérance du P&L
        (moyenne sur N paths, comparée à son IC 95%) doit coïncider avec la
        somme discrète de
        `-exp(-r*t_i) * 0.5 * gamma(S_i, tau_i) * S_i^2 * (sigma_real^2 - sigma_impl^2) * dt`
        le long des trajectoires (signe MOINS, et `exp(-r*t_i)` DANS la somme
        -- voir PIÈGE, ce facteur a fait tomber le test une première fois).
        C'est une IDENTITÉ (le P&L moyen doit tomber dans l'IC de la
        référence), pas une inégalité de signe -- une identité est un test
        bien plus dur à satisfaire par accident qu'un simple "le signe est
        bon".
    INDICE 1 (intuition) : couvrir en delta neutralise le PREMIER ordre
        (le mouvement linéaire du spot), jamais le second (la courbure,
        gamma). Si tu as hedgé en supposant une vol trop basse, tu es
        structurellement SOUS-couvert en gamma : chaque mouvement de spot un
        peu plus grand que prévu te coûte, ou te rapporte, selon le signe de
        l'écart de vol. C'est le mécanisme du "gamma scalping" à l'envers.
    INDICE 2 (structure) : la fonction elle-même ne fait rien de plus que
        `hedging_error` (4.2), actualisée -- toute la subtilité est dans le
        TEST, pas dans l'implémentation : il doit reconstruire une référence
        indépendante en sommant `bs_gamma` le long des trajectoires, avec la
        VRAIE vol des chemins (`sigma_real`, connue du test puisque c'est lui
        qui a simulé les paths) et la vol de hedge (`sigma_impl`, passée à
        cette fonction).
    INDICE 3 (formule) : pose e_t = Pi_t - V_t (portefeuille moins le prix
        marked-to-model à sigma_impl). En écrivant Itô sur e_t et en
        substituant le theta via la PDE de Black-Scholes, les termes en
        delta*dS s'annulent (delta = dV/dS) mais PAS tous les termes en r --
        il reste `de_t = r*e_t*dt - 0.5*Gamma_t*S_t^2*(sigma_real^2 - sigma_impl^2)*dt`,
        une EDO linéaire en e_t (pas juste un terme à intégrer tel quel : e_t
        capitalise lui-même au taux r). Avec e_0 = 0 (Pi_0 = V0 par
        construction), la résolution donne
        `e_T = - integral_0^T exp(r*(T-s)) * 0.5*Gamma_s*S_s^2*(sigma_real^2 - sigma_impl^2) ds`,
        soit, actualisé,
        `P&L moyen actualisé ≈ E[ - integral_0^T exp(-r*s) * 0.5*Gamma_s*S_s^2*(sigma_real^2 - sigma_impl^2) ds ]`
        où Gamma_s est évalué à sigma_impl (la vol du HEDGEUR, celle qui
        définit son modèle de delta/gamma -- pas sigma_real, qu'il ne connaît
        pas).
    PIÈGE : DEUX pièges de signe/facteur empilés ici, tombés dans cet ordre en
        écrivant ce module.
        (1) le signe -- MOINS, pas PLUS. `V0` te place du côté VENDEUR (tu
        encaisses `V0`, tu dois le payoff à maturité), donc ton portefeuille
        de réplication est court gamma. Si sigma_real > sigma_impl (le marché
        bouge plus que prévu), un court-gamma PERD de l'argent -- "vendre de
        la vol, c'est parier que le marché ne bougera pas plus que prévu".
        (2) le facteur `exp(-r*s)` DOIT être DANS l'intégrale, terme par
        terme -- pas juste `exp(-r*T)` sorti en facteur global devant toute la
        somme. La raison : l'erreur de couverture accumulée à l'instant s
        capitalise elle-même au taux r jusqu'à T (c'est le terme `r*e_t*dt`
        de l'EDO ci-dessus) ; l'oublier laisse un écart systématique de
        quelques % qui NE DIMINUE PAS quand n_steps augmente -- ce n'est pas
        un biais de discrétisation qui s'estompe, c'est un terme manquant
        dans la formule, un piège plus retors que celui du signe parce que le
        résultat reste plausible (bon ordre de grandeur, bon signe) à toutes
        les échelles de test.
        Troisième piège, documenté dans l'énoncé : le P&L est path-dependent
        (chaque trajectoire a son propre P&L, parfois très éloigné de zéro),
        même si son ESPÉRANCE ne dépend que de sigma_real et sigma_impl --
        ne pas confondre "l'identité tient en moyenne" avec "chaque path est
        proche de la référence".
    """
    delta = bs_delta_hedge_deltas(paths, K, sigma_impl, r, T, option=option)
    V_T = portfolio_terminal_value(paths, delta, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return np.exp(-r * T) * (V_T - payoff)


def turnover(deltas: np.ndarray) -> np.ndarray:
    """Total absolute share turnover along a hedge, per path.

    Params
    ------
    deltas : (N, n_steps) as in portfolio_terminal_value. delta_{-1} := 0, so
        the initial purchase of deltas[:, 0] shares counts as turnover too.

    Returns
    -------
    (N,) sum_i |delta_i - delta_{i-1}|, in shares.

    QUÊTE 4.4 — coûts de transaction [40 XP]
    OBJECTIF : mesurer, par path, combien de titres au total ont changé de
        mains le long du hedge -- la brique de base, réutilisée telle quelle
        par `transaction_costs` (pondérée en $) et par 4.5 (comme diagnostic
        brut, sans coût attaché).
    DÉBLOQUE : `transaction_costs`, `hedging_error_with_costs`.
    VALIDATION : `test_turnover_deltas_constants_et_alternes` -- deltas
        constants (turnover = |delta_0|, un seul achat, jamais de
        rebalancement ensuite) et deltas qui alternent de signe à chaque pas
        (turnover = somme de tous les |écarts|, cas où rien ne s'annule) :
        deux identités fermées, calculables à la main.
    INDICE 1 (intuition) : le turnover ne regarde que les ACTIONS échangées,
        pas leur prix -- c'est `transaction_costs` qui, ensuite, pondère
        chaque échange par le prix auquel il a lieu.
    INDICE 2 (structure) : `deltas` a `n_steps` colonnes ; il y a `n_steps`
        échanges au total -- le tout premier (l'achat initial de
        `deltas[:, 0]` titres, puisque tu partais de zéro action) puis
        `n_steps - 1` rebalancements entre colonnes consécutives.
    INDICE 3 (formule) : `turnover = |deltas[:,0]| + sum_i |deltas[:,i+1] - deltas[:,i]|`
        pour `i = 0 .. n_steps-2`.
    PIÈGE : ne pas confondre "turnover total sur tout le chemin" (ce que rend
        CETTE fonction, un scalaire par path) avec "turnover à CHAQUE date"
        (un vecteur par path, `(N, n_steps)`) -- `transaction_costs` a besoin
        du second pour pondérer chaque échange par le prix DU MOMENT, pas du
        premier : sommer d'abord en actions puis multiplier par un prix
        global n'a pas de sens (quel prix choisir ?), l'ordre des opérations
        compte.
    """
    turnover = np.abs(deltas[:, 0])  # initial purchase counts
    turnover += np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)
    return turnover


def transaction_costs(paths: np.ndarray,
                      deltas: np.ndarray,
                      cost_rate: float) -> np.ndarray:
    """Proportional transaction cost on delta turnover, in dollar terms.

    Params
    ------
    paths, deltas : as in portfolio_terminal_value.
    cost_rate : proportional cost, e.g. 0.001 for 10 bps per dollar traded.

    Returns
    -------
    (N,) total undiscounted cost paid along the path,
    cost_rate * sum_i |delta_i - delta_{i-1}| * S_i.

    QUÊTE 4.4 (suite) — pondérer le turnover par le prix
    OBJECTIF : transformer un turnover en ACTIONS (`turnover`, ci-dessus) en
        un coût en DOLLARS, en pondérant chaque échange par le prix auquel il
        a réellement lieu.
    VALIDATION : `test_transaction_costs_frequence_optimale` -- voir le bloc
        principal sur `hedging_error_with_costs`.
    INDICE 1 (intuition) : chaque rebalancement a lieu À UN PRIX PRÉCIS,
        celui du spot à cette date-là -- pas un prix moyen, pas le prix final.
        Le coût de chaque échange s'écrit AVANT de sommer sur les dates, pas
        après.
    INDICE 2 (structure) : contrairement à `turnover`, qui réduit tout de
        suite à un scalaire par path, ici il faut garder un tableau
        `(N, n_steps)` de turnover PAR DATE -- `|deltas[:,0]|` en colonne 0,
        puis `|diff(deltas, axis=1)|` pour les colonnes suivantes (même
        contenu que dans `turnover`, mais SANS sommer sur l'axe des dates
        avant de multiplier par le prix).
    INDICE 3 (formule) : `turnover_par_date = concatenate([|deltas[:,0:1]|, |diff(deltas, axis=1)|], axis=1)`,
        un tableau `(N, n_steps)` ; puis
        `cost = cost_rate * sum(turnover_par_date * paths[:, :-1], axis=1)`
        (`paths[:, :-1]`, PAS `paths[:, 1:]` -- l'échange à la date `i` se
        règle au prix `S_i`, celui affiché À CETTE DATE, pas au prochain).
    PIÈGE : appeler `turnover(deltas)` ici et multiplier le résultat (un
        scalaire par path) par `paths[:, :-1]` (une matrice par path) ne
        marche pas -- ce sont deux formes différentes du même calcul,
        l'agrégation (somme sur les dates) doit avoir lieu APRÈS la
        pondération par le prix, pas avant. `turnover` répond à "combien
        d'actions au total ?", `transaction_costs` a besoin de "combien
        d'actions, À CHAQUE date, à QUEL prix ?" -- deux questions
        différentes, malgré le nom qui se ressemble.
    """
    turnover = np.abs(deltas[:, 0])  # initial purchase counts
    turnover = turnover[:, np.newaxis]  # shape (N, 1) for broadcasting
    turnover = np.concatenate([turnover, np.abs(np.diff(deltas, axis=1))], axis=1)
    return cost_rate * np.sum(turnover * paths[:, :-1], axis=1)


def hedging_error_with_costs(paths: np.ndarray,
                             deltas: np.ndarray,
                             K: float,
                             r: float,
                             T: float,
                             V0: float,
                             cost_rate: float,
                             option: str = "put") -> np.ndarray:
    """Hedging error (as in hedging_error) net of proportional transaction costs.

    Params
    ------
    paths, deltas, K, r, T, V0, option : as in hedging_error.
    cost_rate : as in transaction_costs.

    Returns
    -------
    (N,) array, undiscounted.

    QUÊTE 4.4 (suite) — la fréquence optimale
    OBJECTIF : montrer qu'il existe un `n_steps` optimal, ni trop rare (erreur
        de couverture élevée) ni trop fréquent (coûts qui dominent) --
        l'arbitrage concret que fait un desk.
    VALIDATION (l'identité qui ferme la 4.4) :
        `test_transaction_costs_frequence_optimale` -- sur une grille de
        `n_steps` croissants (mêmes trajectoires que 4.2, cost_rate fixé) :
        (a) le coût moyen de transaction CROÎT en `sqrt(n_rebal)` (pente log-log
        ≈ +1/2, testée avec la même tolérance que la pente -1/2 de 4.2) ;
        (b) l'écart-type de l'erreur de hedging SANS coûts continue de
        décroître comme en 4.2 ; (c) en combinant les deux dans une mesure
        d'erreur totale (ex. RMS de `hedging_error_with_costs`), il existe un
        `n_steps` intermédiaire qui minimise cette mesure -- ni le plus petit
        ni le plus grand de la grille.
    INDICE 1 (intuition) : rebalancer plus souvent réduit l'erreur de
        réplication (4.2) mais chaque rebalancement coûte quelque chose
        (4.4) -- les deux effets bougent en sens opposé avec `n_steps`, donc
        quelque part au milieu, leur somme est minimale. C'est l'argument de
        Leland (1985) : au-delà d'une fréquence optimale, se couvrir plus
        souvent coûte plus cher que ça ne rapporte en précision.
    INDICE 2 (structure) : cette fonction ne fait qu'assembler ce qui existe
        déjà -- `portfolio_terminal_value` (4.1), le payoff (comme dans
        `hedging_error`, 4.2), et `transaction_costs` (ci-dessus) -- toute la
        substance de la quête est dans le TEST, comme pour 4.2/4.3.
    INDICE 3 (formule, référence indépendante) : Leland (1985) ajuste la vol
        de hedge d'un terme `sigma_Leland^2 = sigma^2 * (1 + sqrt(2/pi) * k / (sigma*sqrt(dt)))`,
        où `k` est le taux de coût proportionnel -- le coût total attendu
        croît comme `E[turnover] * S * k`, et `E[turnover]` croît lui-même en
        `sqrt(n_rebal)` (marche aléatoire du delta, dont les incréments ont un
        écart-type en `sqrt(dt) = sqrt(T/n_rebal)`, sommés sur `n_rebal` pas
        indépendants). Donné en référence théorique de la pente attendue ; le
        test ne calcule pas cette formule, il mesure la pente empirique du
        coût.
    PIÈGE : `hedging_error_with_costs` retourne V_T - payoff - costs, donc un
        `cost_rate` élevé rend l'erreur SYSTÉMATIQUEMENT négative (en moyenne)
        -- ce n'est plus centré en zéro comme en 4.2, et c'est ATTENDU : les
        coûts sont une perte certaine, pas un bruit. Ne pas réutiliser le test
        "moyenne dans l'IC autour de zéro" de 4.2 tel quel ici.
    """
    V_T = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    costs = transaction_costs(paths, deltas, cost_rate)
    return V_T - payoff - costs


def naive_barrier_hedge_error(paths: np.ndarray,
                              K: float,
                              H: float,
                              sigma: float,
                              r: float,
                              T: float,
                              V0: float,
                              barrier: str = "do",
                              option: str = "put") -> np.ndarray:
    """Discrete hedge of a DO/DI put using the vanilla put's BS delta.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, used both to hedge (BS delta at
        each column) and to read the true DO/DO payoff at maturity via
        barriers.do_put / barriers.di_put logic.
    K, H, sigma, r, T : strike, barrier, vol used for the delta, rate, maturity.
    V0 : initial portfolio value -- the true barrier option's price, not the
        vanilla's.
    barrier : "do" or "di".
    option  : "call" or "put".

    Returns
    -------
    (N,) hedging error, undiscounted.

    QUÊTE 4.5 — delta près d'une barrière [40 XP]
    OBJECTIF : mesurer ce qui casse quand on hedge une option À BARRIÈRE avec
        le SEUL delta disponible en forme fermée -- celui du vanille sur le
        même K, T. Le vrai delta d'un DO/DI put (la dérivée du PRIX BARRIÈRE
        par rapport au spot) n'a pas de formule fermée ici (monitoring
        discret, cf. README "Not covered") ; c'est le mismatch qu'un desk
        affronte concrètement dès qu'aucun Grec barrière n'est disponible.
    DÉBLOQUE : rien en aval (branche indépendante de 4.6), mais c'est LA
        quête "vrai problème métier" de l'acte.
    VALIDATION : `test_naive_barrier_hedge_degrade_pres_de_la_barriere` --
        pas de référence fermée (donc pas d'identité), le test porte sur ce
        qui DOIT casser : à `n_steps` fixé, l'écart-type de l'erreur de hedge
        AUGMENTE nettement quand `H` se rapproche de `S0`, comparé à un `H`
        loin du spot. Un test qui prouve une DÉGRADATION vaut autant qu'un
        test qui prouve une convergence (cf. 4.2) -- il porte juste sur le
        sens de variation d'une statistique, pas sur un seuil numérique nu.
    INDICE 1 (intuition) : le delta vanille ne "voit" jamais la barrière --
        il varie doucement avec S, alors que le VRAI prix d'un DO/DI put a
        une pente qui change brutalement quand S croise H (le payoff lui-même
        est discontinu en trajectoire : `max(K-S_T,0)` multiplié par un
        indicateur qui bascule à 0 ou 1). Plus H est proche de S0, plus les
        trajectoires traversent H souvent, plus le delta vanille est
        structurellement à côté de la plaque à ces moments-là.
    INDICE 2 (structure) : `deltas = bs_delta_hedge_deltas(paths, K, sigma, r, T, option)`
        (exactement comme en 4.2, AUCUNE dépendance à `H` dans le calcul du
        delta -- c'est le point) ; `V_T = portfolio_terminal_value(paths, deltas, r, T, V0)` ;
        puis le VRAI payoff barrière (pas le payoff vanille) à partir de
        `paths.min(axis=1)` et de la condition `barrier`.
    INDICE 3 (formule) : payoff vanille `p = maximum(K - S_T, 0)` (put) ou
        `maximum(S_T - K, 0)` (call) ; `min_path = paths.min(axis=1)` ;
        DO -- payoff = `where(min_path >= H, p, 0)` (survit tant que la
        barrière n'est jamais touchée, convention stricte comme dans
        `barriers.py`) ; DI -- payoff = `where(min_path < H, p, 0)`.
        Erreur = `V_T - payoff`.
    PIÈGE : `V0` doit être le prix du VRAI DO/DI put (typiquement
        `barriers.do_put(paths, K, H, r, T)[0]` ou `di_put(...)`, sur les
        MÊMES trajectoires), pas `bs.put_bs(...)` -- financer la réplication
        avec le mauvais prix de départ biaiserait l'erreur d'un terme
        constant qui n'aurait rien à voir avec la dégradation du delta,
        exactement le même piège que le `sigma` en dur repéré en 4.2.
    """
    deltas = bs_delta_hedge_deltas(paths, K, sigma, r, T, option)
    V_T = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff_vanilla = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    min_path = paths.min(axis=1)
    if barrier == "do":
        payoff_barrier = np.where(min_path >= H, payoff_vanilla, 0)
    elif barrier == "di":
        payoff_barrier = np.where(min_path < H, payoff_vanilla, 0)
    else:
        raise ValueError("barrier must be 'do' or 'di'")
    return V_T - payoff_barrier

def heston_delta_bs_hedge_error(S: np.ndarray,
                                K: float,
                                sigma_hedge: float,
                                r: float,
                                T: float,
                                V0: float,
                                option: str = "put") -> np.ndarray:
    """Discrete BS-delta-only hedge of paths simulated under Heston.

    Params
    ------
    S : (N, n_steps+1) HESTON spot paths (heston.heston_paths, first array
        only -- the variance path is not used here).
    K, sigma_hedge, r, T : delta and pricing parameters, a single constant
        vol standing in for the whole (flat, wrong) hedge.
    V0 : initial portfolio value.
    option : "call" or "put".

    Returns
    -------
    (N,) hedging error, undiscounted.

    QUÊTE 4.6 — delta-vega sous Heston [50 XP]
    OBJECTIF : hedger en delta BS SEUL (une vol constante, plate) des
        trajectoires simulées sous un modèle qui a un vrai smile -- montrer
        que le hedge est biaisé ET à variance élevée, faute de vega couvert.
    DÉBLOQUE : `heston_delta_vega_hedge_error`, ci-dessous (qui rajoute
        l'overlay vega et doit faire chuter la variance).
    VALIDATION : `test_heston_delta_vega_reduit_la_variance` -- voir le bloc
        principal sur `heston_delta_vega_hedge_error`.
    INDICE 1 (intuition) : le delta BS suppose une vol CONSTANTE dans le
        temps et dans l'espace ; sous Heston la vol est STOCHASTIQUE (elle a
        sa propre source d'aléa, corrélée au spot par `rho`). Un hedge qui ne
        regarde que le spot laisse filer tout le risque porté par les
        mouvements de la vol elle-même -- c'est exactement ce qu'un vega
        mesure, et ce hedge n'en a aucun.
    INDICE 2 (structure) : structurellement identique à `hedging_error` (4.2)
        -- `bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)`,
        `portfolio_terminal_value(S, deltas, r, T, V0)`, payoff à maturité --
        seule différence : `S` vient de `heston.heston_paths`, pas de
        `gbm_paths`.
    INDICE 3 (formule) : rien de nouveau, c'est un simple branchement de 4.2
        sur des trajectoires Heston au lieu de GBM.
    PIÈGE : `V0` doit être cohérent avec `sigma_hedge` -- typiquement
        `bs.put_bs(S0, K, sigma_hedge, r, T)`, PAS `heston.heston_put(...)`
        (qui donnerait le vrai prix du modèle, pas celui, plat et faux, que
        le hedgeur croit utiliser). Mélanger les deux ferait disparaître
        EXACTEMENT le biais qu'on cherche à mesurer.
    """
    deltas = bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)
    V_T = portfolio_terminal_value(S, deltas, r, T, V0)
    S_T = S[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return V_T - payoff


def heston_delta_vega_hedge_error(S: np.ndarray,
                                  K: float,
                                  K_vega: float,
                                  sigma_hedge: float,
                                  r: float,
                                  T: float,
                                  V0: float,
                                  option: str = "put") -> np.ndarray:
    """Same hedge as heston_delta_bs_hedge_error, plus a static vega overlay.

    Params
    ------
    S : (N, n_steps+1) Heston spot paths.
    K : target option's strike.
    K_vega : strike of the vanilla instrument added to the book, held in a
        fixed quantity computed once at t=0 to neutralise vega -- a static
        overlay, not a dynamically rebalanced vega hedge.
    sigma_hedge, r, T, V0, option : as in heston_delta_bs_hedge_error.

    Returns
    -------
    (N,) hedging error of the delta+static-vega book, undiscounted.

    QUÊTE 4.6 (suite) — l'overlay vega
    OBJECTIF : ajouter, EN PLUS du delta hedge dynamique de
        `heston_delta_bs_hedge_error`, une position STATIQUE (figée à t=0,
        jamais rebalancée) dans un vanille sur un autre strike, dimensionnée
        pour annuler le vega de la position à t=0 -- et montrer que la
        variance résiduelle chute.
    VALIDATION (l'identité qui ferme la 4.6) :
        `test_heston_delta_vega_reduit_la_variance` -- à N et n_steps
        identiques, `Var(heston_delta_bs_hedge_error) / Var(heston_delta_vega_hedge_error) > 1`,
        avec un intervalle de confiance sur ce ratio (bootstrap ou F-test
        approximatif) -- pas juste "plus petit", un RAPPORT DE VARIANCES
        significativement supérieur à 1, à budget de trajectoires égal.
    INDICE 1 (intuition) : tu ne peux pas rebalancer le vega en continu aussi
        facilement que le delta (il faudrait retrader l'option de couverture
        à chaque pas, ce qui a un coût et sort du cadre de cette quête) --
        mais même une couverture vega STATIQUE, posée une fois à t=0 et
        jamais retouchée, absorbe une bonne partie du risque de vol qui
        échappait au hedge delta seul.
    INDICE 2 (structure) : à t=0, calcule
        `n_vega = bs.vega(S0, K, sigma_hedge, r, T) / bs.vega(S0, K_vega, sigma_hedge, r, T)`
        (le ratio des vegas BS des deux options, à la vol de hedge -- combien
        d'unités de l'option `K_vega` il faut pour égaler le vega de la
        cible). Le portefeuille total détient alors, À CHAQUE date, le delta
        BS de la cible PLUS `n_vega` fois le delta BS de l'instrument
        `K_vega` (les deux calculés par `bs_delta_hedge_deltas`, même
        `sigma_hedge`, chacun sur son propre strike) -- `n_vega` est FIGÉ,
        calculé une seule fois, mais son DELTA, lui, continue d'être
        rebalancé à chaque date comme celui de la cible (piège ci-dessous).
        Le cash de départ doit financer l'achat des `n_vega` unités de
        l'overlay, en plus de la réplication de la cible.
    INDICE 3 (formule) : `deltas_total = bs_delta_hedge_deltas(S,K,...) - n_vega*bs_delta_hedge_deltas(S,K_vega,...)` --
        SOUSTRAIT, pas ajouté (voir PIÈGE 1) ;
        `V0_total = V0 - n_vega*bs.put_bs(S0,K_vega,sigma_hedge,r,T)` (ou
        `call_bs`, selon `option`) ; `payoff_total = payoff(K) - n_vega*payoff(K_vega)`
        (tu ENCAISSES le payoff de l'overlay que tu détiens, il compense une
        partie de ce que tu dois sur la cible) ; erreur =
        `portfolio_terminal_value(S, deltas_total, r, T, V0_total) - payoff_total`.
    PIÈGE : deux pièges ici.
        (1) le signe du delta de l'overlay -- `deltas_total` SOUSTRAIT
        `n_vega*delta_Kvega`, ne l'ajoute pas. Raison : `bs_delta_hedge_deltas`
        rend le delta à utiliser directement pour répliquer une position
        COURTE (c'est la convention déjà validée en 4.2 -- `deltas` branché
        tel quel dans `portfolio_terminal_value` avec `V0` = prime encaissée).
        L'overlay, lui, est une position LONGUE (tu ACHÈTES `n_vega` unités
        de l'option `K_vega`) -- une position longue en delta `δ` a besoin de
        `-δ` actions pour être delta-neutre, pas `+δ`. Se tromper de signe ici
        ne dégrade pas la variance, il l'AUGMENTE largement (mesuré : ratio
        de variance sous 1 au lieu d'être nettement au-dessus) -- si ton
        `test_heston_delta_vega_reduit_la_variance` échoue avec un ratio très
        inférieur à 1, regarde ce signe en premier.
        (2) "statique" qualifie la QUANTITÉ `n_vega` (calculée une fois, à
        t=0), PAS le delta de l'instrument de couverture -- son delta, lui,
        varie avec S et tau comme n'importe quel delta BS, et doit être
        rebalancé à chaque date exactement comme celui de la cible. Un overlay
        "vraiment statique" (delta jamais recalculé non plus) laisserait
        filer un delta résiduel et fausserait la comparaison : la quête ne
        teste la réduction de variance QUE si le delta total reste
        correctement suivi à chaque pas.
    """
    deltas_total = bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)
    deltas_overlay = bs_delta_hedge_deltas(S, K_vega, sigma_hedge, r, T, option)
    n_vega = bs.vega(S[:, 0], K, sigma_hedge, r, T) / bs.vega(S[:, 0], K_vega, sigma_hedge, r, T)
    deltas_total -= n_vega[:, np.newaxis] * deltas_overlay
    V0_total = V0 - n_vega * bs.put_bs(S[:, 0], K_vega, sigma_hedge, r, T) if option == "put" else V0 - n_vega * bs.call_bs(S[:, 0], K_vega, sigma_hedge, r, T)
    V_T = portfolio_terminal_value(S, deltas_total, r, T, V0_total)
    S_T = S[:, -1]
    payoff_target = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    payoff_overlay = np.maximum(K_vega - S_T, 0) if option == "put" else np.maximum(S_T - K_vega, 0)
    payoff_total = payoff_target - n_vega * payoff_overlay
    return V_T - payoff_total
