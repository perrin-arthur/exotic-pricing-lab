"""Acte IV — hedging discret : portefeuille auto-financé, erreur de couverture.

Cette première tranche couvre QUÊTE 4.1 et QUÊTE 4.2 uniquement. Les tests de
4.3 à 4.6 et du BOSS 4 arrivent une fois la sortie de 4.2 postée et validée --
c'est une règle du CLAUDE.md du repo, pas un oubli.

4.1 est une identité algébrique (deltas arbitraires, 1e-12) : elle ne dépend
d'aucun pricing. 4.2 est un test de TAUX de convergence en log-log, jamais un
seuil numérique nu -- voir test_hedging_error_moyenne_dans_ic_et_pente_log_log.

Run: .venv/bin/python -m pytest tests/test_hedging.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np
import pytest
from scipy.stats import norm

import barriers
import bs
import hedging
from bs import call_bs, put_bs, vega
from heston import heston_paths
from mc_engine import gbm_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Paramètres de référence, cohérents avec ceux des actes précédents.
HP = dict(S0=100.0, K=100.0, sigma=0.20, r=0.05, T=1.0)

# Paramètres Heston de référence, identiques à ceux de tests/test_heston.py.
HESTON_HP = dict(S0=100.0, v0=0.04, r=0.05, T=1.0,
                  kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


def _half_width(v: np.ndarray) -> float:
    """95% CI half-width -- la convention du repo."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))



def test_autofinancement_deltas_arbitraires():
    """L'identité d'auto-financement tient pour N'IMPORTE QUELLE suite de deltas.

    Deltas tirés au hasard dans [-5, 5] (négatifs, énormes, rien à voir avec un
    vrai delta d'option) : ce test ne valide pas un pricing, il valide une
    comptabilité. La récursion de référence est reconstruite ICI, dans le
    test, indépendamment de `hedging.py` -- comparer la fonction à elle-même
    ne prouverait rien.
    """
    rng = np.random.default_rng(0)
    N, n_steps = 500, 12
    r, T = 0.03, 1.0
    dt = T / n_steps

    paths = gbm_paths(S0=100.0, sigma=0.25, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = rng.uniform(-5.0, 5.0, size=(N, n_steps))
    V0 = rng.uniform(1.0, 20.0, size=N)

    # Récursion de référence, écrite indépendamment de hedging.py.
    B = V0 - deltas[:, 0] * paths[:, 0]
    for i in range(n_steps - 1):
        B = B * np.exp(r * dt) - (deltas[:, i + 1] - deltas[:, i]) * paths[:, i + 1]
    V_T_ref = deltas[:, -1] * paths[:, -1] + B * np.exp(r * dt)

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)

    assert V_T.shape == (N,)
    assert np.max(np.abs(V_T - V_T_ref)) < 1e-12


def test_autofinancement_delta_nul_est_du_cash_pur():
    """deltas = 0 partout : le portefeuille ne touche jamais l'action.

    V_T doit alors être exactement V0 capitalisé au taux sans risque --
    aucune trajectoire ne devrait intervenir dans le résultat.
    """
    rng = np.random.default_rng(1)
    N, n_steps = 1_000, 20
    r, T = 0.04, 1.0

    paths = gbm_paths(S0=100.0, sigma=0.3, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = np.zeros((N, n_steps))
    V0 = 7.5

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)

    assert np.max(np.abs(V_T - V0 * np.exp(r * T))) < 1e-10


def test_autofinancement_delta_constant_formule_fermee():
    """deltas = c partout : identité fermée, indépendante de la boucle interne.

    Sans rebalancement, l'argent placé en cash capitalise pendant toute la
    période, et le résultat s'écrit en circuit fermé :
        V_T = V0*exp(rT) + c*(S_T - S0*exp(rT))
    C'est une SECONDE référence indépendante de la récursion pas-à-pas testée
    plus haut -- une erreur qui se compenserait par hasard dans la récursion
    devrait quand même se voir ici.
    """
    rng = np.random.default_rng(2)
    N, n_steps = 1_000, 15
    r, T, S0 = 0.05, 1.0, 100.0
    c = 0.4

    paths = gbm_paths(S0=S0, sigma=0.2, r=r, T=T, n_steps=n_steps,
                       N=N, rng=rng)
    deltas = np.full((N, n_steps), c)
    V0 = 10.0

    V_T = hedging.portfolio_terminal_value(paths, deltas, r, T, V0)
    V_T_ref = V0 * np.exp(r * T) + c * (paths[:, -1] - S0 * np.exp(r * T))

    assert np.max(np.abs(V_T - V_T_ref)) < 1e-9


def test_bs_delta_hedge_deltas_vs_formule_fermee():
    """Au premier pas (tau = T), le delta doit coïncider avec N(d1) - 1 (put).

    Référence écrite directement ici avec scipy.stats.norm, indépendante de
    bs.py. Tolérance 1e-3 : pas une identité algébrique (bs_delta_hedge_deltas
    peut passer par une différence finie interne), mais un ordre de grandeur
    qui ne pardonne pas une formule fausse.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N, n_steps = 2_000, 10

    paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(10))
    deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")

    assert deltas.shape == (N, n_steps)

    S_t0 = paths[:, 0]
    d1 = (np.log(S_t0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta_ref = norm.cdf(d1) - 1.0

    assert np.max(np.abs(deltas[:, 0] - delta_ref)) < 1e-3


def test_hedging_error_moyenne_dans_ic_et_pente_log_log():
    """4.2, la validation qui ferme la quête.

    (a) Sur un run donné, la moyenne de l'erreur de hedging doit être dans son
        propre IC 95% autour de zéro : un hedge BS correct, sous le modèle
        qu'il suppose, ne biaise pas le portefeuille de réplication.
    (b) Sur une grille de n_steps, l'écart-type de l'erreur décroît, et la
        pente de la régression log(std) ~ log(n_steps) doit être proche de
        -1/2 (Boyle & Emanuel, 1980 -- donné en INDICE 3 de hedging.py, pas
        ici). Tolérance large sur la pente ([-0.8, -0.2]) car c'est un ajustement
        statistique sur peu de points, pas une identité -- ce qui est testé
        est la DIRECTION et l'ORDRE DE GRANDEUR de la convergence, jamais un
        seuil numérique sur une seule valeur d'écart-type.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N = 30_000
    n_steps_grid = [8, 16, 32, 64, 128, 256]

    V0 = put_bs(S0, K, sigma, r, T)
    stds = []

    for n_steps in n_steps_grid:
        paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                           N=N, rng=np.random.default_rng(1000 + n_steps))
        deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")
        errors = hedging.hedging_error(paths, deltas, K, r, T, V0, option="put")

        mean, half_width = hedging.hedging_error_stats(errors)
        assert abs(mean) < half_width * 1.5, (
            f"n_steps={n_steps}: moyenne {mean:.4f} hors IC (+/-{half_width:.4f})"
        )

        stds.append(errors.std(ddof=1))

    stds = np.asarray(stds)
    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(stds), 1)
    print(f"pente log-log mesuree = {slope:.3f} (attendu ~ -0.5)")

    assert np.all(np.diff(stds) < 0.0)          # l'ecart-type decroit
    assert -0.8 < slope < -0.2


def test_hedging_error_stats_convention():
    """hedging_error_stats suit la convention du repo : (moyenne, demi-IC 95%).

    Identité pure sur un échantillon quelconque, sans lien avec le hedging --
    verrouille juste la formule 1.96*sd/sqrt(n) sur ddof=1, comme partout
    ailleurs dans le repo.
    """
    rng = np.random.default_rng(3)
    errors = rng.normal(loc=0.7, scale=2.5, size=5_000)

    mean, half_width = hedging.hedging_error_stats(errors)

    assert abs(mean - errors.mean()) < 1e-12
    assert abs(half_width - _half_width(errors)) < 1e-12


# ---------------------------------------------------------------------------
# QUÊTE 4.3 — vol arbitrage / modèle faux
# ---------------------------------------------------------------------------


def test_bs_gamma_vs_difference_finie_sur_delta():
    """Gamma est la dérivée du delta par rapport au spot.

    Référence indépendante de la formule fermée : une différence finie
    centrée sur bs.delta (déjà elle-même une différence finie sur le prix).
    Tolérance 1e-3 -- deux couches de différences finies empilées, l'erreur
    ne peut pas être 1e-10.
    """
    K, sigma, r = 100.0, 0.25, 0.03
    h_outer = 0.01

    for S0, tau in [(90.0, 0.5), (100.0, 1.0), (115.0, 0.25)]:
        gamma_closed = hedging.bs_gamma(S0, K, sigma, r, tau)

        delta_up = bs.delta(S0 + h_outer, 1e-4, K, sigma, r, tau)
        delta_down = bs.delta(S0 - h_outer, 1e-4, K, sigma, r, tau)
        gamma_fd = (delta_up - delta_down) / (2 * h_outer)

        print(f"S0={S0} tau={tau}  gamma_ferme={gamma_closed:.6f}  gamma_fd={gamma_fd:.6f}")
        assert abs(gamma_closed - gamma_fd) < 1e-3


def test_vol_arbitrage_identite_gamma():
    """L'identité de 4.3 : P&L moyen actualisé = -integrale de gamma ponderee.

    Reference reconstruite ICI, independamment de vol_arbitrage_pnl : la somme
    discrete de -exp(-r*t_i)*0.5*Gamma_i*S_i^2*(sigma_real^2-sigma_impl^2)*dt
    le long de chaque trajectoire, moyennee sur les N paths. Signe MOINS : V0
    place le portefeuille du cote VENDEUR (court gamma), qui perd quand la vol
    realisee depasse la vol de hedge. Le facteur exp(-r*t_i) DANS la somme
    (pas juste exp(-rT) en facteur global) vient de la resolution de l'EDO
    de_t = r*e_t*dt - 0.5*Gamma*S^2*(sigma_real^2-sigma_impl^2)*dt sur
    l'erreur de couverture e_t = Pi_t - V_t : l'erreur accumulee capitalise
    elle-meme au taux r jusqu'a maturite, elle n'est pas juste actualisee une
    fois a la fin (voir INDICE 3 de vol_arbitrage_pnl dans hedging.py -- et le
    PIEGE qui documente cette erreur, tombee une premiere fois ici). C'est une
    IDENTITE testee sur la MOYENNE (comparee a son IC 95%), pas une inegalite
    de signe -- un signe correct peut sortir d'une formule fausse par chance,
    une identite numerique beaucoup plus difficilement.
    """
    S0, K, r, T = 100.0, 100.0, 0.05, 1.0
    sigma_impl, sigma_real = 0.20, 0.30
    N, n_steps = 20_000, 100
    dt = T / n_steps

    paths = gbm_paths(S0=S0, sigma=sigma_real, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(42))
    V0 = put_bs(S0, K, sigma_impl, r, T)

    pnl = hedging.vol_arbitrage_pnl(paths, K, sigma_impl, r, T, V0, option="put")
    mean_pnl, half_width_pnl = hedging.hedging_error_stats(pnl)

    t_i = np.arange(n_steps) * dt                 # debut de chaque intervalle
    tau = T - t_i                                  # (n_steps,) -- une par colonne rebalancee
    S_rebal = paths[:, :-1]                        # (N, n_steps)
    gamma = hedging.bs_gamma(S_rebal, K, sigma_impl, r, tau)
    integrand = -np.exp(-r * t_i) * 0.5 * gamma * S_rebal**2 * (sigma_real**2 - sigma_impl**2) * dt
    ref_per_path = integrand.sum(axis=1)
    mean_ref = ref_per_path.mean()

    print(f"P&L moyen mesure={mean_pnl:.4f}+/-{half_width_pnl:.4f}  "
          f"reference gamma={mean_ref:.4f}")

    assert abs(mean_pnl - mean_ref) < half_width_pnl * 1.5


# ---------------------------------------------------------------------------
# QUÊTE 4.4 — coûts de transaction
# ---------------------------------------------------------------------------


def test_turnover_deltas_constants_et_alternes():
    """Deux identités fermées sur le turnover, aux deux extrêmes.

    Deltas constants : un seul achat, jamais de rebalancement ensuite --
    turnover = |delta_0| exactement. Deltas qui alternent de signe à chaque
    pas : rien ne s'annule jamais, turnover = somme de tous les |ecarts|,
    calculable à la main terme à terme.
    """
    N, n_steps = 200, 10

    deltas_const = np.full((N, n_steps), 3.5)
    assert np.max(np.abs(hedging.turnover(deltas_const) - 3.5)) < 1e-12

    # alterne +2, -2, +2, -2, ... : |2-0| + |−2−2| + |2−(−2)| + ... = 2 + 4*(n_steps-1)
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n_steps)])
    deltas_alt = 2.0 * np.tile(signs, (N, 1))
    attendu = 2.0 + 4.0 * (n_steps - 1)
    assert np.max(np.abs(hedging.turnover(deltas_alt) - attendu)) < 1e-12


def test_transaction_costs_deltas_constants():
    """Deltas constants : un seul échange, au tout premier prix S_0.

    Identité fermée indépendante de `turnover` : cost = cost_rate * |delta| * S0,
    puisqu'aucun rebalancement n'a lieu après l'achat initial.
    """
    rng = np.random.default_rng(5)
    N, n_steps, S0 = 500, 8, 100.0
    paths = gbm_paths(S0=S0, sigma=0.2, r=0.05, T=1.0, n_steps=n_steps,
                       N=N, rng=rng)
    c = 1.7
    deltas = np.full((N, n_steps), c)
    cost_rate = 0.002

    costs = hedging.transaction_costs(paths, deltas, cost_rate)
    attendu = cost_rate * abs(c) * S0

    assert np.max(np.abs(costs - attendu)) < 1e-9


def test_transaction_costs_frequence_optimale():
    """4.4, la validation qui ferme la quête.

    (a) le coût moyen de transaction CROÎT en sqrt(n_rebal) -- pente log-log
        positive, proche de +1/2 (tolérance large, même esprit que 4.2) ;
    (b) l'écart-type de l'erreur de hedging SANS coûts continue de décroître
        (rappel de 4.2, mêmes trajectoires) ;
    (c) il existe un n_steps qui minimise le RMS de l'erreur AVEC coûts --
        ni le plus petit ni le plus grand de la grille testée. Un test qui
        prouve l'existence d'un optimum intermédiaire, pas une convergence.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    N = 20_000
    n_steps_grid = [4, 8, 16, 32, 64, 128, 256, 512]
    cost_rate = 0.005

    V0 = put_bs(S0, K, sigma, r, T)
    mean_costs, rms_with_costs = [], []

    for n_steps in n_steps_grid:
        paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                           N=N, rng=np.random.default_rng(2000 + n_steps))
        deltas = hedging.bs_delta_hedge_deltas(paths, K, sigma, r, T, option="put")

        costs = hedging.transaction_costs(paths, deltas, cost_rate)
        mean_costs.append(costs.mean())

        errors_with_costs = hedging.hedging_error_with_costs(
            paths, deltas, K, r, T, V0, cost_rate, option="put")
        rms_with_costs.append(np.sqrt(np.mean(errors_with_costs**2)))

    mean_costs = np.asarray(mean_costs)
    rms_with_costs = np.asarray(rms_with_costs)

    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(mean_costs), 1)
    print(f"pente log-log du cout moyen = {slope:.3f} (attendu ~ +0.5)")
    print(f"RMS erreur avec couts = {np.round(rms_with_costs, 4)}")

    assert np.all(np.diff(mean_costs) > 0.0)      # le cout croit
    assert 0.2 < slope < 0.8

    i_min = int(np.argmin(rms_with_costs))
    assert 0 < i_min < len(n_steps_grid) - 1       # optimum intermediaire, pas aux bords


# ---------------------------------------------------------------------------
# QUÊTE 4.5 — delta près d'une barrière
# ---------------------------------------------------------------------------


def test_naive_barrier_hedge_degrade_pres_de_la_barriere():
    """4.5, la validation qui ferme la quête : pas de convergence, une DÉGRADATION.

    Deux barrières, mêmes trajectoires (CRN) : une loin du spot (H=60, le
    knock-out est rare, le delta vanille hedge presque un vanille), une
    proche du spot (H=90, les trajectoires croisent la barrière souvent, le
    delta vanille ne voit jamais la discontinuité du payoff).

    (a) l'ecart-type de l'erreur pres de la barriere doit etre nettement plus
        grand que loin de la barriere (ratio > 2, mesure empiriquement ~2.8) ;
    (b) pres de la barriere, augmenter n_steps ne fait PAS decroitre
        l'ecart-type comme en 4.2 -- la pente log-log doit rester proche de 0
        (mesuree ~0.0, tres different du -0.5 de 4.2), signe que l'erreur est
        un biais structurel du delta vanille, pas du bruit MC qui se moyenne.
    """
    S0, K, sigma, r, T = HP["S0"], HP["K"], HP["sigma"], HP["r"], HP["T"]
    H_far, H_near = 60.0, 90.0
    N, n_steps = 20_000, 50

    paths = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n_steps,
                       N=N, rng=np.random.default_rng(21))

    V0_far = barriers.do_put(paths, K, H_far, r, T)[0]
    V0_near = barriers.do_put(paths, K, H_near, r, T)[0]

    err_far = hedging.naive_barrier_hedge_error(
        paths, K, H_far, sigma, r, T, V0_far, barrier="do", option="put")
    err_near = hedging.naive_barrier_hedge_error(
        paths, K, H_near, sigma, r, T, V0_near, barrier="do", option="put")

    std_far, std_near = err_far.std(ddof=1), err_near.std(ddof=1)
    print(f"std(H={H_far})={std_far:.4f}  std(H={H_near})={std_near:.4f}  "
          f"ratio={std_near / std_far:.3f}")
    assert std_near / std_far > 2.0

    stds_near = []
    n_steps_grid = [16, 32, 64, 128, 256]
    for n in n_steps_grid:
        p = gbm_paths(S0=S0, sigma=sigma, r=r, T=T, n_steps=n,
                       N=N, rng=np.random.default_rng(3000 + n))
        v0_n = barriers.do_put(p, K, H_near, r, T)[0]
        e = hedging.naive_barrier_hedge_error(
            p, K, H_near, sigma, r, T, v0_n, barrier="do", option="put")
        stds_near.append(e.std(ddof=1))

    slope, _ = np.polyfit(np.log(n_steps_grid), np.log(stds_near), 1)
    print(f"pente log-log pres de la barriere = {slope:.3f} (attendu ~ 0, PAS -0.5)")
    assert -0.15 < slope < 0.15


# ---------------------------------------------------------------------------
# QUÊTE 4.6 — delta-vega sous Heston
# ---------------------------------------------------------------------------


def test_heston_delta_bs_hedge_error_biaise():
    """Hedge delta BS seul sous Heston : erreur non nulle, à variance notable.

    Pas d'identité fermée ici (c'est tout le point de la 4.6) -- juste la
    garantie que la fonction tourne, rend la bonne forme, et une variance
    largement non nulle (sinon le test de reduction de variance qui suit
    n'aurait aucun sens : il n'y aurait rien a reduire).
    """
    S0, K, r, T = HESTON_HP["S0"], HP["K"], HESTON_HP["r"], HESTON_HP["T"]
    sigma_hedge = np.sqrt(HESTON_HP["theta"])
    N, n_steps = 20_000, 50

    S, _ = heston_paths(**HESTON_HP, n_steps=n_steps, N=N,
                         rng=np.random.default_rng(30))
    V0 = put_bs(S0, K, sigma_hedge, r, T)

    err = hedging.heston_delta_bs_hedge_error(S, K, sigma_hedge, r, T, V0, option="put")

    assert err.shape == (N,)
    assert err.std(ddof=1) > 0.5


def test_heston_delta_vega_reduit_la_variance():
    """4.6, la validation qui ferme la quête et l'acte.

    Overlay vega statique sur K_vega=110, dimensionne par le ratio des vegas
    BS a sigma_hedge. Sur les MEMES trajectoires Heston (CRN), le ratio de
    variance (delta seul / delta+vega) doit etre significativement > 1 --
    mesure empiriquement ~3.0, seuil fixe a 1.5 pour laisser de la marge au
    bruit MC.
    """
    S0, K, K_vega, r, T = HESTON_HP["S0"], HP["K"], 110.0, HESTON_HP["r"], HESTON_HP["T"]
    sigma_hedge = np.sqrt(HESTON_HP["theta"])
    N, n_steps = 30_000, 50

    S, _ = heston_paths(**HESTON_HP, n_steps=n_steps, N=N,
                         rng=np.random.default_rng(31))
    V0 = put_bs(S0, K, sigma_hedge, r, T)

    err_delta = hedging.heston_delta_bs_hedge_error(S, K, sigma_hedge, r, T, V0, option="put")
    err_delta_vega = hedging.heston_delta_vega_hedge_error(
        S, K, K_vega, sigma_hedge, r, T, V0, option="put")

    ratio = err_delta.var(ddof=1) / err_delta_vega.var(ddof=1)
    print(f"Var(delta seul)={err_delta.var(ddof=1):.4f}  "
          f"Var(delta+vega)={err_delta_vega.var(ddof=1):.4f}  ratio={ratio:.3f}")

    assert ratio > 1.5


# ---------------------------------------------------------------------------
# BOSS 4 — convergence et P&L de réplication
# ---------------------------------------------------------------------------


def test_boss4_artefacts():
    """Vérifie les artefacts produits par scripts/boss4_hedging.py.

    Run: .venv/bin/python scripts/boss4_hedging.py

    Pas de nouvelle identité numérique -- tout a déjà été validé quête par
    quête dans ce fichier. Ce test vérifie seulement que le script tourne, que
    le contrat de sortie (documenté en tête de boss4_hedging.py) est respecté,
    et que la pente mesurée sur "convergence" retombe dans la même fourchette
    que celle de 4.2.
    """
    fig = ROOT / "figures" / "boss4_hedging.png"
    res = ROOT / "figures" / "boss4_results.json"
    assert fig.exists(), "figure manquante"
    assert res.exists(), "resultats manquants"

    data = json.loads(res.read_text())

    convergence = data["convergence"]
    assert len(convergence) >= 5
    n_steps_list = [p["n_steps"] for p in convergence]
    assert n_steps_list == sorted(n_steps_list)
    stds = [p["std_error"] for p in convergence]
    assert all(a > b for a, b in zip(stds, stds[1:]))       # decroissant

    assert -0.8 < data["slope_measured"] < -0.2

    pnl = data["pnl"]
    assert set(pnl.keys()) == {"modele_correct", "modele_faux", "avec_couts"}
    for scenario in pnl.values():
        assert set(scenario.keys()) == {"mean", "std", "p5", "p50", "p95"}
        assert scenario["p5"] < scenario["p50"] < scenario["p95"]

    # "avec_couts" est une perte certaine : jamais centre en zero.
    assert pnl["avec_couts"]["mean"] < 0.0
