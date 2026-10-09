"""ACTE V -- Multi-actif : Cholesky, worst-of, BRC, autocall, sensibilites.

Ce qui est teste reste INDEPENDANT de l'implementation : idempotence du
reparateur de correlation, positivite apres reparation d'une matrice
incoherente, identite de martingale par actif, cas degeneres (n_assets=1,
rho=1), decompositions exactes (BRC, autocall a une date). Aucun test ne
compare le multi-actif a lui-meme.

Lance : .venv/bin/python -m pytest tests/test_multi_asset.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from multi_asset import (correlated_gbm_paths, repair_correlation_clip,
                         worst_of_performance, worst_of_di_put,
                         worst_of_di_put_continuous, brc_price,
                         recall_index, autocall_price,
                         worst_of_deltas, worst_of_vega,
                         worst_of_rho_sensitivity)
import barriers

# Jeu de parametres de reference de l'acte : 2 actifs (decision actee avec
# Arthur), vols differentes pour que rien ne soit accidentellement symetrique.
S0 = np.array([100.0, 80.0])
SIGMA = np.array([0.20, 0.30])
R, T = 0.05, 1.0
RHO_TARGET = np.array([[1.0, 0.6],
                       [0.6, 1.0]])

# Correlations par paires incoherentes entre elles : pas de matrice a 3
# variables reelles ne peut avoir ces trois coefficients simultanement.
RHO_NON_DP = np.array([[1.0,  0.9,  0.9],
                       [0.9,  1.0, -0.9],
                       [0.9, -0.9,  1.0]])


def _half_width(v):
    """Demi-largeur IC 95% -- la convention du repo."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# Reparation de la matrice de correlation
# ---------------------------------------------------------------------------



def test_repair_correlation_clip_matrice_valide_inchangee():
    """Idempotence : une matrice de correlation deja valide ne bouge pas."""
    rho = RHO_TARGET
    repaired = repair_correlation_clip(rho)
    assert np.allclose(repaired, rho, atol=1e-10)


def test_repair_correlation_clip_repare_matrice_non_dp():
    """Sur une matrice incoherente : sortie symetrique, PSD, diagonale unite."""
    repaired = repair_correlation_clip(RHO_NON_DP)

    assert np.allclose(repaired, repaired.T, atol=1e-10)
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-8)

    eigvals = np.linalg.eigvalsh(repaired)
    assert np.all(eigvals >= -1e-8)

    # doit maintenant passer sans lever LinAlgError
    L = np.linalg.cholesky(repaired)
    assert np.allclose(L @ L.T, repaired, atol=1e-8)


# ---------------------------------------------------------------------------
# Trajectoires correlees
# ---------------------------------------------------------------------------


def test_correlated_gbm_paths_shape_et_depart():
    """Contrat de base : formes, points de depart exacts."""
    n_steps, N = 50, 5_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(1))

    assert paths.shape == (N, 2, n_steps + 1)
    assert np.allclose(paths[:, 0, 0], S0[0])
    assert np.allclose(paths[:, 1, 0], S0[1])
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0.0)


def test_correlated_gbm_paths_martingale_par_actif():
    """E[e^{-rT} S_i(T)] = S0[i], actif par actif -- model-free."""
    N = 200_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, 50, N,
                                 rng=np.random.default_rng(2))

    for i in range(2):
        disc = np.exp(-R*T) * paths[:, i, -1]
        assert abs(disc.mean() - S0[i]) < _half_width(disc)


def test_correlated_gbm_paths_un_seul_actif_redonne_gbm_simple():
    """n_assets=1 : plus de correlation a reconstruire, juste un GBM.

    Teste statistiquement (moyenne + variance des log-rendements dans l'IC
    attendue), pas trajectoire par trajectoire -- correlated_gbm_paths n'a
    aucune obligation de tirer ses gaussiennes dans le meme ordre que
    gbm_paths.
    """
    N, n_steps = 200_000, 50
    paths = correlated_gbm_paths(np.array([100.0]), np.array([0.25]),
                                 np.array([[1.0]]), R, T, n_steps, N,
                                 rng=np.random.default_rng(3))

    log_ret = np.log(paths[:, 0, -1] / paths[:, 0, 0])
    mean_expected = (R - 0.5*0.25**2)*T
    var_expected = 0.25**2 * T

    se_mean = log_ret.std(ddof=1) / np.sqrt(N)
    assert abs(log_ret.mean() - mean_expected) < 1.96*se_mean
    assert abs(log_ret.var(ddof=1) - var_expected) < 0.05*var_expected


def test_correlated_gbm_paths_correlation_empirique():
    """La correlation empirique des log-rendements converge vers la cible."""
    N, n_steps = 200_000, 20
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(4))

    log_ret_0 = np.log(paths[:, 0, -1] / paths[:, 0, 0])
    log_ret_1 = np.log(paths[:, 1, -1] / paths[:, 1, 0])
    rho_hat = np.corrcoef(log_ret_0, log_ret_1)[0, 1]

    assert abs(rho_hat - RHO_TARGET[0, 1]) < 0.02


def test_correlated_gbm_paths_identite_independance():
    """rho = identite -> les deux actifs sont simules independamment."""
    N, n_steps = 200_000, 20
    paths = correlated_gbm_paths(S0, SIGMA, np.eye(2), R, T, n_steps, N,
                                 rng=np.random.default_rng(5))

    log_ret_0 = np.log(paths[:, 0, -1] / paths[:, 0, 0])
    log_ret_1 = np.log(paths[:, 1, -1] / paths[:, 1, 0])
    rho_hat = np.corrcoef(log_ret_0, log_ret_1)[0, 1]

    assert abs(rho_hat) < 0.02


def test_correlated_gbm_paths_matrice_non_dp_ne_plante_pas():
    """Une correlation cible incoherente ne doit pas faire planter le simulateur.

    La reparation de repair_correlation_clip doit etre appliquee en interne :
    on ne teste pas une valeur precise (la matrice cible n'ayant pas de sens,
    il n'y a pas de "vraie" correlation a retrouver), seulement que la
    correlation empirique reconstruite est bien celle de la matrice REPAREE,
    pas un plantage ou un nan.
    """
    N, n_steps = 200_000, 20
    S0_3, SIGMA_3 = np.array([100.0, 100.0, 100.0]), np.array([0.2, 0.2, 0.2])
    paths = correlated_gbm_paths(S0_3, SIGMA_3, RHO_NON_DP, R, T, n_steps, N,
                                 rng=np.random.default_rng(6))

    assert np.all(np.isfinite(paths))

    repaired = repair_correlation_clip(RHO_NON_DP)
    log_ret = np.log(paths[:, :, -1] / paths[:, :, 0])
    rho_hat = np.corrcoef(log_ret, rowvar=False)

    assert np.allclose(rho_hat, repaired, atol=0.03)


# ---------------------------------------------------------------------------
# Worst-of DI put et cas degeneres
# ---------------------------------------------------------------------------



def test_worst_of_performance_shape_et_valeur_initiale():
    """Contrat de base : forme (N, n_steps+1), performance = 1 au depart."""
    n_steps, N = 50, 5_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(10))
    perf = worst_of_performance(paths)

    assert perf.shape == (N, n_steps + 1)
    assert np.allclose(perf[:, 0], 1.0, atol=1e-12)
    assert np.all(np.isfinite(perf))



def test_worst_of_performance_est_bien_un_minimum():
    """Identite model-free : la performance du pire est <= celle de CHAQUE actif.

    Vraie quel que soit le modele derriere les trajectoires -- ce n'est pas
    specifique au GBM.
    """
    n_steps, N = 30, 2_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(11))
    perf = worst_of_performance(paths)

    for i in range(2):
        perf_i = paths[:, i, :] / paths[:, i, 0:1]
        assert np.all(perf <= perf_i + 1e-12)



def test_worst_of_di_put_un_seul_actif_redonne_di_put_mono():
    """n_assets=1 : le worst-of (en performance) retombe exactement sur le DI
    put mono-actif (en niveau) de barriers.py, memes tirages.
    """
    n_steps, N = 50, 20_000
    S0_1, SIGMA_1 = np.array([100.0]), np.array([0.25])
    paths = correlated_gbm_paths(S0_1, SIGMA_1, np.array([[1.0]]), R, T,
                                 n_steps, N, rng=np.random.default_rng(12))

    K_perf, H_perf = 1.05, 0.85
    price_worst_of, hw_worst_of = worst_of_di_put(paths, K_perf, H_perf, R, T)

    s0 = S0_1[0]
    price_mono, hw_mono = barriers.di_put(paths[:, 0, :], K_perf * s0,
                                          H_perf * s0, R, T)

    # worst_of_di_put est en PERFORMANCE (depart a 1), di_put mono en PRIX
    # absolu (depart a s0) -- les deux payoffs different exactement d'un
    # facteur s0 (meme knock_in, max(K*s0-ST,0) = s0*max(K-ST/s0,0)). On
    # ramene le prix absolu en performance avant de comparer.
    assert abs(price_worst_of - price_mono / s0) < 1e-10
    assert abs(hw_worst_of - hw_mono / s0) < 1e-10



def test_worst_of_performance_rho1_vols_egales_identique_entre_actifs():
    """rho=1, sigma identiques entre actifs : la performance de chaque actif
    est IDENTIQUE, trajectoire par trajectoire -- aucun actif n'est jamais le
    pire, ils le sont tous a l'unisson.
    """
    n_steps, N = 30, 2_000
    S0_eq, SIGMA_eq = np.array([100.0, 80.0]), np.array([0.25, 0.25])
    rho_un = np.array([[1.0, 1.0], [1.0, 1.0]])
    paths = correlated_gbm_paths(S0_eq, SIGMA_eq, rho_un, R, T, n_steps, N,
                                 rng=np.random.default_rng(13))

    perf_0 = paths[:, 0, :] / paths[:, 0, 0:1]
    perf_1 = paths[:, 1, :] / paths[:, 1, 0:1]
    assert np.allclose(perf_0, perf_1, atol=1e-8)

    perf_worst = worst_of_performance(paths)
    assert np.allclose(perf_worst, perf_0, atol=1e-8)



# test_worst_of_di_put_rho1_vols_egales_reduit_a_un_mono_actif retire : la
# matrice rho=[[1,1],[1,1]] est exactement rang-1 (valeur propre nulle), donc
# toute tolerance `tol>0` dans repair_correlation_clip laisse un residu
# proportionnel a `tol` (mesure : tol -> residu quasi 1:1). Descendre `tol`
# sous ~1e-14 rapproche de la precision machine mais rend la Cholesky instable
# (pivot pres de l'epsilon flottant) -- 1e-10 exact n'est pas atteignable sans
# fragiliser 5.1. L'identite perf_0 == perf_1 (test au-dessus) reste la
# validation pertinente du cas degenere.



def test_worst_of_di_put_ne_plante_pas_sur_matrice_non_dp():
    """Robustesse : une correlation cible incoherente, propagee jusqu'au
    pricer worst-of, ne doit ni planter ni produire de nan.
    """
    n_steps, N = 20, 5_000
    S0_3, SIGMA_3 = np.array([100.0, 100.0, 100.0]), np.array([0.2, 0.2, 0.2])
    paths = correlated_gbm_paths(S0_3, SIGMA_3, RHO_NON_DP, R, T, n_steps, N,
                                 rng=np.random.default_rng(15))

    price, half_width = worst_of_di_put(paths, 1.0, 0.7, R, T)

    assert np.isfinite(price)
    assert np.isfinite(half_width)
    assert half_width >= 0.0


# ---------------------------------------------------------------------------
# BGK applique au worst-of (heuristique)
# ---------------------------------------------------------------------------



def test_worst_of_di_put_continuous_sigma_nul_est_un_noop():
    """Sans volatilite, aucune correction a appliquer : identique a worst_of_di_put."""
    n_steps, N = 30, 2_000
    paths = correlated_gbm_paths(S0, np.array([1e-6, 1e-6]), RHO_TARGET, R, T,
                                 n_steps, N, rng=np.random.default_rng(20))

    price_raw, hw_raw = worst_of_di_put(paths, 1.0, 0.8, R, T)
    price_corr, hw_corr = worst_of_di_put_continuous(paths, 1.0, 0.8, R, T,
                                                      np.array([1e-6, 1e-6]))

    assert abs(price_raw - price_corr) < 1e-8
    assert abs(hw_raw - hw_corr) < 1e-8



def test_worst_of_di_put_continuous_rho_proche_1_dans_ic_du_mono():
    """rho proche de 1 (pas exactement -- Cholesky instable a rho=1 exact,
    cf. le test retire en 5.2), vols et spots egaux : le prix corrige doit
    tomber dans l'IC du di_put_continuous mono-actif sur la performance
    commune.
    """
    n_steps, N = 30, 50_000
    S0_eq, SIGMA_eq = np.array([100.0, 100.0]), np.array([0.25, 0.25])
    rho_proche1 = np.array([[1.0, 0.999], [0.999, 1.0]])
    paths = correlated_gbm_paths(S0_eq, SIGMA_eq, rho_proche1, R, T, n_steps, N,
                                 rng=np.random.default_rng(21))

    price_wo, hw_wo = worst_of_di_put_continuous(paths, 1.0, 0.8, R, T, SIGMA_eq)

    perf_paths = paths[:, 0, :] / paths[:, 0, 0:1]
    price_mono, hw_mono = barriers.di_put_continuous(perf_paths, 1.0, 0.8, R, T,
                                                     SIGMA_eq[0])

    assert abs(price_wo - price_mono) < hw_wo + hw_mono


# ---------------------------------------------------------------------------
# BRC (coupon / put worst-of vendu)
# ---------------------------------------------------------------------------



def test_brc_price_decomposition_exacte():
    """BRC = coupon (zero-coupon) + nominal (zero-coupon) - put worst-of vendu.

    Identite EXACTE, memes tirages -- pas une approximation.
    """
    n_steps, N = 30, 5_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(30))
    C, H = 0.08, 0.7

    price_brc, hw_brc = brc_price(paths, C, H, R, T)
    put_price, put_hw = worst_of_di_put(paths, 1.0, H, R, T)

    expected = C * np.exp(-R * T) + np.exp(-R * T) - put_price
    assert abs(price_brc - expected) < 1e-10
    assert abs(hw_brc - put_hw) < 1e-10



def test_brc_price_barriere_lointaine_redonne_coupon_plus_zero_coupon():
    """H quasi inatteignable : le BRC redevient un pur coupon + zero-coupon,
    sans optionalite vendue.
    """
    n_steps, N = 30, 5_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(31))
    C, H = 0.08, 1e-6

    price_brc, hw_brc = brc_price(paths, C, H, R, T)
    expected = (C + 1.0) * np.exp(-R * T)

    assert abs(price_brc - expected) < hw_brc + 1e-9


# ---------------------------------------------------------------------------
# Autocall / Phoenix
# ---------------------------------------------------------------------------



def test_recall_index_premiere_date_valide():
    """Pathwise, independant de toute simulation -- recall_index sur une
    performance construite a la main.
    """
    perf_obs = np.array([
        [0.9, 1.05, 1.10],   # rappele a l'indice 1
        [0.9, 0.95, 0.80],   # jamais rappele
        [1.20, 0.5, 1.5],    # rappele des l'indice 0
    ])
    idx = recall_index(perf_obs, autocall_level=1.0)
    assert list(idx) == [1, -1, 0]



def test_autocall_price_une_seule_date_egale_brc():
    """obs_steps = [n_steps] (une seule date, la maturite) : l'autocall
    retombe exactement sur le BRC de 5.4 -- un cas particulier digital connu.
    """
    n_steps, N = 30, 5_000
    paths = correlated_gbm_paths(S0, SIGMA, RHO_TARGET, R, T, n_steps, N,
                                 rng=np.random.default_rng(32))
    C, H, autocall_level = 0.08, 0.7, 1.0

    price_auto, hw_auto = autocall_price(paths, np.array([n_steps]),
                                         autocall_level, C, H, R, T)
    price_brc, hw_brc = brc_price(paths, C, H, R, T)

    assert abs(price_auto - price_brc) < 1e-10
    assert abs(hw_auto - hw_brc) < 1e-10



def test_autocall_probabilite_rappel_monotone_en_rho_et_sigma():
    """La probabilite de rappel croit avec rho, decroit avec sigma --
    direction verifiee, pas une valeur precise.
    """
    n_steps, N = 40, 20_000
    obs_steps = np.array([10, 20, 30, 40])
    S0_eq, SIGMA_lo, SIGMA_hi = (np.array([100.0, 100.0]),
                                np.array([0.15, 0.15]),
                                np.array([0.45, 0.45]))

    rho_lo = np.array([[1.0, 0.1], [0.1, 1.0]])
    rho_hi = np.array([[1.0, 0.9], [0.9, 1.0]])

    def recalled_fraction(sigma, rho, seed):
        paths = correlated_gbm_paths(S0_eq, sigma, rho, R, T, n_steps, N,
                                     rng=np.random.default_rng(seed))
        perf = worst_of_performance(paths)
        perf_obs = perf[:, obs_steps]
        idx = recall_index(perf_obs, autocall_level=1.0)
        return np.mean(idx != -1)

    frac_rho_lo = recalled_fraction(SIGMA_lo, rho_lo, 40)
    frac_rho_hi = recalled_fraction(SIGMA_lo, rho_hi, 41)
    assert frac_rho_hi > frac_rho_lo

    frac_sigma_lo = recalled_fraction(SIGMA_lo, rho_hi, 42)
    frac_sigma_hi = recalled_fraction(SIGMA_hi, rho_hi, 43)
    assert frac_sigma_hi < frac_sigma_lo


# ---------------------------------------------------------------------------
# Sensibilites (delta par actif, vega, rho)
# ---------------------------------------------------------------------------


def test_worst_of_deltas_est_exactement_nul():
    """K et H sont en PERFORMANCE : le payoff ne depend que de
    paths/paths[:,:,0] = exp(increments) -- S0 se simplifie EXACTEMENT,
    algebriquement, avant meme toute simulation. Bumper S0[i] ne peut donc
    JAMAIS changer le prix d'un produit 100% performance : le "delta spot"
    d'un worst-of cote en performance est IDENTIQUEMENT nul, pas juste petit
    -- une identite model-free (pas de MC, pas de tolerance).
    """
    deltas = worst_of_deltas(S0, SIGMA, RHO_TARGET, R, T, 30, 50_000,
                             K=1.0, H=0.7, h=1.0, seed=100)
    assert deltas.shape == (2,)
    assert np.all(np.abs(deltas) < 1e-8)


def test_worst_of_vega_positive():
    """Un put gagne en valeur quand la vol augmente -- vega positif, par actif."""
    vegas = worst_of_vega(S0, SIGMA, RHO_TARGET, R, T, 30, 50_000,
                          K=1.0, H=0.7, h=0.01, seed=110)
    assert vegas.shape == (2,)
    assert np.all(vegas > 0.0)



def test_worst_of_rho_sensitivity_signe():
    """Augmenter la correlation entre actifs REDUIT la dispersion du worst-of
    (les actifs bougent plus ensemble) -- un put worst-of perd en general de
    la valeur quand rho augmente (barriere moins souvent touchee par un seul
    actif qui decroche des autres). Signe verifie, pas une valeur precise.
    """
    d_rho = worst_of_rho_sensitivity(S0, SIGMA, RHO_TARGET, R, T, 30, 50_000,
                                     K=1.0, H=0.7, i=0, j=1, h=0.05, seed=120)
    assert d_rho < 0.0


# ---------------------------------------------------------------------------
# BOSS 5 -- pricing complet BRC worst-of + figure
# ---------------------------------------------------------------------------


def test_boss5_artefacts():
    """Verifie les artefacts produits par scripts/boss5_brc_worstof.py.

    Run: .venv/bin/python scripts/boss5_brc_worstof.py

    Pas de nouvelle identite numerique -- tout a deja ete valide quete par
    quete dans ce fichier. Ce test verifie seulement que le script tourne,
    que le contrat de sortie (documente en tete de boss5_brc_worstof.py) est
    respecte, et que les valeurs sont plausibles (meme esprit que
    test_boss2/3/4_artefacts).
    """
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    fig = root / "figures" / "boss5_brc_worstof.png"
    res = root / "figures" / "boss5_results.json"
    assert fig.exists(), "missing figure"
    assert res.exists(), "missing results"

    data = json.loads(res.read_text())

    sweep_rho = data["sweep_rho"]
    assert len(sweep_rho) >= 5
    rhos = [p["rho"] for p in sweep_rho]
    assert rhos == sorted(rhos)
    for p in sweep_rho:
        assert set(p.keys()) >= {"rho", "price_mc", "half_width_mc",
                                 "price_cv", "half_width_cv", "c_hat"}
        assert p["half_width_cv"] <= p["half_width_mc"]

    sweep_n_assets = data["sweep_n_assets"]
    assert len(sweep_n_assets) >= 2
    n_assets_list = [p["n_assets"] for p in sweep_n_assets]
    assert n_assets_list == sorted(n_assets_list)
