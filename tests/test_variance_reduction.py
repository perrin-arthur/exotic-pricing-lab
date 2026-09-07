"""ACTE II — reduction de variance. Tests ecrits AVANT le code.

MECANIQUE DU JEU
----------------
Tous les tests sont marques @pytest.mark.xfail(strict=True) :
  * tant que la fonction leve NotImplementedError -> XFAIL (attendu, suite verte)
  * des que ton implementation est correcte -> XPASS -> et strict=True fait
    ECHOUER la suite.
Cet echec est le signal de victoire : tu retires le marqueur xfail de CE test,
et la quete est validee. Retirer un marqueur sans avoir implemente = tricher
contre toi-meme, le test passera au rouge franc.

Lance : .venv/bin/python -m pytest tests/test_variance_reduction.py -v
Etat initial attendu : que des xfailed, zero passed, zero xpassed.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np
import pytest

import barriers
import mc_engine
from bs import put_bs
from mc_engine import (control_variate, control_variate_antithetic,
                       gbm_paths, gbm_paths_antithetic, pilot_c)

PARAMS = dict(S0=100, K=100, sigma=0.2, r=0.05, T=1)
PATH_PARAMS = dict(S0=100, sigma=0.2, r=0.05, T=1)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _echantillon_correle(n, rng):
    """(Y, X, EX, EY) synthetiques : aucune finance, moments connus EXACTEMENT.

    X = 1 + Z1                      -> E[X] = 1
    Y = 3 + 2*Z1 + 0.5*Z2           -> E[Y] = 3, Cov(Y,X) = 2, Var(X) = 1
    donc c* = 2 et rho* = 2/sqrt(4.25) = 0.97014...
    """
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    return 3.0 + 2.0*Z1 + 0.5*Z2, 1.0 + Z1, 1.0, 3.0


def _half_width(v):
    """Demi-largeur IC 95% d'un echantillon i.i.d. — la convention du repo."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# QUETE 2.1 — control_variate                                          [30 XP]
# ---------------------------------------------------------------------------


def test_cv_unbiased():
    """L'estimateur tombe dans son PROPRE IC autour de la vraie valeur E[Y]=3,
    et sa demi-largeur est strictement plus petite que celle du MC brut."""
    n = 20_000
    Y, X, EX, EY = _echantillon_correle(n, np.random.default_rng(42))
    est, hw, c_hat, rho_hat = control_variate(Y, X, EX)

    assert abs(est - EY) < hw
    assert hw < _half_width(Y)
    assert abs(c_hat - 2.0) < 0.05                 # c* = 2
    assert abs(rho_hat - 2/np.sqrt(4.25)) < 0.02   # rho* = 0.9701
    assert -1.0 <= rho_hat <= 1.0


def test_cv_c_zero_reproduit_le_mc_brut():
    """c = 0 doit redonner EXACTEMENT le Monte-Carlo brut sur Y.

    Cas de base : si celui-la casse, l'erreur n'est pas dans l'estimation de c,
    elle est dans la plomberie (moyenne, demi-largeur, ordre du tuple).
    """
    n = 5_000
    Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(0))
    est, hw, c_hat, _ = control_variate(Y, X, EX, c=0.0)

    assert abs(est - Y.mean()) < 1e-12
    assert abs(hw - _half_width(Y)) < 1e-12
    assert abs(c_hat) < 1e-15          # le c IMPOSE est renvoye tel quel


def test_cv_utilise_bien_EX():
    """PIEGE PARENTHESAGE : Y - c*X - EX au lieu de Y - c*(X - EX).

    EX = 1000 rend l'erreur impossible a rater : la version fautive decale le
    prix de ~3000. Le second assert attrape en plus l'erreur de SIGNE
    Y + c*(X - EX), qui double la variance au lieu de la reduire.
    """
    n = 20_000
    rng = np.random.default_rng(1)
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    X = 1000.0 + Z1
    Y = 3.0 + 2.0*Z1 + 0.5*Z2

    est, hw, _, _ = control_variate(Y, X, EX=1000.0)

    assert abs(est - 3.0) < 0.05
    assert hw < _half_width(Y)


def test_cv_half_width_sur_le_residu():
    """PIEGE : demi-largeur calculee sur Y au lieu du residu Z = Y - c(X - EX).

    Ici Y et X sont quasi colineaires : le residu est presque constant, donc la
    demi-largeur doit s'effondrer. Calculee sur Y, elle ne bougerait pas d'un
    poil et le gain de variance serait invisible — alors que c'est TOUT l'objet
    de la technique.
    """
    n = 10_000
    rng = np.random.default_rng(2)
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    X = Z1
    Y = 5.0 + Z1 + 1e-6*Z2

    est, hw, _, _ = control_variate(Y, X, EX=0.0)

    assert hw < 0.01 * _half_width(Y)
    assert abs(est - 5.0) < 1e-4


def test_cv_rho_invariant_par_echelle():
    """PIEGE PRECEDENCE : rho = cov/(sd_Y*sd_X), pas cov/sd_Y*sd_X.

    Python lit la version fautive comme (cov/sd_Y)*sd_X. Tant que les echelles
    valent 1 personne ne voit rien ; on multiplie X par 1000 et rho explose
    hors de [-1, 1]. Meme famille d'erreur que le /2*h du delta.

    Au passage : le controle est invariant par changement d'echelle du controle
    (c absorbe le facteur), donc estimate et demi-largeur ne doivent PAS bouger.
    """
    n = 10_000
    Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(3))

    est1, hw1, c1, rho1 = control_variate(Y, X, EX)
    est2, hw2, c2, rho2 = control_variate(Y, 1000.0*X, 1000.0*EX)

    assert abs(rho2 - rho1) < 1e-10
    assert -1.0 <= rho2 <= 1.0
    assert abs(1000.0*c2 - c1) < 1e-8
    assert abs(est2 - est1) < 1e-9
    assert abs(hw2 - hw1) < 1e-9


def test_cv_retourne_des_floats():
    """4 floats PYTHON, pas des np.float64 ni des arrays 0-d.

    Ce n'est pas du purisme : le BOSS 2 serialise ces valeurs en JSON, et
    json.dump refuse np.float64. Tout l'Acte I caste deja (float(disc.mean())).
    """
    Y, X, EX, _ = _echantillon_correle(500, np.random.default_rng(4))
    out = control_variate(Y, X, EX)

    assert isinstance(out, tuple) and len(out) == 4
    for v in out:
        assert type(v) is float
    json.dumps(list(out))          # doit passer sans TypeError


def test_cv_pas_de_N_fantome(monkeypatch):
    """PIEGE GLOBALE FANTOME (deja arrive deux fois) : n = len(Y), point.

    Deux verifications :
      1. la demi-largeur doit scaler en 1/sqrt(n) — un N capte dans le module
         la fige et le ratio s'ecroule ;
      2. planter un mc_engine.N = 999_999 ne doit RIEN changer au resultat.
    L'appel se fait depuis une fonction locale, sans aucune variable ambiante
    qui pourrait etre capturee par accident.
    """
    def run(n, seed):
        Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(seed))
        return control_variate(Y, X, EX)

    hw_petit = run(137, 1)[1]
    hw_grand = run(137*4, 1)[1]
    assert 1.7 < hw_petit/hw_grand < 2.3       # facteur 2 attendu

    avant = run(137, 1)
    monkeypatch.setattr(mc_engine, "N", 999_999, raising=False)
    apres = run(137, 1)
    assert avant == apres


# ---------------------------------------------------------------------------
# QUETE 2.2 — boss de tutoriel : le cas degenere                       [20 XP]
# ---------------------------------------------------------------------------

def test_cv_degenere_Y_egal_X():
    """Y = X = put vanille, EX = prix BS ferme.

    Le controle connait alors PARFAITEMENT l'erreur commise : il la retranche
    en entier et l'estimateur redonne le prix ferme, avec une demi-largeur
    numeriquement nulle. Le Monte-Carlo a disparu.

    Tolerance 1e-12 et pas un IC : c'est une identite algebrique, aucun
    argument statistique n'intervient. Si tu es tente d'ecrire "< ci*1.5" ici,
    c'est que tu n'as pas vu ce qui se passe.
    """
    N = 20_000
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=N,
                      rng=np.random.default_rng(7))
    disc = np.exp(-PARAMS["r"]*PARAMS["T"]) * np.maximum(
        PARAMS["K"] - paths[:, -1], 0.0)
    EX = put_bs(**PARAMS)

    est, hw, c_hat, rho_hat = control_variate(disc, disc, EX)

    assert abs(est - EX) < 1e-12
    assert hw < 1e-12
    assert abs(c_hat - 1.0) < 1e-10
    assert abs(rho_hat - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# QUETE 2.3 — branchement sur le DI put                                [40 XP]
# ---------------------------------------------------------------------------

def test_di_put_payoffs_valeurs_a_la_main():
    """Trajectoires jouet, resultat calcule a la main. Deux pieges vises.

    PIEGE np.max vs np.maximum : np.max(K - ST) renvoie UN scalaire (le max de
    tout le tableau) -> les trois chemins recevraient le meme payoff.
    PIEGE axe du min : paths.min() sans axis=1 renvoie le min GLOBAL -> le
    chemin 1, qui ne touche jamais, serait compte comme touche.

    Aucune trajectoire ne touche exactement H : la convention (< ou <=) n'est
    pas testee ici, elle doit juste rester celle de di_put.
    """
    K, H, r, T = 100.0, 85.0, 0.10, 2.0
    paths = np.array([
        [100.,  95.,  80.,  90.],   # min=80  <= 85 : touche   -> put = 10
        [100.,  98.,  96.,  94.],   # min=94  >  85 : intacte  -> put = 6 mais Y=0
        [100.,  90.,  84., 130.],   # min=84  <= 85 : touche   -> put = 0
    ])
    disc = np.exp(-r*T)

    Y, X = barriers.di_put_payoffs(paths, K=K, H=H, r=r, T=T)

    assert Y.shape == (3,) and X.shape == (3,)
    assert np.allclose(X, disc*np.array([10.0, 6.0, 0.0]), atol=1e-12)
    assert np.allclose(Y, disc*np.array([10.0, 0.0, 0.0]), atol=1e-12)


def test_di_put_payoffs_actualisation_coherente():
    """PIEGE ACTUALISATION : Y, X et EX doivent vivre dans la MEME unite.

    EX = put_bs est un prix actualise. Si X sort non actualise, X.mean() - EX
    ne mesure plus une erreur mais un ecart d'unite, et le controle DECALE le
    prix au lieu de le stabiliser. On verrouille en exigeant que les moyennes
    des deux vecteurs redonnent exactement les pricers de l'Acte I.
    """
    N = 20_000
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=N,
                      rng=np.random.default_rng(42))
    Y, X = barriers.di_put_payoffs(paths, K=100, H=90, r=0.05, T=1.0)

    assert Y.shape == X.shape == (N,)
    assert abs(Y.mean() - barriers.di_put(paths, K=100, H=90, r=0.05, T=1.0)[0]) < 1e-12
    assert abs(X.mean() - barriers.van_put(paths, K=100, r=0.05, T=1.0)[0]) < 1e-12
    # le controle est le put vanille : sa moyenne doit etre proche du BS ferme
    assert abs(X.mean() - put_bs(**PARAMS)) < _half_width(X) * 1.5


def test_di_put_cv_accord_avec_mc_brut():
    """Meme prix que le MC brut (a l'IC pres), sur les MEMES trajectoires.

    L'ecart entre les deux estimateurs vaut exactement c*(X_barre - EX) : il est
    de l'ordre de l'IC du CONTROLE, pas plus. Un ecart plus grand = le controle
    n'est pas centre (EX faux, mauvais sigma, mauvaise actualisation).
    """
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=50_000,
                      rng=np.random.default_rng(11))
    mc, ci_mc = barriers.di_put(paths, K=100, H=90, r=0.05, T=1.0)
    _, ci_van = barriers.van_put(paths, K=100, r=0.05, T=1.0)
    est, hw, c_hat, rho_hat = barriers.di_put_cv(paths, K=100, H=90,
                                                 r=0.05, T=1.0, sigma=0.2)

    assert abs(est - mc) < ci_van
    assert 0.0 < rho_hat < 1.0
    assert 0.0 < c_hat < 2.0
    assert hw < ci_mc


def test_di_put_cv_reduit_la_variance():
    """Le gain doit exister, et ne JAMAIS se retourner en perte.

    H = 95% du spot : le DI put est presque le put vanille, rho est enorme, on
    exige au moins un facteur 2 sur la demi-largeur.
    H = 60% du spot : le DI put ne ressemble plus a grand-chose au vanille, rho
    s'effondre — mais un c optimal ne peut pas degrader (au pire c -> 0). Le
    ratio doit rester sous 1. C'est la question d'entretien : "et si le controle
    est mal choisi ?"
    """
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=50_000,
                      rng=np.random.default_rng(12))
    ratios = {}
    for H in (95.0, 60.0):
        _, ci_mc = barriers.di_put(paths, K=100, H=H, r=0.05, T=1.0)
        _, hw, _, rho = barriers.di_put_cv(paths, K=100, H=H, r=0.05, T=1.0,
                                           sigma=0.2)
        ratios[H] = hw / ci_mc
        print(f"H={H:5.1f}  rho={rho:6.3f}  ratio IC={ratios[H]:6.3f}")

    assert ratios[95.0] < 0.5
    assert ratios[60.0] <= 1.0


# ---------------------------------------------------------------------------
# QUETE 2.4 — antithetiques x controle                            [20+20 XP]
# ---------------------------------------------------------------------------


def test_gbm_paths_antithetic_partage_Z():
    """Identite EXACTE : log(up) + log(down) ne depend pas du tirage.

    Les deux branches ne different que par le signe du terme en sigma, donc
    leur somme en log est la trajectoire de drift deterministe, doublee. Deux
    appels separes a standard_normal (= deux tirages independants, zero
    antithetique) cassent cette identite immediatement.
    """
    n_steps, N = 10, 1_000
    up, down = gbm_paths_antithetic(**PATH_PARAMS, n_steps=n_steps, N=N,
                                    rng=np.random.default_rng(3))

    assert up.shape == down.shape == (N, n_steps + 1)
    assert np.all(up[:, 0] == PATH_PARAMS["S0"])
    assert np.all(down[:, 0] == PATH_PARAMS["S0"])
    assert not np.allclose(up, down)          # sinon Z = 0 partout, pas d'alea

    t = np.linspace(0.0, PATH_PARAMS["T"], n_steps + 1)
    attendu = 2.0*(np.log(PATH_PARAMS["S0"])
                   + (PATH_PARAMS["r"] - 0.5*PATH_PARAMS["sigma"]**2)*t)
    assert np.max(np.abs(np.log(up) + np.log(down) - attendu)) < 1e-10



def test_cv_antithetic_N_est_le_nombre_de_paires():
    """PIEGE : diviser par 2N au lieu de N -> demi-largeur fausse d'un sqrt(2).

    Les 2N tirages ne sont pas 2N observations independantes. On forme les
    paires D'ABORD (l'echantillon i.i.d. a N points), on applique le controle
    ENSUITE. Ce test pose l'egalite exacte avec cette definition : si tes quatre
    sorties ne collent pas au 1e-12, c'est l'ordre des operations qui est faux.
    """
    n = 8_000
    rng = np.random.default_rng(5)
    Z = rng.standard_normal(n)
    X_up, X_down = np.exp(Z), np.exp(-Z)
    EX = float(np.exp(0.5))                      # E[e^Z] exact
    Y_up = np.maximum(X_up - 1.0, 0.0)
    Y_down = np.maximum(X_down - 1.0, 0.0)

    attendu = control_variate(0.5*(Y_up + Y_down), 0.5*(X_up + X_down), EX)
    obtenu = control_variate_antithetic(Y_up, Y_down, X_up, X_down, EX)

    for a, b in zip(attendu, obtenu):
        assert abs(a - b) < 1e-12


def test_cv_antithetic_domine_chaque_technique_seule():
    """A budget de trajectoires EGAL (2N), la combinaison bat chaque technique.

    Comparer des demi-largeurs a budget different n'a aucun sens : 2N chemins
    partout. Le put vanille sert de Y ici (payoff monotone en Z, cas ou les
    antithetiques donnent leur maximum) et le controle est le forward
    actualise, dont l'esperance est connue exactement : E[e^{-rT} S_T] = S0.
    """
    N = 25_000                                   # -> 2N = 50 000 trajectoires
    S0, K, r, T = 100.0, 100.0, 0.05, 1.0
    disc = np.exp(-r*T)

    up, down = gbm_paths_antithetic(**PATH_PARAMS, n_steps=25, N=N,
                                    rng=np.random.default_rng(21))
    plat = gbm_paths(**PATH_PARAMS, n_steps=25, N=2*N,
                     rng=np.random.default_rng(22))

    put = lambda p: disc*np.maximum(K - p[:, -1], 0.0)
    fwd = lambda p: disc*p[:, -1]

    hw_brut = _half_width(put(plat))
    hw_av = _half_width(0.5*(put(up) + put(down)))
    _, hw_cv, _, _ = control_variate(put(plat), fwd(plat), S0)
    _, hw_both, _, _ = control_variate_antithetic(put(up), put(down),
                                                  fwd(up), fwd(down), S0)
    print(f"brut={hw_brut:.5f}  AV={hw_av:.5f}  CV={hw_cv:.5f}  AV+CV={hw_both:.5f}")

    assert hw_both < hw_av
    assert hw_both < hw_cv
    assert hw_both < hw_brut


# ---------------------------------------------------------------------------
# QUETE 2.5 — c figé sur run pilote                                    [20 XP]
# ---------------------------------------------------------------------------


def test_pilot_c_est_un_scalaire():
    """pilot_c ne connait ni EX ni l'actualisation : c'est un ratio de moments.

    Deux invariances qui le prouvent :
      * translater Y d'une constante ne change pas c (moments CENTRES — si tu
        oublies de centrer, ca saute ici) ;
      * multiplier X par a divise c par a.
    """
    Y, X, _, _ = _echantillon_correle(5_000, np.random.default_rng(6))
    c = pilot_c(Y, X)

    assert type(c) is float
    assert abs(pilot_c(Y + 7.0, X) - c) < 1e-10
    assert abs(pilot_c(Y, 3.0*X) - c/3.0) < 1e-10


def test_pilot_c_proche_du_c_plein():
    """Un pilote de 10 000 chemins suffit : c n'a pas besoin d'etre precis.

    C'est le point cle a savoir dire en entretien — une erreur sur c ne
    biaise pas le prix, elle rogne seulement une partie du gain de variance
    (la variance est quadratique en c autour de son optimum, donc plate).
    """
    pilote = gbm_paths(**PATH_PARAMS, n_steps=50, N=10_000,
                       rng=np.random.default_rng(31))
    plein = gbm_paths(**PATH_PARAMS, n_steps=50, N=100_000,
                      rng=np.random.default_rng(32))

    Yp, Xp = barriers.di_put_payoffs(pilote, K=100, H=90, r=0.05, T=1.0)
    c_pilote = pilot_c(Yp, Xp)
    _, _, c_plein, _ = barriers.di_put_cv(plein, K=100, H=90, r=0.05, T=1.0,
                                          sigma=0.2)
    print(f"c_pilote={c_pilote:.4f}  c_plein={c_plein:.4f}")

    assert abs(c_pilote - c_plein) / abs(c_plein) < 0.05


def test_pilot_c_elimine_le_biais():
    """c estime sur un echantillon INDEPENDANT -> estimateur exactement centre.

    Controle de sanite : 8 repetitions independantes avec le c pilote fige,
    moyennees, doivent retomber sur une reference haute precision. Ce test ne
    "prouve" pas l'absence de biais O(1/N) — il est trop petit pour ca ; la
    preuve est theorique (c fige est deterministe, donc E[c(X_barre-EX)] = 0).
    Il attrape en revanche toute reutilisation des chemins pilotes.
    """
    ref_paths = gbm_paths(**PATH_PARAMS, n_steps=20, N=200_000,
                          rng=np.random.default_rng(99))
    ref, ci_ref = barriers.di_put(ref_paths, K=100, H=90, r=0.05, T=1.0)

    pilote = gbm_paths(**PATH_PARAMS, n_steps=20, N=10_000,
                       rng=np.random.default_rng(100))
    Yp, Xp = barriers.di_put_payoffs(pilote, K=100, H=90, r=0.05, T=1.0)
    c = pilot_c(Yp, Xp)

    estimations, hws = [], []
    for k in range(8):
        paths = gbm_paths(**PATH_PARAMS, n_steps=20, N=50_000,
                          rng=np.random.default_rng(200 + k))
        est, hw, _, _ = barriers.di_put_cv(paths, K=100, H=90, r=0.05, T=1.0,
                                           sigma=0.2, c=c)
        estimations.append(est)
        hws.append(hw)

    moyenne = float(np.mean(estimations))
    hw_moyenne = float(np.mean(hws)) / np.sqrt(len(estimations))
    print(f"ref={ref:.5f}+/-{ci_ref:.5f}  moyenne CV={moyenne:.5f}+/-{hw_moyenne:.5f}")

    assert abs(moyenne - ref) < np.hypot(ci_ref, hw_moyenne) * 1.5


# ---------------------------------------------------------------------------
# BOSS 2 — balayage de barriere                                        [80 XP]
# ---------------------------------------------------------------------------


def test_boss2_artefacts():
    """Le boss ne se valide pas sur du code : il se valide sur ses ARTEFACTS.

    Lance : .venv/bin/python scripts/boss2_barrier_sweep.py
    Il doit produire figures/boss2_barrier_sweep.png et figures/boss2_results.json.

    Ce que le JSON doit raconter : quand la barriere remonte vers le spot, le
    DI put ressemble de plus en plus au put vanille, rho monte, et le ratio des
    demi-largeurs s'effondre. C'est la courbe a savoir dessiner au tableau.
    """
    fig = ROOT / "figures" / "boss2_barrier_sweep.png"
    res = ROOT / "figures" / "boss2_results.json"
    assert fig.exists(), "figure manquante"
    assert res.exists(), "resultats manquants"

    data = json.loads(res.read_text())
    sweep = data["sweep"]
    assert len(sweep) >= 6

    h = [p["H_pct"] for p in sweep]
    rho = [p["rho"] for p in sweep]
    ratio = [p["ratio_half_width"] for p in sweep]

    assert h == sorted(h)
    assert abs(h[0] - 0.60) < 1e-9 and abs(h[-1] - 0.95) < 1e-9
    assert all(0.0 < x <= 1.0 for x in ratio)          # jamais de degradation
    assert all(-1.0 <= x <= 1.0 for x in rho)
    # rho croissant en H, avec une tolerance pour le bruit MC
    assert all(rho[i+1] > rho[i] - 0.02 for i in range(len(rho) - 1))
    assert rho[-1] > rho[0]
    assert ratio[-1] < 0.5                             # gain massif pres du spot
