# PROGRESS — exotic-pricing-lab

**XP : 340 / 450**  ·  Acte I bouclé à 190/220 (les digitales manquent) · Acte II à 150/230

> **Prochaine quête débloquée : BOSS 2 — balayage de barrière 60 % → 95 % + figure (80 XP)**
> `scripts/boss2_barrier_sweep.py` · valide avec `pytest tests/test_variance_reduction.py::test_boss2_artefacts`
> Il débloque l'ACTE III (Heston).

> Suite au vert : **28 passed, 1 xfailed** (BOSS 2).

Règle du jeu : une quête n'est acquise que si son test de validation est **vert**
contre une référence indépendante (formule fermée, parité, identité model-free).
Les tests de l'Acte II sont écrits en `xfail(strict=True)` : quand ton code est
juste, le test passe en XPASS et fait échouer la suite — c'est le signal pour
retirer le marqueur. Retirer le marqueur avant d'avoir codé ne trompe personne.

---

## ACTE I — Fondations (190 / 220 XP)

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 1.1 Black-Scholes fermé (call/put, q, delta, vega) | ✅ | 20/20 | — | `test_bs_reference_value`, `test_call_bs_dividende`, `test_put_call_parity_bs` |
| 1.2 Pricer MC un-pas + IC 95% | ✅ | 20/20 | 1.1 | `test_mc_within_ci` |
| 1.3 Antithétiques (un-pas) | ✅ | 20/20 | 1.2 | variance ÷ 2 mesurée |
| 1.4 Delta MC en common random numbers | ✅ | 30/30 | 1.2 | `test_delta_crn` vs N(d1) |
| 1.5 **Digitales (réplication call spread)** | ⬜ **OUVERTE** | **0/30** | 1.1 | *aucun code dans `src/`, aucun test — trou à combler* |
| 1.6 `gbm_paths` multi-pas (cumsum vectorisé) | ✅ | 30/30 | 1.2 | `test_gbm_paths_shape_et_depart`, `_call_europeen`, `_loi_independante_de_n_steps` |
| 1.7 Barrières DI / DO put + parité pathwise | ✅ | 40/40 | 1.6 | `test_di_do_van` (DI + DO = vanille, 1e-12) |
| 1.8 Vol implicite (Newton, seed Manaster-Koenig) | ✅ | 30/30 | 1.1 | `test_implied_vol_call` (tolérance vega-dépendante) |

Restent en dette technique, non scorées : monitoring discret (prix vs `n_steps`,
Broadie-Glasserman-Kou) et delta près de la barrière — les deux étaient au
programme du TP barrières et n'ont pas été faits.

## ACTE II — Réduction de variance (150 / 230 XP)

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 2.1 `control_variate(Y, X, EX, c=None)` générique | ✅ | 30/30 | 1.2 | `pytest -k test_cv_` — 7 XPASS |
| 2.2 Cas dégénéré Y = X = put vanille (boss de tuto) | ✅ | 20/20 | 2.1 | `test_cv_degenere_Y_egal_X` — prix BS exact à 1e-12 |
| 2.3 Branchement sur le DI put | ✅ | 40/40 | 2.1, 1.7 | `pytest -k di_put` — 4 XPASS |
| 2.4a `gbm_paths_antithetic` multi-pas | ✅ | 20/20 | 1.6 | `test_gbm_paths_antithetic_partage_Z` |
| 2.4b Antithétiques × contrôle (paires D'ABORD) | ✅ | 20/20 | 2.4a, 2.1 | `pytest -k antithetic` — 2 XPASS |
| 2.5 `pilot_c` — c figé sur run pilote | ✅ | 20/20 | 2.3 | `pytest -k pilot` — 3 XPASS |
| **BOSS 2** Balayage barrière 60 % → 95 % + figure | ⬜ TODO | 0/80 | 2.3, 2.5 | `test_boss2_artefacts` (lit `figures/boss2_results.json`) |

Où se trouve chaque bloc TODO :
`scripts/boss2_barrier_sweep.py` → BOSS 2. `src/mc_engine.py` : plus rien pour l'Acte II.

### Séance 5 — mesures de la 2.5 (`pytest -k pilot`)

DI put H=90, S0=100, K=100, r=0.05, σ=0.20, T=1, n_steps=20 :

    c_pilote (10 000 chemins) = 1.0013     c_plein (100 000) = 1.0003
    ref brut = 5.29734 +/- 0.03840         moyenne CV (c figé) = 5.27146 +/- 0.00391

`c ≈ 1` car à H=90 le DI put est presque le put vanille. Le pilote 10× plus
court donne `c` à **0,1 %** — démonstration empirique que la variance est
**plate** autour de `c*` : un `c` grossier ne coûte qu'une fraction du gain,
jamais du biais. Demi-largeur 10× plus serrée, et cette fois avec un `c`
**déterministe** vis-à-vis de l'échantillon final, donc `E[c(X̄−EX)] = 0`
exactement — le biais en O(1/N) de la 2.1 a disparu.

Réponse d'entretien : « vous estimez c sur le même échantillon que le prix, où
est le problème ? » → `ĉ` et `X̄` sont corrélés, donc `E[ĉ(X̄−EX)] ≠ 0` : biais
en O(1/N), négligeable devant l'erreur MC en O(1/√N) — mais un desk qui produit
un mark quotidien sur le même générateur voit un décalage systématique, pas un
bruit qui se moyenne.

### Séance 5 — mesures de la 2.4 (`test_cv_antithetic_domine_chaque_technique_seule`)

Put vanille S0=K=100, r=0.05, σ=0.20, T=1, n_steps=25, budget **2N = 50 000
trajectoires pour les quatre estimateurs** (comparer à budget différent ne veut
rien dire). Contrôle = forward actualisé, E[e^{-rT} S_T] = S0 exactement.

| Estimateur | demi-IC | gain **variance** |
|---|---:|---:|
| brut | 0.07575 | 1× |
| AV seul | 0.05783 | 1.72× |
| CV seul | 0.04939 | 2.35× |
| AV + CV | 0.02372 | **10.2×** |

Corrélations mesurées : ρ(Y_up, Y_down) = **−0.416** → gain AV = 2/(1+ρ) = 1.71
(le facteur 2 supposerait ρ = 0 ; on fait mieux car le put est monotone en Z).
ρ(put, fwd) = **−0.758** → gain CV = 1/(1−ρ²) = 2.35.

Le point non trivial : 1.72 × 2.35 = 4.0, or on mesure **10.2**. Après pairage,
ρ(put, fwd) passe de −0.758 à **+0.912** — la moyenne par paire est une fonction
**paire** de Z, donc put pairé et forward pairé croissent tous deux en |Z| et
deviennent quasi colinéaires. L'AV n'a pas seulement réduit la variance : elle a
**amélioré le contrôle**. Ne pas généraliser (propre à ce couple), mais c'est la
remarque à placer en entretien.

Asymétrie AV / CV à savoir énoncer : un CV mal choisi ne dégrade jamais (c* → 0),
un AV mal choisi **dégrade** — payoff non monotone en Z (straddle) → ρ > 0 → la
variance monte à budget égal.

### Séance 4 — mesures de la 2.3 (`scripts/run_s4.py`)

S0=100, K=100, T=1, r=0.02, σ=0.30, n_steps=252, N=100 000, seed=20240904 :

|       H | Prix MC | demi-IC | Prix CV | demi-IC | rho_hat | gain IC | parité KI+KO−van |
|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-----------------:|
| 65.0000 |  5.5295 |  0.0833 |  5.4965 |  0.0500 |  0.7998 | 1.6658× |       3.5527e-15 |
| 85.0000 | 10.5477 |  0.0880 | 10.5044 |  0.0100 |  0.9936 | 8.8272× |       3.5527e-15 |

Le gain vaut `1/√(1−ρ²)` : 1.667 attendu / 1.666 mesuré à ρ=0.80, 8.90 / 8.83 à
ρ=0.9936. ρ mesure la fraction de chemins où le DI put **coïncide** avec son
contrôle : à H=85 (≈0.5·σ√T sous le spot en log) presque tous les chemins dans
la monnaie ont touché, le contrôle explique 99 % de la variance ; à H=65
(≈1.4·σ√T) l'indicatrice découple les deux payoffs. Relation **non linéaire** —
un contrôle « correct » ne paie presque rien, seul un contrôle quasi parfait
paie. Corollaire vérifié à H=60 : un mauvais contrôle ne dégrade jamais (c*→0).

## ACTE III — Heston mono-actif

🔒 **LOCKED — se débloque au BOSS 2.** Têtes de chapitre dans `src/heston.py`.

---

### Pièges surveillés par les tests de l'Acte II

| Piège | Test qui l'attrape |
|---|---|
| `np.max` au lieu de `np.maximum` (et `min` sans `axis=1`) | `test_di_put_payoffs_valeurs_a_la_main` |
| Précédence : `cov/sd_Y*sd_X` au lieu de `cov/(sd_Y*sd_X)` | `test_cv_rho_invariant_par_echelle` |
| Parenthésage : `Y - c*X - EX` au lieu de `Y - c*(X - EX)` | `test_cv_utilise_bien_EX` |
| Variable globale qui fuit (le `N` fantôme, 2 fois déjà) | `test_cv_pas_de_N_fantome` |
| Actualisation incohérente entre Y, X et EX | `test_di_put_payoffs_actualisation_coherente` — **tombé dans le panneau S4** |
| Demi-largeur calculée sur Y au lieu du résidu Z | `test_cv_half_width_sur_le_residu` |
| Division par 2N au lieu de N (nombre de paires) | `test_cv_antithetic_N_est_le_nombre_de_paires` |
| Code pas exécuté avant d'être montré | tous — le test doit tourner, pas être lu |

### Pièges tombés en séance 4 (à ne pas refaire)

| Piège | Où | Coût |
|---|---|---|
| `np.cov(ddof=1)` vs `np.var(ddof=0)` — normalisations différentes | 2.1 | `c_hat` faux d'un facteur n/(n−1) : 5e-5 contre une tolérance à 1e-10 |
| `np.correlate` ≠ `np.corrcoef` | 2.1 | corrélation croisée du signal, pas Pearson |
| `np.corrcoef(...)` renvoie une **matrice 2×2**, pas un scalaire | 2.1 | `assert` sur array → `ValueError: truth value ambiguous` |
| Paramètre `c` ignoré (recalculé inconditionnellement) | 2.1 | 4 tours pour le voir ; `c=0` doit redonner le MC brut |
| Payoffs **non actualisés** rendus par `di_put_payoffs` | 2.3 | X̄−EX faux d'un facteur `e^{rT}` **et de signe opposé** : biais silencieux |
| `van_put` est un **pricer** (rend un tuple), pas un vecteur de payoffs | 2.3 | confusion des couches finance / stats |
| Spot **hardcodé** dans `put_bs(100, ...)` | 2.3 | tests verts (tous à S0=100) mais prix faux de 11 pts à S0=80, avec IC nul |
| `paths[:, 0]` (array) au lieu de `paths[0, 0]` (scalaire) | 2.3 | 100 000 calculs BS identiques, 800 Ko, 50× plus lent |

### Pièges tombés en séance 5 (à ne pas refaire)

| Piège | Où | Coût |
|---|---|---|
| Appeler `gbm_antithetic` (un-pas) pour faire du multi-pas | 2.4a | `n_steps` n'apparaissait pas dans le corps — le tell : un paramètre non consommé |
| 6 arguments positionnels pour une signature à 7 | 2.4a | `N` → `n_steps`, `rng` → `N` : `TypeError` opaque. Nommer les arguments |
| Réutiliser `gbm_paths` (qui **cache** son `Z`) pour construire des paires | 2.4a | impossible par construction : une fonction qui cache son aléa n'est pas composable — même raison que l'injection de `rng` dans `delta_mc` |
| Passer deux fois le même objet `rng` en croyant rejouer les mêmes tirages | 2.4a | un `Generator` a un **état** : il avance. Deux `default_rng(3)` auraient donné `up == down` |
| `S0np.exp(...)` (`*` manquant) | 2.4a | pas une `SyntaxError` — accès attribut valide → `NameError` à l'exécution, avalé par le `xfail` |
| Copier-coller : `X_pair = (Y_up + Y_down)/2` | 2.4b | retombe sur le cas dégénéré 2.2 → demi-IC **exactement 0** et un test au vert. Attrapé par le seul test qui pose une **égalité**, pas une inégalité |
| `np.cov(Y, X)` avec `Y`/`X` absents de la signature (params : `Y_pilot`/`X_pilot`) | 2.5 | **3e occurrence** de la globale fantôme. Sauvé par le `NameError` faute de global homonyme — avec un `X` au niveau module, c'était un `c` faux et silencieux |
| `[0, 1]` et `float(...)` oubliés sur `np.cov` | 2.5 | déjà tombé en 2.1, réécrit correctement ligne 133 puis refait 150 lignes plus bas |

Leçon transverse de la séance : `xfail(strict)` **avale n'importe quelle
exception** (typo, `NameError`, `TypeError`) et l'affiche comme un XFAIL
attendu. Tant qu'un test est marqué, il ne diagnostique rien — appeler la
fonction à la main dans un REPL est le seul moyen de voir la vraie erreur.

Le fil : les 4 derniers pièges de la S4 sont **invisibles sans référence externe**. Un
contrôle décentré rend un prix plausible et une demi-largeur qui rétrécit — plus
il est faux, plus il a l'air précis. D'où la règle du repo : valider contre une
formule fermée ou une identité de parité, jamais contre « ça a l'air correct ».
