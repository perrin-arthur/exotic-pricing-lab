# PROGRESS — exotic-pricing-lab

**XP : 1000 / 1000**  ·  Acte I **bouclé à 220/220** · Acte II **bouclé à
230/230** · Acte III **bouclé à 210/210** · Acte IV **bouclé à 340/340**

> **TOUT L'ARBRE ACTUEL EST BOUCLÉ.** 1.5 (digitales) a comblé le dernier trou
> de l'Acte I ; l'Acte IV (hedging discret) est clos avec son BOSS 4. Prochaine
> étape : l'acte suivant, mis de côté quand l'Acte IV est passé devant lui à la
> demande explicite du prompt qui l'a ouvert — multi-actif par Cholesky →
> worst-of → autocall → BRC worst-of sous Heston, le produit phare annoncé en
> tête de `CLAUDE.md`.

> Suite au vert : **62 passed, 0 xfailed** (48 + 12 tests de l'Acte IV + 2 de
> la quête 1.5).

Règle du jeu : une quête n'est acquise que si son test de validation est **vert**
contre une référence indépendante (formule fermée, parité, identité model-free).
Les tests de l'Acte II sont écrits en `xfail(strict=True)` : quand ton code est
juste, le test passe en XPASS et fait échouer la suite — c'est le signal pour
retirer le marqueur. Retirer le marqueur avant d'avoir codé ne trompe personne.

---

## ACTE I — Fondations (220 / 220 XP) ✅

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 1.1 Black-Scholes fermé (call/put, q, delta, vega) | ✅ | 20/20 | — | `test_bs_reference_value`, `test_call_bs_dividende`, `test_put_call_parity_bs` |
| 1.2 Pricer MC un-pas + IC 95% | ✅ | 20/20 | 1.1 | `test_mc_within_ci` |
| 1.3 Antithétiques (un-pas) | ✅ | 20/20 | 1.2 | variance ÷ 2 mesurée |
| 1.4 Delta MC en common random numbers | ✅ | 30/30 | 1.2 | `test_delta_crn` vs N(d1) |
| 1.5 Digitales (réplication call spread) | ✅ | 30/30 | 1.1 | `test_digital_call_bs_vs_mc` (IC) + `test_digital_call_replication_converge_en_h2` (ratio ~4.000) |
| 1.6 `gbm_paths` multi-pas (cumsum vectorisé) | ✅ | 30/30 | 1.2 | `test_gbm_paths_shape_et_depart`, `_call_europeen`, `_loi_independante_de_n_steps` |
| 1.7 Barrières DI / DO put + parité pathwise | ✅ | 40/40 | 1.6 | `test_di_do_van` (DI + DO = vanille, 1e-12) |
| 1.8 Vol implicite (Newton, seed Manaster-Koenig) | ✅ | 30/30 | 1.1 | `test_implied_vol_call` (tolérance vega-dépendante) |

### Séance 8 — la 1.5

Call spread `(C(K-h) - C(K+h))/(2h)` contre `digital_call_bs` : ratio de
convergence mesuré **3.999 / 3.9998 / 4.000** en divisant `h` par 2 à chaque
fois (0.4 → 0.2 → 0.1 → 0.05) — O(h²) exact, à trois décimales de 4. Contre la
formule fermée elle-même, `digital_call_bs` vs MC (N=200 000) :
`0.53232` contre `0.53256 ± 0.00207`, dans l'IC.

Restent en dette technique, non scorées : monitoring discret (prix vs `n_steps`,
Broadie-Glasserman-Kou) et delta près de la barrière — les deux étaient au
programme du TP barrières et n'ont pas été faits.

## ACTE II — Réduction de variance (230 / 230 XP) ✅

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 2.1 `control_variate(Y, X, EX, c=None)` générique | ✅ | 30/30 | 1.2 | `pytest -k test_cv_` — 7 XPASS |
| 2.2 Cas dégénéré Y = X = put vanille (boss de tuto) | ✅ | 20/20 | 2.1 | `test_cv_degenere_Y_egal_X` — prix BS exact à 1e-12 |
| 2.3 Branchement sur le DI put | ✅ | 40/40 | 2.1, 1.7 | `pytest -k di_put` — 4 XPASS |
| 2.4a `gbm_paths_antithetic` multi-pas | ✅ | 20/20 | 1.6 | `test_gbm_paths_antithetic_partage_Z` |
| 2.4b Antithétiques × contrôle (paires D'ABORD) | ✅ | 20/20 | 2.4a, 2.1 | `pytest -k antithetic` — 2 XPASS |
| 2.5 `pilot_c` — c figé sur run pilote | ✅ | 20/20 | 2.3 | `pytest -k pilot` — 3 XPASS |
| **BOSS 2** Balayage barrière 60 % → 95 % + figure | ✅ | 80/80 | 2.3, 2.5 | `test_boss2_artefacts` — XPASS |

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

### BOSS 2 — le balayage (`scripts/boss2_barrier_sweep.py`)

S0=100, K=100, σ=0.20, r=0.05, T=1, n_steps=50, N=100 000, seed=42, **un seul
jeu de trajectoires pour les 8 barrières** (CRN : sinon ρ(H) tremble du bruit MC
et on ne sait plus si un creux est un effet ou un artefact).

| H/S0 | ρ | ratio demi-IC | c_hat | prix MC | prix CV |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.2989 | 0.9543 | 0.1035 | 0.2296 | 0.2346 |
| 0.70 | 0.6211 | 0.7838 | 0.4502 | 1.3391 | 1.3605 |
| 0.80 | 0.8847 | 0.4661 | 0.8669 | 3.5978 | 3.6390 |
| 0.90 | 0.9918 | 0.1279 | 1.0003 | 5.2880 | 5.3355 |
| 0.95 | 0.9995 | 0.0303 | 1.0009 | 5.5017 | 5.5493 |

Gain de 33× sur la demi-largeur au point haut, soit **1000× en variance**.
`c_hat` monte de 0.10 à 1.00 : à H=95 % le DI put **est** le put vanille. À
H=60 % le déclenchement est rare, `c` s'effondre vers 0 et l'estimateur se
replie tout seul sur le MC brut — **c'est pourquoi le ratio ne dépasse jamais
1**. Réponse à « et si votre contrôle est mal choisi ? ».

`prix_cv − prix_mc ≈ +0.047` partout, **proportionnel à `c_hat`** : c'est
`c·(X̄−EX)`, la correction mesurée sur ce jeu de chemins. Un écart *non*
proportionnel à `c` signalerait un bug.

**Limite de la figure, à savoir énoncer (trouvée par Arthur, pas par le test).**
La courbe « théorique » √(1−ρ̂²) n'est **pas** une validation indépendante :
avec `c = Cov/Var` estimé sur l'échantillon, `Var(Z) = Var(Y)(1−ρ̂²)`
identiquement, donc ratio = √(1−ρ̂²) **par algèbre**. Écart mesuré aux 8
points : **1e-15**, la précision machine. Ce que ça teste réellement :
la cohérence interne (mêmes `ddof` entre `cov`, `var` et `std`, demi-largeur
calculée sur le résidu et pas sur Y, même `n` des deux côtés). Une vraie
référence externe demanderait la formule fermée de Reiner-Rubinstein — qui
suppose un monitoring **continu**, alors qu'on monitore en 50 pas (dette
technique Broadie-Glasserman-Kou). Présenter une tautologie comme une
validation est le genre de chose qui coûte cher si l'examinateur pousse.

## ACTE III — Heston mono-actif (210 / 210 XP) ✅

🔓 **OUVERT.** Blocs TODO dans `src/heston.py`, tests dans `tests/test_heston.py`
(16 xfailed à l'ouverture), boss dans `scripts/boss3_heston_barrier.py`.

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 3.1 `heston_paths` — Euler, 2 browniens corrélés | ✅ | 40/40 | 1.6 | `pytest -k paths` — 5 XPASS |
| 3.2 `heston_cf` / `heston_call` / `heston_put` — semi-analytique | ✅ | 50/50 | 3.1 | 6 XPASS : limite BS, parité, accord MC |
| 3.3 `heston_smile` — vol implicite par strike | ✅ | 40/40 | 3.2, 1.8 | `pytest -k smile` — 4 XPASS |
| **BOSS 3** DI put sous Heston + CV + figure smile | ✅ | 80/80 | 3.3, Acte II | `test_boss3_artefacts` — XPASS |

Le fil de l'acte : **simuler** (3.1) → se donner une **référence analytique**
(3.2) → **lire** ce que le modèle produit (3.3) → **rebrancher l'Acte II
dessus** (BOSS 3). Chaque validation est indépendante du modèle : martingale
`E[e^{-rT}S_T]=S0`, moyenne exacte du CIR, limite dégénérée ξ→0 vers
Black-Scholes, parité call-put. Aucun test ne compare Heston à Heston.

### Séance 6 — la 3.1

`E[v_T]` attendu 0.051157 / mesuré 0.050911 (0,5 %), `ρ` imposé −0.7 / mesuré
−0.6986. Martingale et limite BS dans l'IC.

**Décision de schéma à assumer.** Le tableau `v` rendu est une *sortie* : une
variance négative y serait un `nan` en attente au premier `sqrt` en aval. D'où
la séparation état interne / sortie rapportée (vecteur de travail `v_brut`,
écriture tronquée dans le tableau). Attention au détail qui change le schéma :
si la récursion relit la valeur **tronquée**, ce n'est plus la full truncation
mais le schéma **absorbé** — la full truncation garde la mémoire de l'excursion
négative, l'absorbé la remet à zéro. Écart mesuré ici : 1e-5 relatif, parce que
Feller tient (2κθ = 0.12 > 0.09 = ξ²). **Sur des paramètres calibrés au marché,
Feller est presque toujours violée** et l'écart devient visible. Lord, Koekkoek
& van Dijk (2010) : la full truncation est la variante d'Euler la moins biaisée.

### Séance 6 — la 3.2

Route retenue : Heston 1993 / Gatheral, `P1`/`P2` par `scipy.integrate.quad`,
forme d'Albrecher pour la fonction caractéristique (pas de saut de branche).

| K | Heston | BS(20 %) | lecture |
|---:|---:|---:|---|
| 80 | 25.095 | 24.589 | strikes bas **plus chers** |
| 100 | 10.362 | 10.451 | ATM à peu près aligné |
| 120 | 2.193 | 3.247 | calls OTM **un tiers moins chers** |

Le skew de ρ = −0.7, visible avant même d'avoir écrit 3.3.

**Accord semi-analytique / MC** (N=200 000, n_steps=250) :
`K=90 : MC 17.0984±0.0644 vs exact 17.1069` · `K=100 : 10.3505±0.0523 vs 10.3619`
· `K=110 : 5.3053±0.0380 vs 5.3180`. Dans l'IC, mais **l'exact est au-dessus aux
trois strikes, de ~0.012 à chaque fois**. Trois fois le même signe = biais de
discrétisation d'Euler, pas du bruit. Vérifiable : doubler `n_steps` doit
diviser l'écart par ~2 (Euler est en O(dt)).

**Correction d'un test faux (le mien).** `test_heston_cf_limite_gaussienne`
exigeait 1e-6 à ξ=1e-3 : impossible, l'écart réel y vaut 3.5e-4. L'écart à la
gaussienne est en **O(ξ)**, pas O(ξ²) — le terme de skew `ρ·ξ·u³` est d'ordre
UN, c'est la corrélation qui brise la symétrie (la courbure, elle, est en ξ²).
Mesuré : 3.48e-3 → 3.48e-4 → 3.48e-5 pour ξ = 1e-2, 1e-3, 1e-4, et scaling en
u³ à ξ fixé. Sous ξ ≈ 1e-5 l'erreur **remonte** (annulation catastrophique via
`κθ/ξ²`). Le test vérifie désormais un **ratio de convergence** (facteur 10
mesuré : 10.00) plutôt qu'un seuil — une formule fausse rate la pente, pas
seulement le niveau.

### Séance 6 — la 3.3 : ρ fait la pente, ξ fait la courbure

Smile à ρ=−0.7, K de 80 à 120 (S0=100, F=105.13, T=1) :
`0.2326 0.2235 0.2147 0.2061 0.1976 0.1895 0.1817 0.1744 0.1678`
Pente ajustée en log-moneyness : **−0.1613** — un skew d'indice actions
réaliste à 1 an. Vol ATM 19.76 % contre √θ = 20 % (le forward est au-dessus
du spot).

Courbure (différence seconde en log-moneyness, ρ=0 pour isoler ξ) :

| ξ | courbure | ratio vs ξ=0.1 | ξ² attendu |
|---:|---:|---:|---:|
| 0.1 | 0.00088 | 1 | 1 |
| 0.2 | 0.00353 | 4.01 | 4 |
| 0.3 | 0.00781 | 8.9 | 9 |
| 0.5 | 0.01952 | 22.2 | 25 (saturation) |

**Courbure ∝ ξ² à trois chiffres significatifs.** Deux mesures indépendantes
concordent : la fonction caractéristique donnait un terme de skew `ρ·ξ·u³`
d'ordre UN (séance 6, 3.2), la surface de vol donne une courbure d'ordre DEUX.

Traduction desk : ρ pilote le **risk reversal** (linéairement), ξ pilote le
**butterfly** (quadratiquement). Et c'est le risk reversal qui décide du prix
d'un BRC, parce que le down-and-in put vendu par l'investisseur vit dans l'aile
gauche — là où le skew rend la vol chère. D'où « pourquoi Heston et pas BS ».

### BOSS 3 — le DI put sous Heston (`scripts/boss3_heston_barrier.py`)

S0=K=100, v0=0.04, κ=1.5, θ=0.04, ξ=0.3, ρ=−0.7, r=0.05, T=1, n_steps=50,
N=100 000, seed=42, **une seule simulation pour les 20 barrières** (CRN).
Contrôle = put vanille sur les mêmes trajectoires, `EX = heston_put` (3.2).

| H/S0 | ρ(payoff, contrôle) | ratio demi-IC | c_hat | prix MC | prix CV |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.6390 | 0.7692 | 0.4557 | 1.3064 | 1.2966 |
| 0.70 | 0.8267 | 0.5626 | 0.7501 | 2.7142 | 2.6980 |
| 0.80 | 0.9497 | 0.3132 | 0.9483 | 4.3431 | 4.3227 |
| 0.90 | 0.9960 | 0.0894 | 1.0003 | 5.3445 | 5.3229 |
| 0.95 | 0.9997 | 0.0229 | 1.0005 | 5.4864 | 5.4648 |

Gain de **44× sur la demi-largeur** au point haut (~1900× en variance).

Trois lectures :

1. **`c_hat` démarre à 0.46**, contre 0.10 en Black-Scholes au BOSS 2. Sous
   Heston le DI put ressemble davantage au put vanille dès H=60 % : le skew
   épaissit l'aile gauche, donc les trajectoires qui finissent dans la monnaie
   ont plus souvent touché une barrière basse.
2. **`prix_cv − prix_mc = −0.0216·c_hat`**, exactement proportionnel à `c_hat`
   → le contrôle est centré, `EX` est cohérent avec les trajectoires. Le piège
   « EX en BS sur des trajectoires Heston » a été évité.
3. **`heston_put`(ATM) = 5.4848 contre `put_bs`(20 %) = 5.5735**, soit −0.089 :
   ce que le smile coûte sur un vanille ATM. Petit. Sur un DI put à barrière
   basse l'écart de modèle est bien plus gros — c'est l'argument « pourquoi
   Heston et pas BS » pour un BRC.

**Le dispositif de l'Acte II se rebranche sans une ligne de modification.** Seule
la source de `EX` change : `put_bs` → `heston_put`. La réduction de variance est
une technique statistique, indépendante du modèle.

Les trois pièges qui vont coûter le plus cher, annoncés :
1. **`max(v,0)` oublié** → `nan` silencieux propagé sur toute la trajectoire.
2. **La coupure de branche** de la fonction caractéristique (« the little
   Heston trap ») : prix juste à T=1, délirant à T=5.
3. **`v_{t+dt}` utilisé dans le pas du spot** au lieu de `v_t` — le test de
   corrélation du premier pas est écrit pour ça.

Ensuite : ACTE IV — hedging discret (voir plus bas), qui traite justement le
dernier point de dette technique (delta près d'une barrière, quête 4.5).
Restent ouverts : digitales (1.5, 30 XP), monitoring discret, et — après
l'Acte IV — multi-actif par Cholesky → worst-of → autocall → BRC.

---

## ACTE IV — Hedging discret (340 / 340 XP) ✅

Blocs TODO dans `src/hedging.py`, tests dans `tests/test_hedging.py` (12
verts, 4.1 à 4.6 et `test_boss4_artefacts`), boss dans
`scripts/boss4_hedging.py` (`figures/boss4_hedging.png`,
`figures/boss4_results.json`).

| Quête | Statut | XP | Dépend de | Validation |
|---|---|---|---|---|
| 4.1 Portefeuille auto-financé | ✅ | 30/30 | — | `test_autofinancement_deltas_arbitraires` (1e-12, deltas arbitraires) + 2 cas dégénérés |
| 4.2 Erreur de hedging, bon modèle | ✅ | 50/50 | 4.1 | `test_hedging_error_moyenne_dans_ic_et_pente_log_log` — pente -0.485 mesurée |
| 4.3 Vol arbitrage / modèle faux | ✅ | 50/50 | 4.2 | `test_vol_arbitrage_identite_gamma` — identité gamma-pondérée, signe MOINS |
| 4.4 Coûts de transaction | ✅ | 40/40 | 4.2 | `test_transaction_costs_frequence_optimale` — pente coût +0.44, optimum RMS intermédiaire |
| 4.5 Delta près d'une barrière | ✅ | 40/40 | 4.2 | `test_naive_barrier_hedge_degrade_pres_de_la_barriere` — ratio std 2.82, pente ≈0 (pas -0.5) |
| 4.6 Delta-vega sous Heston | ✅ | 50/50 | 4.2, Acte III | `test_heston_delta_vega_reduit_la_variance` — ratio de variance 3.05 |
| **BOSS 4** Convergence + P&L (figure) | ✅ | 80/80 | 4.2, 4.3, 4.4 | `test_boss4_artefacts` — pente -0.491, 3 scénarios de P&L distincts |

Arbre de dépendance de l'acte :

```
4.1 (auto-financement)
  └─ 4.2 (erreur de hedging, la brique commune)
       ├─ 4.3 (vol arbitrage)
       ├─ 4.4 (coûts de transaction)
       ├─ 4.5 (barrière)
       └─ 4.6 (Heston, dépend aussi de l'Acte III)
            └─ BOSS 4
```

Le fil de l'acte, trois choses défendables en entretien : la **convergence**
de l'erreur de hedging en `n_rebal^{-1/2}` (4.2), l'**identité de vol
arbitrage** (4.3), et la **dégradation du delta près d'une barrière** (4.5) —
le vrai problème métier, celui qui n'a pas de convergence à montrer, juste une
cassure à documenter.

### Séance 7 — les bugs qui ont coûté le plus cher

Contrairement aux actes précédents, la plupart des bugs de cet acte n'étaient
**pas** dans le code de l'étudiant — trois sont tombés dans les tests écrits
par l'assistant, un bon rappel que « le test a raison » n'est pas un axiome.

| Piège | Où | Nature | Coût |
|---|---|---|---|
| `B[-1]` au lieu de `B` dans le calcul de `V_T` | 4.1 | indexation scalaire d'un vecteur par path | invisible sur 2 des 3 tests (deltas dégénérés, où `B` est identique sur tous les paths par construction) — repéré seulement par le test à deltas ARBITRAIRES |
| `deltas` reçu en paramètre mais recalculé en interne avec `sigma=0.2` en dur | 4.2 | paramètre ignoré + magic number | invisible tant que `sigma` du test coïncidait avec 0.20 |
| **Référence de test manquant `exp(-r*t_i)` dans la somme de gamma** | 4.3 | formule de référence incomplète (bug du test, pas du code) | écart stable ~2.4%, NE DIMINUAIT PAS avec `n_steps` — a fait suspecter le code de l'étudiant avant qu'une dérivation par Itô révèle le terme manquant (l'erreur de couverture capitalise elle-même au taux r) |
| Ligne `turnover += np.sum(...)` en trop, reliquat d'une version précédente | 4.4 | code mort qui double-compte un total et le réinjecte dans chaque colonne | pente log-log du coût mesurée à 1.48 au lieu de ~0.5 — repéré par comparaison avec la pente du turnover en actions seul |
| Overlay vega ajouté avec le mauvais signe (`+n_vega*delta` au lieu de `-n_vega*delta`) | 4.6 | confusion position longue / courte dans la convention de `bs_delta_hedge_deltas` | ratio de variance mesuré sous 1 (l'overlay AGGRAVAIT la variance) au lieu d'être nettement au-dessus |

Leçon transverse : un test qui donne un résultat plausible (bon signe, bon
ordre de grandeur) peut quand même être faux d'un terme manquant — la
vérification qui a débloqué 4.3 n'était pas « le signe est bon » mais
« l'écart ne diminue pas avec `n_steps`, donc ce n'est pas un biais de
discrétisation, c'est une formule incomplète ».

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
