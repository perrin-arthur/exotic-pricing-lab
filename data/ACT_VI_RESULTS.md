# Act VI — résultats (snapshot 2026-10-25)

Source : fichiers commités dans `data/` et `figures/`, générés par les scripts. Aucun chiffre saisi à la main.

## 1. Marché et taux

- Taux sans risque r = 4.0118% (^IRX, continu)
- GLD spot = 366.6100
- SLV spot = 48.7200
- USO spot = 169.5300

## 2. Heston par nom (6.2) — NON APPLICABLE comme modèle de marché

Calibrage joint sur les 12 échéances (20 à 400 jours). Part des cotes dans la bande bid-ask, et médiane |résidu| en demi-spreads.

| nom | v0 | κ | θ | ξ | ρ | dans la bande | médiane |res| |
|---|---|---|---|---|---|---|---|
| GLD | 0.0558 | 9.435 | 0.0583 | 1.799 | -0.117 | 21.0% | 2.17 |
| SLV | 0.1689 | 5.243 | 0.1827 | 2.718 | +0.016 | 32.5% | 1.54 |
| USO | 0.4064 | 4.992 | 0.2021 | 3.000 | +0.201 | 69.6% | 0.59 |

ξ = 3.000 (USO) est à la borne haute du calibrage. Ces paramètres ne reproduisent pas le marché à la précision bid-ask.

## 3. SVI par échéance (6.3)

Paramètres bruts SVI, w(k) = a + b(ρ(k−m) + √((k−m)²+σ²)), k = ln(K/F). Dans la bande par tranche.

### GLD — 89.0% dans la bande (quote-weighted)

| échéance | T (ans) | a | b | ρ | m | σ | dans la bande |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | 0.0712 | -0.03877 | 0.10852 | +0.1245 | +0.0749 | 0.3932 | 90% |
| 2026-11-27 | 0.0904 | -0.08855 | 0.16310 | +0.1849 | +0.1349 | 0.5791 | 100% |
| 2026-12-04 | 0.1096 | -0.11991 | 0.18552 | +0.1681 | +0.1442 | 0.6842 | 100% |
| 2026-12-18 | 0.1479 | -0.08298 | 0.15294 | +0.0709 | +0.0677 | 0.5907 | 76% |
| 2026-12-31 | 0.1836 | -0.26622 | 0.26448 | -0.0620 | -0.0420 | 1.0410 | 72% |
| 2027-01-15 | 0.2247 | -0.08274 | 0.15921 | +0.0223 | +0.0428 | 0.5841 | 95% |
| 2027-02-19 | 0.3205 | -0.32464 | 0.30152 | -0.1145 | -0.1155 | 1.1336 | 81% |
| 2027-03-19 | 0.3973 | -0.30899 | 0.29552 | -0.0411 | -0.0311 | 1.1103 | 89% |
| 2027-03-31 | 0.4301 | -0.46459 | 0.42769 | -0.4143 | -0.5669 | 1.2469 | 83% |
| 2027-06-17 | 0.6438 | -0.47362 | 0.37600 | +0.0438 | +0.0786 | 1.3460 | 96% |
| 2027-06-30 | 0.6795 | -0.50000 | 0.52326 | -0.5443 | -0.7937 | 1.2175 | 97% |
| 2027-09-17 | 0.8959 | -0.50000 | 0.38350 | -0.0170 | -0.0230 | 1.4255 | 99% |

### SLV — 93.5% dans la bande (quote-weighted)

| échéance | T (ans) | a | b | ρ | m | σ | dans la bande |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | 0.0712 | -0.05425 | 0.14541 | +0.2019 | +0.1285 | 0.4609 | 93% |
| 2026-11-27 | 0.0904 | -0.07979 | 0.19365 | +0.3873 | +0.2553 | 0.5237 | 98% |
| 2026-12-04 | 0.1096 | -0.02929 | 0.13316 | +0.3564 | +0.1772 | 0.3673 | 100% |
| 2026-12-18 | 0.1479 | -0.03970 | 0.14443 | +0.1687 | +0.1004 | 0.4332 | 94% |
| 2026-12-31 | 0.1836 | -0.05157 | 0.16079 | +0.2176 | +0.1241 | 0.4967 | 84% |
| 2027-01-15 | 0.2247 | -0.08215 | 0.19302 | +0.1220 | +0.0835 | 0.5941 | 94% |
| 2027-03-19 | 0.3973 | -0.45160 | 0.39843 | -0.1313 | -0.2120 | 1.2861 | 99% |
| 2027-03-31 | 0.4301 | -0.50000 | 0.40507 | -0.1593 | -0.2913 | 1.4021 | 85% |
| 2027-04-16 | 0.4740 | -0.50000 | 0.38611 | +0.1824 | +0.1983 | 1.4935 | 95% |
| 2027-06-17 | 0.6438 | -0.50000 | 0.40626 | +0.0725 | +0.0077 | 1.4576 | 94% |
| 2027-06-30 | 0.6795 | -0.48871 | 0.60252 | -0.5728 | -1.0000 | 1.1772 | 97% |
| 2027-09-17 | 0.8959 | -0.50000 | 0.42696 | -0.0323 | -0.2016 | 1.4648 | 97% |

### USO — 99.8% dans la bande (quote-weighted)

| échéance | T (ans) | a | b | ρ | m | σ | dans la bande |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | 0.0712 | -0.06209 | 0.19407 | -0.3951 | -0.2835 | 0.4748 | 100% |
| 2026-11-27 | 0.0904 | -0.05694 | 0.18355 | -0.3180 | -0.2458 | 0.4820 | 99% |
| 2026-12-04 | 0.1096 | 0.03021 | 0.04147 | +0.2567 | -0.0584 | 0.0825 | 100% |
| 2026-12-18 | 0.1479 | -0.02854 | 0.15867 | -0.1844 | -0.2058 | 0.4581 | 100% |
| 2027-01-15 | 0.2247 | -0.12176 | 0.23782 | -0.1812 | -0.2592 | 0.7802 | 100% |
| 2027-02-19 | 0.3205 | -0.00910 | 0.18246 | -0.1307 | -0.1858 | 0.4900 | 100% |
| 2027-03-19 | 0.3973 | -0.34515 | 0.37588 | -0.1082 | -0.2200 | 1.1846 | 100% |
| 2027-04-16 | 0.4740 | -0.07106 | 0.24431 | -0.0250 | -0.0835 | 0.7441 | 100% |
| 2027-05-21 | 0.5699 | -0.14110 | 0.30668 | -0.1716 | -0.1868 | 0.8791 | 100% |
| 2027-06-17 | 0.6438 | -0.50000 | 0.45528 | -0.2285 | -0.3568 | 1.4451 | 99% |
| 2027-07-16 | 0.7233 | 0.07862 | 0.16699 | -0.1490 | -0.0288 | 0.4337 | 100% |
| 2027-08-20 | 0.8192 | 0.10005 | 0.15425 | -0.0288 | +0.0636 | 0.4093 | 100% |

**Avertissement d'arbitrage.** Les tranches ne sont pas raccordées en calendrier : la variance totale décroît entre échéances adjacentes à certains k (GLD 530, SLV 857, USO 422 points de grille violés). GLD a aussi un papillon négatif sur l'échéance 2026-11-27 (62 points). Ces paramètres ne forment pas une surface utilisable en l'état. Une tranche isolée reste un ajustement valable pour son échéance.

## 4. Corrélation historique (6.4) — APPLICABLE

Rendements log quotidiens, 2 ans, dates communes, matrice réparée par clipping des valeurs propres.

| | GLD | SLV | USO |
|---|---|---|---|
| GLD | +1.000 | +0.791 | -0.017 |
| SLV | +0.791 | +1.000 | -0.047 |
| USO | -0.017 | -0.047 | +1.000 |

## 5. Holdout sur la plus longue échéance (6.5)

Calibré sur les autres échéances, prédit sur l'échéance la plus longue. Médiane |résidu| en demi-spreads (plus petit = mieux).

| nom | échéance | Heston | SVI interpolé | BS plat (vol ATM) |
|---|---|---|---|---|
| GLD | 2027-09-17 | 1.48 (37%) | 0.62 (75%) | 3.02 (18%, σ=22.62%) |
| SLV | 2027-09-17 | 1.81 (15%) | 0.72 (74%) | 6.16 (15%, σ=38.38%) |
| USO | 2027-08-20 | 0.60 (67%) | 0.34 (99%) | 0.33 (93%, σ=45.89%) |

GLD et SLV : Heston et SVI battent le BS plat. USO : le BS plat gagne en médiane.

## 6. BRC worst-of (6.7, BOSS 6) — DÉPENDANT DU 6.2

Coupon 10%, barrière 70% de performance, échéance 2027-06-17 (T = 0.6438 an), 40000 trajectoires, 52 pas.

| modèle des marginales | prix (nominal 1) | demi-IC 95 % | probabilité de knock-in |
|---|---|---|---|
| Heston calibré + corrélation historique | 0.8741 | ±0.0021 | 54.4% |
| BS plat (vol ATM) + même corrélation | 0.9100 | ±0.0018 | 49.5% |

Sensibilité à la corrélation (Heston, λ = 0 indépendant → 1 historique) :

| λ | Heston | BS plat | KI Heston | KI BS |
|---|---|---|---|---|
| 0.00 | 0.8704 | 0.9047 | 55.1% | 50.9% |
| 0.25 | 0.8705 | 0.9077 | 55.3% | 49.9% |
| 0.50 | 0.8724 | 0.9061 | 54.6% | 50.5% |
| 0.75 | 0.8739 | 0.9095 | 54.5% | 49.6% |
| 1.00 | 0.8743 | 0.9116 | 54.4% | 49.0% |

Le prix Heston dépend d'un calibrage qui ne colle pas au marché (section 2). L'écart de 0.036 avec BS plat est une différence de modèle, pas un prix validé.

## 7. Ce qui n'est pas fait

- **6.6** (test temporel) : un seul snapshot, il en faut 5. Non exécuté.
- **Surface SVI sans arbitrage** : non construite. Il faudrait une paramétrisation qui impose la monotonie en T et la positivité du papillon.
- **Heston sur le marché** : pas de jeu de paramètres qui tombe dans la bande bid-ask.

## 8. Surface SSVI sans arbitrage (correction de l'arbitrage, section 3)

Paramétrisation Gatheral–Jacquier : une surface, paramètres globaux ρ, η, γ partagés par toutes les échéances, un θ (variance totale ATM) par échéance. Vérifications de calendrier et de papillon sur une grille fine (1201 points). Résultat : **zéro violation** pour les trois noms.

| nom | ρ | η | γ | dans la bande | médiane |res| | holdout dans la bande | holdout médiane |
|---|---|---|---|---|---|---|---|
| GLD | -0.095 | 1.101 | 0.468 | 31.2% | 1.70 | 45.2% | 1.20 |
| SLV | +0.015 | 1.398 | 0.413 | 41.6% | 1.32 | 66.7% | 0.88 |
| USO | +0.151 | 0.708 | 0.575 | 83.7% | 0.42 | 79.7% | 0.44 |

L'arbitrage est bien éliminé, mais la forme partagée coûte de l'ajustement : 31 / 42 / 84 % contre 89 / 94 / 99.8 % pour des tranches indépendantes. Le poids de pénalité n'y change rien (testé 0.5, 5, 50) : la contrainte est la structure, pas la pénalité.

## 9. Heston et Bates, multi-départs (section 2)

12 départs quasi-aléatoires (Sobol) par nom et par modèle. Meilleur minimum trouvé, pas de preuve d'optimum global.

| nom | modèle | dans la bande | médiane |res| | SSE meilleur | SSE min / médiane / max sur les départs |
|---|---|---|---|---|---|
| GLD | heston | 21.0% | 2.17 | 17339 | 17339 / 17339 / 17339 |
| GLD | bates | 46.6% | 1.09 | 8828 | 8828 / 14528 / 16175 |
| SLV | heston | 32.5% | 1.54 | 8672 | 8672 / 8672 / 8672 |
| SLV | bates | 48.8% | 1.03 | 5597 | 5597 / 5597 / 6651 |
| USO | heston | 69.6% | 0.59 | 1977 | 1977 / 1977 / 1977 |
| USO | bates | 84.0% | 0.50 | 713 | 713 / 713 / 713 |

Heston : un seul minimum trouvé depuis tous les départs, la borne ξ = 3 est atteinte sur USO. Bates améliore nettement (47 / 49 / 84 %) mais n'atteint pas 80 % sur GLD et SLV. GLD est multimodal (SSE de 8 800 à 16 200 selon le départ).

## 10. Backtest 6.6

Non exécuté sur des données réelles : un seul snapshot, il en faut 5. La logique est écrite (`src/backtest.py`) et vérifiée sur une série de contrôle fabriquée à partir du snapshot. Ces chiffres ne sont pas des résultats et ne sont pas enregistrés.


## 11. Snapshot 2026-10-26 — non calibrable en l'état

Collecté avec `scripts/collect_market.py`. Les spots sont identiques à ceux du 25 (pas de nouvelle clôture le week-end). Les cotes sont dégradées : 142 cotations SLV sur 2064 ont un bid et un ask positifs, et les bids nuls dominent en pré-ouverture.

- Après nettoyage : GLD 257 lignes, SLV 117, USO 149.
- SLV : aucune échéance n'a au moins 3 paires call/put (parité impossible), donc pas de forward, donc pas de calibration SVI.
- GLD : SVI calibré, 58.3 % dans la bande (contre 89.0 % le 25). Résultat indicatif seulement, non retenu.

Le snapshot est conservé tel quel dans `data/`. Il faut un snapshot en séance pour recalibrer.
