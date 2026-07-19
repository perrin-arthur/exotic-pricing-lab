# exotic-pricing-lab

Pricer d'options vanille en Python : formule fermée Black-Scholes et moteur Monte Carlo (avec variance de réduction par variables antithétiques), avec calcul de delta et validation croisée entre les deux méthodes.

## Structure

```
quant_pricing/
├── src/
│   ├── bs.py         # Black-Scholes fermé (call, put via parité, delta)
│   └── mc_engine.py  # Monte Carlo (GBM, call/put, antithétique, delta par CRN)
├── tests/
│   └── tests_pricer.py
└── notebooks/
    └── 01_validations_bs-mc.ipynb
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./quant_pricing numpy scipy pytest
```

## Tests

```bash
.venv/bin/python -m pytest quant_pricing/tests/tests_pricer.py -v
```

Vérifie : valeur de référence Black-Scholes, parité put-call, convergence du Monte Carlo dans son intervalle de confiance, et cohérence du delta Monte Carlo (variables aléatoires communes) avec le delta analytique.
