# exotic-pricing-lab

Monte Carlo pricing engine for equity exotics, built from vanilla options up to worst-of Barrier Reverse Convertibles under multi-asset Heston.
Includes closed-form validation, variance reduction, and Greeks via common random numbers.
Currently at the vanilla stage (Black-Scholes vs. Monte Carlo, antithetic variates, delta by CRN) — multi-asset and stochastic-vol pricing are next.

## Getting started

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ./quant_pricing numpy scipy pytest
.venv/bin/python -m pytest quant_pricing/tests/tests_pricer.py -v
```
