"""BOSS 6 artefacts: the figure and results file exist and carry the expected content.

Validated by the script itself, as with the other bosses: the BRC is priced on
calibrated marginals (Heston, 6.2) and the historical correlation (6.4), against
a flat-BS benchmark on the same correlation. The price is bounded: a BRC pays at
most the coupon plus the nominal.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def test_boss6_artefacts():
    assert (FIG / "boss6_brc_calibrated.png").exists()
    res = json.loads((FIG / "boss6_results.json").read_text())
    assert set(res["marginals"]) == {"GLD", "SLV", "USO"}
    C = res["params"]["C"]
    for model in ("heston", "bs_flat"):
        price = res["base"][model]["price"]
        assert 0.0 < price < 1.0 + C
    lams = [s["lam"] for s in res["sweep"]]
    assert lams == [0.0, 0.25, 0.5, 0.75, 1.0]
