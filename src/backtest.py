"""Quest 6.6: temporal backtest, parameter stability and J+1 prediction.

Arthur's hypothesis, to be settled with numbers: Heston, fitted on a rolling
window of days, gives day-to-day parameters that are more STABLE than SVI's,
and its day-J parameters predict the day-J+1 smile better than SVI's, even
though SVI wins on intra-day fit.

The backtest needs one calibrated fit per snapshot day. Yahoo gives only the
current chain, so the series is built by running scripts/collect_market.py
every day. Until at least MIN_DAYS snapshots exist, the backtest does not run.
"""

import glob
import pathlib

import numpy as np

MIN_DAYS = 5
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def snapshot_days() -> list[str]:
    """Dates of the chain snapshots available, sorted."""
    return sorted(pathlib.Path(f).stem.removeprefix("chains_")
                  for f in glob.glob(str(DATA / "chains_*.csv")))


def coefficient_of_variation(values: np.ndarray) -> float:
    """sd / |mean| of a parameter over days. The stability measure per parameter."""
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / abs(values.mean()))


def j1_rmse(model_calls_next_day: np.ndarray, quotes_next_day) -> float:
    """RMSE of day-J parameters priced against day-J+1 quotes, in half-spread units."""
    res = (model_calls_next_day - quotes_next_day["C_mid"].to_numpy()) / quotes_next_day["half"].to_numpy()
    return float(np.sqrt(np.mean(res**2)))
