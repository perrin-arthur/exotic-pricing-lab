"""Quest 6.4: correlation between the three names, reconstructed from history.

Two-step multi-asset (Arthur's scope, see PROGRESS): each name is calibrated on
its own smile (6.2, 6.3), then the joint dependence comes from realised
returns, not from a joint options calibration. No liquid multi-asset option
market exists to calibrate against.

The estimate is the sample correlation of daily log returns over the common
dates of all names. Sample correlation is not guaranteed positive semi-definite
once pairs are estimated separately or on noisy windows, so it goes through the
same eigenvalue clipping as the rest of Act V before any Cholesky.
"""

import numpy as np
import pandas as pd

from multi_asset import repair_correlation_clip


def historical_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns over the dates where every name has a price."""
    aligned = prices.dropna(how="any")
    return np.log(aligned).diff().dropna(how="any")


def historical_correlation(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw sample correlation and its eigenvalue-repaired version.

    Returns
    -------
    (raw, repaired) : two (n, n) matrices. raw is exactly np.corrcoef of the
        log returns; repaired is repair_correlation_clip(raw), unit diagonal,
        non-negative spectrum, safe for np.linalg.cholesky.
    """
    returns = historical_log_returns(prices)
    raw = np.corrcoef(returns.to_numpy().T)
    return raw, repair_correlation_clip(raw)
