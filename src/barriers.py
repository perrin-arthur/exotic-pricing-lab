"""Down-and-in / down-and-out put payoffs, evaluated on simulated paths.

These pricers take a path array rather than a model, so they work unchanged on
Black-Scholes paths (mc_engine.gbm_paths) and on Heston paths
(heston.heston_paths).

Monitoring is DISCRETE: the barrier is tested at the n_steps+1 simulated dates
only. A continuously monitored barrier is knocked in more often, so these prices
carry a discretisation bias that does not vanish as N grows; the Broadie-
Glasserman-Kou continuity correction is not implemented.

Barrier convention: the knock-in condition is strict, min_t S_t < H. Since the
knock-out condition is min_t S_t >= H, the two are exact complements and the
identity DI + DO = vanilla holds pathwise, to machine precision.
"""

import numpy as np

import bs
import mc_engine

def di_put(paths,K,H,r,T):
    """Monte-Carlo price of a down-and-in put.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    H : barrier level.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) < H}
    ST = paths[:, -1]
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def do_put(paths,K,H,r,T):
    """Monte-Carlo price of a down-and-out put.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    H : barrier level.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    # payoff = max(K-ST,0) * 1_{min(paths) >=  H}
    ST = paths[:, -1]
    min_paths = paths.min(axis=1)
    payoff = np.where(min_paths >= H, np.maximum(K - ST, 0.0), 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def van_put(paths, K, r, T):
    """Monte-Carlo price of a vanilla put, on the same path array.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K : put strike.
    r : risk-free rate.
    T : maturity.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    N = paths.shape[0]
    ST = paths[:, -1]
    payoff = np.maximum(K - ST, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def di_put_payoffs(paths: np.ndarray,
                   K: float,
                   H: float,
                   r: float,
                   T: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-path discounted payoffs of the DI put and of its vanilla control.

    Nothing is averaged here: this function does not price, it builds the two
    paired samples that `mc_engine.control_variate` expects.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0.
    K, H, r, T : strike, barrier, rate, maturity.

    Returns
    -------
    (Y, X) : two (N,) arrays, both DISCOUNTED.
        Y = down-and-in put payoff of the path.
        X = vanilla put payoff of the SAME path, same K, same T.

    Notes
    -----
    Y and X must live in the same units as the control's expectation EX, which
    is a discounted price. Returning them undiscounted shifts the estimate
    instead of stabilising it.

    Y.mean() equals di_put(paths, K, H, r, T)[0] to machine precision, and
    X.mean() equals van_put(paths, K, r, T)[0].
    """

    ST = paths[:, -1]
    min_paths = paths.min(axis=1)

    X = np.maximum(K - ST, 0.0)*np.exp(-r*T)

    Y = np.where(min_paths < H, np.maximum(K - ST, 0.0), 0.0)*np.exp(-r*T)

    return (Y,X)


def di_put_cv(paths: np.ndarray,
              K: float,
              H: float,
              r: float,
              T: float,
              sigma: float,
              c: float | None = None) -> tuple[float, float, float, float]:
    """DI put priced with the vanilla put as control variate, under Black-Scholes.

    The control's expectation is taken from the closed-form BS put, so this
    helper is tied to the Black-Scholes model. Pricing a DI put on Heston paths
    means calling `di_put_payoffs` and `mc_engine.control_variate` directly with
    a Heston EX, as scripts/boss3_heston_barrier.py does.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, S0 in column 0. The spot is read
        back from paths[0, 0].
    K, H, r, T : strike, barrier, rate, maturity.
    sigma : volatility used to SIMULATE paths, needed for the control's exact
        expectation. A mismatch between this sigma and the one behind the paths
        biases the price: that is a model error, not variance reduction.
    c : imposed coefficient, typically from `mc_engine.pilot_c`, or None to
        estimate it in-sample.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat), same convention as
    `mc_engine.control_variate`. Run on the same paths as di_put, the gap
    between the two estimates is exactly c*(X_bar - EX), of the order of the
    control's own CI; a larger gap means the control is not centred.
    """
    Y,X= di_put_payoffs(paths,K,H,r,T)
    EX = bs.put_bs(paths[0, 0],K,sigma,r,T,q=0.0)
    (estimate, half_width, c_hat, rho_hat) = mc_engine.control_variate(Y,X,EX,c)
    return (estimate, half_width, c_hat, rho_hat)
