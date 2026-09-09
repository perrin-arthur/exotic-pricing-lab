"""Monte-Carlo engine for geometric Brownian motion, plus variance reduction.

Two families of simulators live here: single-step samplers of S_T (exact, no
discretisation error) and multi-step path simulators, needed as soon as the
payoff is path-dependent.

Library-wide convention: the second return value of every estimator is a 95%
confidence *half-width*, 1.96 * sd / sqrt(n), never a raw standard error.

Random numbers are always injected through an `rng` argument rather than seeded
inside a function, so that callers can pair samples across calls. `delta_mc` is
the documented exception.
"""

import numpy as np


def gbm(S0,sigma,r,T, N=100_000, rng=None):
    """Sample S_T under GBM in a single exact step.

    Params
    ------
    N : number of draws.
    rng : numpy Generator; a fresh one is created when None.

    Returns
    -------
    (N,) array of terminal spot values.
    """
    if rng is None:
        rng = np.random.default_rng()
    return S0*np.exp((r-0.5*sigma**2)*(T)+sigma*np.sqrt(T)*rng.standard_normal(N))


def pricer_mc_call(S0, K, sigma, r, T, N=100_000, rng=None):
    """Monte-Carlo price of a European call.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    ST = gbm(S0, sigma, r, T, N=N, rng=rng)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def pricer_mc_put(S0,K,sigma,r,T,N=100_000, rng=None):
    """Monte-Carlo price of a European put.

    Returns
    -------
    (price, half_width) : price and 95% CI half-width.
    """
    ST = gbm(S0,sigma,r,T, N=N, rng=rng)
    payoff= np.maximum(K-ST,0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96 * disc.std(ddof=1) / np.sqrt(N))


def gbm_antithetic(S0, sigma, r, T, N=50_000, rng=None):
    """Antithetic pairs of terminal spot values, single step.

    Params
    ------
    N : number of PAIRS, hence 2N simulated values.

    Returns
    -------
    (up, down) : two (N,) arrays built on the same gaussian draw, up on +Z and
        down on -Z.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(N)
    drift = (r - 0.5*sigma**2)*T
    up   = S0*np.exp(drift + sigma*np.sqrt(T)*Z)
    down = S0*np.exp(drift - sigma*np.sqrt(T)*Z)
    return up, down


def pricer_mc_call_av(S0, K, sigma, r, T, N=50_000, rng=None):
    """European call priced with antithetic variates.

    Params
    ------
    N : number of pairs, hence 2N simulated paths.

    Returns
    -------
    (price, half_width) : the half-width is computed over the N pair averages,
        which are i.i.d., not over the 2N correlated draws.
    """
    up,down = gbm_antithetic(S0, sigma, r, T, N=N, rng=rng)
    pair = 0.5*(np.maximum(up-K, 0.0) + np.maximum(down-K, 0.0))  # average within each pair
    disc = np.exp(-r*T)*pair
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def delta_mc(S0,h,K,sigma,r,T,N=100_000,seed = 42):
    """Call delta by finite difference on two Monte-Carlo prices, under CRN.

    Params
    ------
    h : spot bump. Note the argument position: h comes second, before K.
    seed : seed replayed identically for both pricings. This is the one place in
        the library where a fixed seed is part of the estimator rather than a
        reproducibility choice, hence the `seed` argument instead of `rng`.

    Returns
    -------
    float : the delta estimate alone. Unlike every other estimator here, no
        confidence half-width is returned.

    Notes
    -----
    Both pricings must see the same gaussian draws, which is why two distinct
    generators are built from the same seed. Without common random numbers the
    up-down difference is dominated by Monte-Carlo noise, whose variance scales
    as 1/h^2, and the estimate is unusable.
    """
    up   = pricer_mc_call(S0+h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    down = pricer_mc_call(S0-h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    return (up-down)/(2*h)


def gbm_paths(S0, sigma, r, T, n_steps, N=100_000, rng=None):
    """Multi-step GBM paths, vectorised through a cumulative sum in log space.

    Params
    ------
    n_steps : number of time steps; dt = T/n_steps.
    N : number of paths.

    Returns
    -------
    (N, n_steps+1) array. Column 0 is exactly S0 for every path, by way of the
    leading zero column in the cumulative sum.
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T/n_steps
    Z = rng.standard_normal((N, n_steps))
    log_incr = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_paths = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(log_incr, axis=1)],
        axis=1,
    )
    return S0*np.exp(log_paths)


def control_variate(Y: np.ndarray,
                    X: np.ndarray,
                    EX: float,
                    c: float | None = None) -> tuple[float, float, float, float]:
    """Control-variate estimator, agnostic to finance.

    Combines a noisy sample Y with a second sample X whose expectation is known
    exactly, forming Z = Y - c*(X - EX). Z has the same mean as Y and, for a
    well-chosen c, a smaller variance: Var(Z) = Var(Y)*(1 - rho^2).

    Params
    ------
    Y  : (n,) sample whose expectation is to be estimated.
    X  : (n,) control sample, paired with Y. X[i] and Y[i] must come from the
         SAME random draw, otherwise the correlation is zero and the whole
         device is pointless.
    EX : exact expectation of X, an analytical value rather than a Monte-Carlo
         estimate.
    c  : imposed coefficient. Estimated from the sample when None, as
         c* = Cov(Y, X) / Var(X).

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat) : four Python floats.
        estimate   : control-variate estimator of E[Y].
        half_width : 95% CI half-width of THAT estimator, computed on the
                     residual Z, not on Y.
        c_hat      : the coefficient used, estimated or imposed.
        rho_hat    : empirical correlation between Y and X, within [-1, 1].

    Notes
    -----
    * c = 0 reproduces the raw Monte-Carlo on Y exactly, mean and half-width.
    * Y = X with an exact EX returns EX to machine precision, with a
      numerically null half-width.
    * rho_hat is invariant under rescaling of X; c_hat is divided by the scale.
    * Estimating c on the same sample as the estimate leaves an O(1/N) bias, as
      c_hat and (X - EX) are then correlated. See `pilot_c`.
    * Var(X) = 0 is not guarded and yields nan.
    """
    n = len(Y)
    if c is None:
        c_hat = float((np.cov(Y,X,ddof=1)[0,1])/(np.var(X,ddof=1)))
    else:
        c_hat=c
    Z = Y - c_hat*(X - EX)
    estimate = float(Z.mean())
    half_width = float(1.96*Z.std(ddof=1)/np.sqrt(n))
    rho_hat = float(np.corrcoef(X,Y)[0,1])


    return estimate,half_width,c_hat,rho_hat


def gbm_paths_antithetic(S0: float,
                         sigma: float,
                         r: float,
                         T: float,
                         n_steps: int,
                         N: int = 50_000,
                         rng: np.random.Generator | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Antithetic pairs of GBM paths, multi-step.

    Params
    ------
    N : number of PAIRS, hence 2N simulated paths.

    Returns
    -------
    (up, down) : two (N, n_steps+1) arrays. up[i] and down[i] are built on the
        same gaussian draw up to its sign, and up[:, 0] == down[:, 0] == S0
        exactly. A single call to standard_normal feeds both, so that
        log(up) + log(down) is the deterministic drift path, doubled.
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T/n_steps
    Z = rng.standard_normal((N, n_steps))
    log_incr = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_incr2 = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-Z)
    log_paths = np.concatenate([np.zeros((N, 1)), np.cumsum(log_incr, axis=1)],axis=1,)
    log_paths2= np.concatenate([np.zeros((N, 1)), np.cumsum(log_incr2, axis=1)],axis=1,)
    return S0*np.exp(log_paths),S0*np.exp(log_paths2)


def control_variate_antithetic(Y_up: np.ndarray,
                               Y_down: np.ndarray,
                               X_up: np.ndarray,
                               X_down: np.ndarray,
                               EX: float,
                               c: float | None = None
                               ) -> tuple[float, float, float, float]:
    """Control variate applied to an antithetic sample.

    Pairs are formed first, yielding N i.i.d. observations, and the control is
    applied to those. The order matters: 2N antithetic draws are not 2N
    independent observations, and a CI computed over 2N points would be too
    optimistic by a factor sqrt(2).

    Params
    ------
    Y_up, Y_down : (N,) payoffs of the quantity of interest on the two branches
        of a pair; Y_up[i] and Y_down[i] share the draw.
    X_up, X_down : (N,) control payoffs, same pairing.
    EX : exact expectation of the control, unchanged by the pairing.
    c  : imposed coefficient, or None to estimate it.

    Returns
    -------
    (estimate, half_width, c_hat, rho_hat), same convention as
    `control_variate`. The half-width covers N observations, the number of
    pairs, not 2N.
    """
    Y_pair = (Y_up +Y_down)/2
    X_pair = (X_up +X_down)/2
    (estimate, half_width, c_hat, rho_hat)= control_variate(Y_pair,X_pair,EX,c)

    return (estimate, half_width, c_hat, rho_hat)


def pilot_c(Y_pilot: np.ndarray, X_pilot: np.ndarray) -> float:
    """Control coefficient estimated on an independent pilot sample.

    Estimating c on the sample it is then applied to leaves an O(1/N) bias,
    since c_hat and the sampling error (X_bar - EX) are correlated. Deriving c
    from a separate, discarded run makes it deterministic with respect to the
    final sample and removes that bias exactly.

    Params
    ------
    Y_pilot, X_pilot : (n_pilot,) paired observations from a short run
        (10 000 paths is typically enough), drawn independently of the final
        run. Reusing the pilot paths in the final run defeats the purpose.

    Returns
    -------
    float : c = Cov(Y, X) / Var(X). Knows nothing of EX or discounting; it is a
        ratio of moments, meant to be passed as the `c` argument of
        `control_variate`. Accuracy hardly matters: the variance is quadratic
        around the optimum, so a rough c costs a fraction of the gain, never a
        bias.
    """
    c = float(np.cov(Y_pilot,X_pilot,ddof=1)[0,1]/np.var(X_pilot,ddof=1))
    return c
