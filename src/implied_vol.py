"""Black-Scholes implied volatility by Newton-Raphson inversion."""

import bs

import numpy as np

def implied_vol_call(S0, K, r, T, market_price, *, q=0.0, tol=1e-6, max_iter=100):
    """Invert a European call price into its Black-Scholes implied volatility.

    Params
    ------
    market_price : call price to invert. Must lie within the model-free
        no-arbitrage bounds [max(0, S0*exp(-qT) - K*exp(-rT)), S0*exp(-qT)].
    tol : tolerance on the Newton step, i.e. on sigma, not on the price.
    max_iter : iteration cap before giving up.

    Returns
    -------
    float : implied volatility, clamped to [1e-8, 5.0].

    Raises
    ------
    ValueError : market_price outside the no-arbitrage bounds.
    RuntimeError : vega collapsed below 1e-10 (too far from the money for
        Newton to be stable), or no convergence within max_iter.
    """

    lower = np.maximum(0.0, S0*np.exp(-q*T) - K*np.exp(-r*T))
    upper = S0*np.exp(-q*T)
    if not (lower <= market_price <= upper):
        raise ValueError(f"out of [{lower:.4f}, {upper:.4f}]")

    # Manaster-Koenig seed: the starting point that maximises vega, hence the
    # most stable one for Newton. It degenerates at the money, where the
    # Brenner-Subrahmanyam approximation takes over below.
    sigma0 = np.sqrt( (2/T) * np.abs(np.log( S0*np.exp(-q*T) / (K*np.exp(-r*T)) )) )

    if sigma0<1e-4:
        sigma0 = np.sqrt(2*np.pi/T) * market_price / (S0*np.exp(-q*T))
    sigma = min(max(sigma0, 1e-8), 5.0)

    for i in range(max_iter):
        price= bs.call_bs(S0, K, sigma, r, T, q)
        diff = price - market_price
        vega = bs.vega(S0, K, sigma, r, T, q)
        if  vega < 1e-10:
            raise RuntimeError(f"vega degenerated from sigma={sigma:.6f}, K/S={K/S0:.2f}")

        if abs(diff/vega) < tol:
            pas = diff/vega
            sigma = min(max(sigma - pas, 1e-8), 5.0)

            return sigma

        sigma = min(max(sigma - diff/vega, 1e-8), 5.0)
    raise RuntimeError(f"no convergence after {max_iter} iterations")
