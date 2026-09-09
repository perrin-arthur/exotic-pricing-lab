import bs

import numpy as np

def implied_vol_call(S0, K, r, T, market_price, *, q=0.0, tol=1e-6, max_iter=100):

    lower = np.maximum(0.0, S0*np.exp(-q*T) - K*np.exp(-r*T))
    upper = S0*np.exp(-q*T)
    if not (lower <= market_price <= upper):
        raise ValueError(f"out of [{lower:.4f}, {upper:.4f}]")

    # Manaster-Koenig :
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

        sigma = min(max(sigma - diff/vega, 1e-8), 5.0)  # Newton-Raphson step
    raise RuntimeError(f"no convergence after {max_iter} iterations")
