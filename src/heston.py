"""Single-asset Heston model: simulation, characteristic function, smile.

Three independent routes to the same prices, which is what makes the module
testable: an Euler discretisation of the joint (S, v) dynamics, a
semi-analytical European price obtained by Fourier inversion, and Black-Scholes
implied volatilities read off the latter.

Library-wide convention: the second return value of every estimator is a 95%
confidence half-width, 1.96 * sd / sqrt(n).

Notation, following Grzelak and Bouzoubaa-Osseiran:
    v0     INITIAL VARIANCE, not the volatility (v0 = sigma_0^2)
    kappa  mean-reversion speed
    theta  long-run VARIANCE, again not a volatility
    xi     vol-of-vol (often written sigma or eta elsewhere; xi here, so it is
           never confused with the Black-Scholes volatility)
    rho    correlation between the spot and variance Brownian motions
"""

import numpy as np
from scipy.integrate import quad
from implied_vol import implied_vol_call


def heston_paths(S0: float,
                 v0: float,
                 r: float,
                 T: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho: float,
                 n_steps: int,
                 N: int = 100_000,
                 rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Joint (spot, variance) paths under Heston, Euler scheme.

    Two correlated Brownian motions are built from a single (N, n_steps, 2)
    gaussian draw by a 2x2 Cholesky factorisation written out by hand:
    W_v = Z1 and W_S = rho*Z1 + sqrt(1-rho^2)*Z2. The time loop cannot be
    vectorised the way gbm_paths is, since step t+1 depends on v_t; it runs over
    n_steps, handling all N paths at once.

    The spot is simulated in log space, which preserves positivity exactly. Both
    the variance and the spot step read the SAME v_t: the scheme is explicit.

    Params
    ------
    S0    : initial spot.
    v0    : initial VARIANCE (v0 = sigma_0^2; a 20% vol means v0 = 0.04).
    r     : risk-free rate.
    T     : maturity.
    kappa : mean-reversion speed of the variance.
    theta : long-run VARIANCE.
    xi    : vol-of-vol.
    rho   : correlation of the two Brownian motions, in [-1, 1]. Negative on
        equities, which is what produces the skew.
    n_steps : number of discretisation steps.
    N     : number of paths.
    rng   : injected generator; no seed is ever captured inside.

    Returns
    -------
    (S, v) : two (N, n_steps+1) arrays. S[:, 0] == S0 and v[:, 0] == v0 exactly,
        and v is non-negative everywhere.

    Notes
    -----
    An Euler step on a CIR process goes below zero whenever the Feller condition
    2*kappa*theta > xi^2 is violated, and even when it holds if dt is coarse.
    The negative part is truncated here, both in the coefficients and in the
    stored state, so the array returned is directly usable downstream: a
    negative variance would become a nan at the first sqrt and propagate
    silently through a np.maximum(..., 0).

    Carrying the truncated value forward is the "absorption" variant of Lord,
    Koekkoek & van Dijk (2010), not their "full truncation", which keeps the
    untruncated value as the state and only truncates the coefficients. The two
    agree to ~1e-5 relative on parameters satisfying Feller, and diverge
    markedly when it is violated.
    """
    dt = T/n_steps
    S = np.zeros((N,n_steps+1))
    v = np.zeros((N,n_steps+1))
    S[:,0] = S0
    v[:,0] = v0
    if rng is None:
      rng = np.random.default_rng()
    Z = rng.standard_normal((N,n_steps,2))
    W_v = Z[:, :, 0]
    W_S = rho*Z[:, :, 0] + np.sqrt(1-rho**2)*Z[:, :, 1]
    v_brut = np.full(N, v0)
    for t in range(n_steps):
      v_plus = np.maximum(v_brut, 0.0)
      v_brut = v[:, t] + kappa*(theta - v_plus)*dt + xi*np.sqrt(v_plus*dt)*W_v[:,t]
      v[:, t+1] = np.maximum(v_brut, 0.0)
      incr = (r - 0.5*v_plus)*dt + np.sqrt(v_plus*dt)*W_S[:, t]
      S[:,t+1] = S[:,t]*np.exp(incr)
    return (S, v)


def heston_cf(u: np.ndarray | complex,
              S0: float,
              v0: float,
              r: float,
              T: float,
              kappa: float,
              theta: float,
              xi: float,
              rho: float) -> np.ndarray | complex:
    """Characteristic function of log(S_T) under Heston: E[exp(i*u*log S_T)].

    Written in Albrecher's "little trap" form. The original 1993 formulation,
    with the reciprocal g, makes the complex logarithm jump between branches for
    large T: prices come out right at T = 1 and nonsensical at T = 5. This form
    does not.

    Params
    ------
    u : real evaluation point, or array of points.
    (others) : see heston_paths.

    Returns
    -------
    complex, or complex array when u is an array.

    Notes
    -----
    xi^2 sits in a denominator, so xi = 0 is a division by zero. Below xi ~ 1e-5
    the term kappa*theta/xi^2 multiplies the difference of two nearly equal
    numbers and catastrophic cancellation takes over; the degenerate tests
    therefore use xi = 1e-3 and 1e-4, where the O(xi) convergence towards the
    gaussian limit is still measurable.
    """
    b = kappa - rho*xi*1j*u
    d = np.sqrt(b**2 + (xi**2)*(1j*u + u**2))
    g = (b-d)/(b+d)

    Duration = ((b-d)/(xi**2))*((1-np.exp(-d*T))/(1-g*np.exp(-d*T)))

    C = (r*1j*u*T) + (kappa*theta/(xi**2))*((b-d)*T - 2*np.log((1-g*np.exp(-d*T))/(1-g)))


    phi = np.exp(1j*u*np.log(S0) + Duration*v0 + C)

    return phi


def heston_call(S0: float,
                K: float,
                v0: float,
                r: float,
                T: float,
                kappa: float,
                theta: float,
                xi: float,
                rho: float) -> float:
    """European call price under Heston, by Fourier inversion of heston_cf.

    Follows the Heston 1993 / Gatheral decomposition C = S0*P1 - K*exp(-rT)*P2,
    where P1 and P2 are Gil-Pelaez integrals of a real part over u in (0, inf),
    evaluated with scipy.integrate.quad. P1 uses the share-measure
    characteristic function phi(u - i) / phi(-i), with phi(-i) = S0*exp(rT).

    The alternative route is the COS expansion of Fang & Oosterlee, which is
    faster but needs a truncation range derived from the cumulants.

    Params
    ------
    See heston_paths, plus K, the strike.

    Returns
    -------
    float : a Python float, not a np.float64, so that callers can json.dump it.

    Notes
    -----
    The integrand is singular at u = 0 through the 1/(i*u) factor. It has a
    finite limit there, but the quadrature can fail on it, hence the lower bound
    at 1e-8 rather than 0.

    quad's own error estimates are discarded. Verified over T in [0.25, 30] and
    xi up to 1.2: prices stay within the model-free bounds
    max(S0 - K*exp(-rT), 0) <= C <= S0, calls stay convex in K down to 1e-7, and
    no IntegrationWarning is raised. A lower bound violated is the typical
    symptom of a truncated integration range under-estimating P1.
    """

    val,erreur2 = quad(lambda u:(np.real(np.exp(-1j*u*np.log(K))*heston_cf(u,S0,v0,r,T,kappa,theta,xi,rho)/(1j*u))),1e-8,np.inf)
    P2 = 0.5 + val/np.pi
    valo,erreur1 = quad(lambda u:(np.real(np.exp(-1j*u*np.log(K))*heston_cf(u-1j,S0,v0,r,T,kappa,theta,xi,rho)/(1j*u*S0*np.exp(r*T)))),1e-8,np.inf)
    P1 = 0.5 + valo/np.pi
    return float(S0*P1 -K*np.exp(-r*T)*P2)


def heston_put(S0: float,
               K: float,
               v0: float,
               r: float,
               T: float,
               kappa: float,
               theta: float,
               xi: float,
               rho: float) -> float:
    """European put price under Heston, from call-put parity.

    No new integral: parity is model-free, it follows from no-arbitrage alone
    and holds here for the same reason it holds in bs.put_bs.

    Returns
    -------
    float : a Python float.
    """

    C = heston_call(S0,K,v0,r,T,kappa,theta,xi,rho)

    P = C + K*np.exp(-r*T) - S0
    return float(P)


def heston_smile(S0: float,
                 strikes: np.ndarray,
                 v0: float,
                 r: float,
                 T: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho: float) -> np.ndarray:
    """Black-Scholes implied volatilities of Heston calls, strike by strike.

    Implied volatility is a unit of price, not a model parameter: the sigma one
    must feed Black-Scholes to recover the observed price. Applied to Heston
    prices, it exposes the model's signature in the form the market quotes.

    Params
    ------
    strikes : (m,) strike grid.
    (others) : see heston_paths.

    Returns
    -------
    (m,) implied volatilities, in the order of `strikes`.

    Notes
    -----
    rho drives the SLOPE: rho < 0 gives a negative skew, OTM puts quoting richer.
    xi drives the CURVATURE: xi = 0 gives a flat smile. kappa and theta drive the
    term structure, the smile flattening with T at the reversion speed.

    Inversion goes through implied_vol_call, which expects a CALL price. Far from
    the money vega collapses and the Newton solver becomes unstable, which bounds
    the usable strike range rather than reflecting anything about Heston.

    Plot the result against log-moneyness log(K/F), not against K: two maturities
    are not comparable on a K axis.
    """
    iv = []
    for K in strikes:
      market_price = heston_call(S0,K,v0,r,T,kappa,theta,xi,rho)
      iv.append(implied_vol_call(S0, K, r, T, market_price, q=0.0, tol=1e-6, max_iter=100))
    return np.asarray(iv)
