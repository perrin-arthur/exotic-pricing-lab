"""Closed-form Black-Scholes prices and Greeks.

Everything is priced at t = 0 under the risk-neutral measure, with a continuous
dividend yield q.

Parameter convention, shared by the whole library:
    S0     spot at t = 0
    K      strike
    sigma  annualised volatility
    r      risk-free rate, continuously compounded
    T      maturity, in years
    q      continuous dividend yield (default 0.0)
"""

import numpy as np

from scipy.stats import norm


def call_bs(S0,K,sigma,r,T,q=0.0):
    """European call price.

    Returns
    -------
    float : discounted price.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) -np.exp(-r*(T))*K*norm.cdf(d2)


def put_bs(S0,K,sigma,r,T,q=0.0):
    """European put price, derived from call-put parity.

    The put being *defined* by the parity relation, checking parity against this
    function is not an independent test of `call_bs`. The dividend identity
    C(S0, q) == C(S0*exp(-qT), 0) is, and is what tests/test_pricers.py uses.

    Returns
    -------
    float : discounted price.
    """
    return call_bs(S0,K,sigma,r,T,q)+K*np.exp(-r*T) - S0*np.exp(-q*T)


def delta(S0,h,K,sigma,r,T,q=0.0):
    """Call delta, by central finite difference on the closed-form price.

    Params
    ------
    h : spot bump. Note the argument position: h comes second, before K.

    Returns
    -------
    float : (C(S0+h) - C(S0-h)) / (2h), an O(h^2) approximation of
        exp(-qT) * N(d1).
    """
    return (call_bs(S0+h,K,sigma,r,T,q) - call_bs(S0-h,K,sigma,r,T,q))/(2*h)


def vega(S0,K,sigma,r,T,q=0.0):
    """Call vega, dC/dsigma, in price units per unit of volatility.

    Returns
    -------
    float : collapses towards zero away from the money, which is what bounds the
        accuracy of the Newton solver in implied_vol.py.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S0*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
