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


def digital_call_bs(S0,K,sigma,r,T,q=0.0):
    """Cash-or-nothing digital call, closed form: exp(-rT) * N(d2).

    Pays 1 (in cash) if S_T > K at maturity, 0 otherwise.

    Returns
    -------
    float : discounted price.

    QUEST 1.5 -- digitals (call-spread replication) [30 XP]
    GOAL: price a cash-or-nothing digital through TWO independent routes --
        a closed form here, and a static call-spread replication right below
        (`digital_call_replication`) -- and check that they converge to each
        other.
    UNLOCKS: nothing else in Act I; closes the last piece of technical debt
        on the roadmap.
    VALIDATION: `tests/test_pricers.py::test_digital_call_bs_vs_mc` --
        compared against a discounted Monte-Carlo estimate of the indicator
        `1{S_T > K}`, within the 95% CI; and
        `test_digital_call_replication_converge_en_h2` -- the gap between
        this function and `digital_call_replication(h)` must shrink as
        `O(h^2)` when `h` decreases, tested as a SLOPE (convergence ratio),
        not as a threshold on a single value of `h`.
    HINT 1 (intuition): the digital payoff is a step function (0 then 1,
        jumping at `K`) -- exactly the derivative, up to sign, of a call's
        payoff with respect to ITS OWN STRIKE. `N(d2)` is that derivative,
        discounted.
    HINT 2 (structure): `d2` is the SAME `d2` as in `call_bs`
        (`d1 - sigma*sqrt(T)`) -- do not rewrite `d1` by hand, all the
        machinery already lives in `call_bs`; this function only needs the
        last piece.
    HINT 3 (formula): `digital_call_bs = exp(-r*T) * N(d2)`, with
        `d1 = (log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))` and
        `d2 = d1 - sigma*sqrt(T)` -- identical to the ones in `call_bs`.
    PITFALL: do not discount twice -- `N(d2)` is already a (risk-neutral)
        PROBABILITY, not a price; it is `exp(-r*T)` that turns it into a
        price, once, exactly as everywhere else in this module.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(d2)


def digital_call_replication(S0,h,K,sigma,r,T,q=0.0):
    """Digital call approximated by a tight call spread, in strike space.

    Params
    ------
    h : half-width of the spread, IN STRIKE SPACE -- note the argument
        position, mirrors `delta`'s `h` (spot bump), second positional slot,
        before `K`.

    Returns
    -------
    float : (call_bs(K-h) - call_bs(K+h)) / (2h), converging to
        `digital_call_bs` as h -> 0, error in O(h^2).

    QUEST 1.5 (continued) -- the static replication
    GOAL: build the digital's price the way a desk would without a quoted
        digital contract -- a very tight call spread, normalised by its
        width, rather than a closed form.
    VALIDATION: see `digital_call_bs` above.
    HINT 1 (intuition): being long a call at K-h and short a call at K+h
        reproduces a ramp rising from 0 to 1 between K-h and K+h -- dividing
        by the width `2h` normalises that ramp to a height of 1, like the
        true digital payoff, and the ramp tightens into a step as `h -> 0`.
    HINT 2 (structure): exactly `delta`, but the bump sits on `K` instead of
        `S0` -- same centred finite difference, same formula, a different
        axis.
    HINT 3 (formula): `(call_bs(S0,K-h,sigma,r,T,q) - call_bs(S0,K+h,sigma,r,T,q)) / (2*h)`.
    PITFALL: the sign -- a call's price DECREASES with strike (`dC/dK < 0`),
        so `C(K-h) > C(K+h)`, and it is indeed `(C(K-h) - C(K+h))/(2h)`, NOT
        the other way round, that gives a POSITIVE number. Second pitfall,
        different from `delta`'s: no Monte-Carlo noise here, but too small an
        `h` still degenerates the computation -- the difference of two
        nearly equal `call_bs` values divided by a tiny `h`, floating-point
        rounding that blows up. Too large an `h`, conversely, biases the
        result (the spread is no longer steep enough to approximate a step).
        The right order of magnitude is measured, not guessed.
    """
    d1= (np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return (call_bs(S0,K-h,sigma,r,T,q) - call_bs(S0,K+h,sigma,r,T,q)) / (2*h)
