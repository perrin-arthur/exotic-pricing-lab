"""Discrete hedging and replication P&L.

A price without a replicating portfolio is not validated: this module builds
the self-financed delta-hedge simulation that closes the loop opened by
bs.py / mc_engine.py / heston.py, and measures what breaks it (discretisation
noise, a wrong volatility, transaction costs, a barrier, an unhedged vega).

Library-wide conventions carried over: the second return value of an
estimator is a 95% confidence half-width, 1.96 * sd / sqrt(n). This module
takes already-simulated paths as input rather than drawing its own randomness,
so no function here owns an `rng` argument -- the paths passed in were built
with one, upstream, in gbm_paths / heston_paths.
"""

import numpy as np
import scipy

import bs

def portfolio_terminal_value(paths: np.ndarray,
                             deltas: np.ndarray,
                             r: float,
                             T: float,
                             V0: np.ndarray | float) -> np.ndarray:
    """Terminal value of a self-financed replicating portfolio.

    Params
    ------
    paths  : (N, n_steps+1) simulated price paths, S0 in column 0.
    deltas : (N, n_steps) shares held over [t_i, t_{i+1}); deltas[:, i] is
        decided at t_i and held until the next rebalancing date.
    r, T   : risk-free rate, maturity. dt = T / n_steps is read off
        paths.shape[1] - 1.
    V0     : initial portfolio value, (N,) or a scalar broadcastable to it --
        typically the option premium, so that a perfect hedge reproduces the
        payoff exactly.

    Returns
    -------
    (N,) terminal portfolio value V_T, one per path.

    QUEST 4.1 -- the self-financed portfolio [30 XP]
    GOAL: simulate the cash + shares account of a replication strategy,
        without assuming `deltas` is the right delta -- this is bookkeeping
        mechanics, not yet a pricing question.
    UNLOCKS: 4.2, and by transitivity the rest of the act -- this is the
        foundation `hedging_error` (4.2) is built on.
    VALIDATION: `tests/test_hedging.py::test_autofinancement_deltas_arbitraires`
        -- with RANDOMLY DRAWN deltas (positive, negative, huge), the
        self-financing recursion must hold to 1e-12. No pricing reference is
        involved: this is an algebraic identity, true for any sequence of
        deltas.
    HINT 1 (intuition): at every rebalancing date you buy or sell shares;
        that money comes from nowhere else than a cash account, which itself
        has grown at the risk-free rate since the last date. The portfolio
        (shares + cash) never receives nor loses money "from the outside"
        between two dates -- THAT is self-financing.
    HINT 2 (structure): keep two quantities alive in parallel over the loop
        of dates -- the number of shares held (given, `deltas[:, i]`) and a
        cash account `B` (to compute). `B` starts at `V0 - deltas[:, 0]*S0`.
        Between two dates, `B` grows at rate `r` over `dt`, THEN absorbs the
        (positive or negative) cost of the change in share position at the
        next date. At the very last date there is no more rebalancing: the
        position `deltas[:, -1]` is held until `S[:, -1]`.
    HINT 3 (formula): for i = 0 .. n_steps-2,
        `B_{i+1} = B_i*exp(r*dt) - (deltas[:,i+1]-deltas[:,i])*S[:,i+1]`;
        `V_T = deltas[:,-1]*S[:,-1] + B_{n_steps-1}*exp(r*dt)`.
    PITFALL: `paths` has `n_steps+1` columns, `deltas` has `n_steps` --
        `deltas[:, i]` is decided AT `t_i` and holds over `[t_i, t_{i+1})`, so
        `S[:, i+1]` is the price at which this change of position settles,
        not `S[:, i]`. A one-column shift between `paths` and `deltas`
        remains invisible with a "reasonable" delta (the error drowns in the
        hedging noise); it only becomes obvious with absurd deltas, exactly
        what 4.1 tests.
    """
    B = V0 - deltas[:, 0] * paths[:, 0]
    dt = T / (paths.shape[1] - 1)
    V_T = np.empty(paths.shape[0])
    for i in range(deltas.shape[1] - 1):
        B = B * np.exp(r * dt) - (deltas[:, i + 1] - deltas[:, i]) * paths[:, i + 1]
    V_T = deltas[:, -1] * paths[:, -1] + B * np.exp(r * dt)
    return V_T


def bs_delta_hedge_deltas(paths: np.ndarray,
                          K: float,
                          sigma: float,
                          r: float,
                          T: float,
                          option: str = "put") -> np.ndarray:
    """Black-Scholes delta at each rebalancing date, on shrinking maturity.

    Params
    ------
    paths  : (N, n_steps+1) simulated paths.
    K, sigma, r, T : as in bs.py.
    option : "call" or "put".

    Returns
    -------
    (N, n_steps) deltas, deltas[:, i] evaluated at t_i with remaining maturity
    tau = T - t_i.

    QUEST 4.2 -- hedging error under the correct model [50 XP]
    GOAL: produce, for every path and every rebalancing date, the BS delta
        computed on the REMAINING maturity -- not the initial maturity T,
        which would not vary from one column to the next.
    UNLOCKS: `hedging_error` then `hedging_error_stats`, and further down
        4.3, 4.4, 4.5 (which reuses the principle for the vanilla put), 4.6.
    VALIDATION: `test_bs_delta_hedge_deltas_vs_formule_fermee` -- at the
        first step (tau = T), compares deltas[:, 0] against N(d1) - 1 (put)
        computed by a closed form written directly in the test, independent
        of `bs.py`. Tolerance justified by the order of magnitude of a
        well-tuned finite difference (1e-3), not 1e-12 -- this is not an
        algebraic identity.
    HINT 1 (intuition): an option's delta is not a fixed number, it depends
        on how much time is left before expiry. At date t_i, T - t_i remains
        before maturity -- THAT is the time to feed into the delta formula,
        not T.
    HINT 2 (structure): `bs.delta(S0, h, K, sigma, r, T, q=0.0)` computes a
        CALL delta by finite difference -- watch the order of positional
        arguments (h before K, a pitfall already documented in bs.py). The
        PUT delta follows from parity: delta_put = delta_call - 1 (derivative
        of C - P = S - K*exp(-rT) with respect to S). Loop (or vectorise)
        over the n_steps columns of `paths[:, :-1]`, with a remaining
        maturity tau_i = T - i*dt at each column i.
    HINT 3 (formula): delta_put(S, tau) = N(d1(S, tau)) - 1, with
        d1(S, tau) = (log(S/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau)).
    PITFALL: at the LAST rebalancing date (i = n_steps - 1), tau = dt, not
        0 -- the portfolio only reaches tau = 0 at the very end, once the
        final position is already locked in (cf. 4.1, HINT 3). Passing
        tau = 0 here would give an indicator delta (0 or -1 depending on K
        vs S), not N(d1) - 1.
    """
    dt = T / (paths.shape[1] - 1)
    tau = T - np.arange(paths.shape[1] - 1) * dt
    if option == "put":
        return bs.delta(paths[:, :-1], 1e-4, K, sigma, r, tau) - 1
    elif option == "call":
        return  bs.delta(paths[:, :-1], 1e-4, K, sigma, r, tau)
    return np.full_like(paths[:, :-1], np.nan)  # pragma: defensive


def hedging_error(paths: np.ndarray,
                  deltas: np.ndarray,
                  K: float,
                  r: float,
                  T: float,
                  V0: float,
                  option: str = "put") -> np.ndarray:
    """Per-path replication error: portfolio minus payoff, at maturity.

    Params
    ------
    paths, deltas : as in portfolio_terminal_value.
    K, r, T, option : payoff parameters.
    V0 : initial portfolio value, passed through to portfolio_terminal_value.

    Returns
    -------
    (N,) array, V_T - payoff(S_T), undiscounted.

    QUEST 4.2 (continued) -- assembling the hedge
    GOAL: wire `bs_delta_hedge_deltas` into `portfolio_terminal_value` (4.1)
        then subtract the payoff actually owed at maturity.
    VALIDATION: `test_hedging_error_moyenne_dans_ic_et_pente_log_log` (see
        the main block on `bs_delta_hedge_deltas`).
    HINT 1 (intuition): V_T (4.1) is what YOUR replicating portfolio is worth
        at expiry; the payoff is what you OWE the option's holder. The gap
        between the two is your hedging error -- zero in continuous time,
        not under discrete rebalancing.
    HINT 2 (structure): `deltas = bs_delta_hedge_deltas(...)`,
        `V_T = portfolio_terminal_value(paths, deltas, r, T, V0)`, then
        `payoff = maximum(K - S_T, 0)` for a put (`maximum(S_T - K, 0)` for
        a call) with `S_T = paths[:, -1]`.
    HINT 3 (formula): `hedging_error = V_T - payoff`.
    PITFALL: do NOT discount here -- V_T and the payoff both live at
        maturity, discounting only makes sense if you want to compare to a
        price at t=0 (which `vol_arbitrage_pnl` does explicitly, in 4.3).
    """
    portofolio = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return portofolio - payoff




def hedging_error_stats(errors: np.ndarray) -> tuple[float, float]:
    """Mean and 95% CI half-width of a hedging error sample.

    Params
    ------
    errors : (N,) array, e.g. the output of hedging_error.

    Returns
    -------
    (mean, half_width) : same convention as every estimator in the library.

    QUEST 4.2 (continued) -- statistics and convergence rate
    GOAL: produce (mean, 95% CI half-width) on a sample of errors, EXACTLY
        like `pricer_mc_put` or `control_variate` -- no new formula, just the
        repo's convention applied to a new object.
    VALIDATION (the identity that closes 4.2):
        `test_hedging_error_moyenne_dans_ic_et_pente_log_log` --
        (a) on a single run, the mean error must sit inside its own 95% CI
            around zero (a correctly built BS hedge does not bias the
            replication);
        (b) on a grid of `n_steps` (e.g. 8, 16, 32, ..., 256), the error's
            standard deviation DECREASES, and the log-log regression of that
            standard deviation against `n_steps` has a slope close to -1/2.
            Tested as a SLOPE (tolerance on the regression), never as a
            numerical threshold on a single value -- a wrong slope betrays a
            discretisation error, a missed threshold could just be a poor
            choice of N.
    HINT 1 (intuition): the more often you rebalance, the closer your
        portfolio tracks the option -- but never perfectly, because between
        two dates the delta you hold is already slightly wrong (the spot has
        moved). Doubling the frequency does not halve the error, it divides
        it by the square root of two -- this is a discretisation error of a
        continuous process, not MC noise that would melt away by raising N.
    HINT 2 (structure): for each `n_steps` in the grid, simulate a NEW set of
        paths (`gbm_paths`), compute the hedging error on that set, take its
        standard deviation (`ddof=1`), and run a linear regression of
        `log(std)` against `log(n_steps)` (`np.polyfit(..., 1)`): the first
        coefficient is the slope you are after.
    HINT 3 (formula, independent reference): Boyle & Emanuel (1980) give the
        limiting variance of the discrete hedging error for a delta-neutral
        portfolio: it is of the order of
        `(pi/4) * (1/n_rebal) * E[ (Gamma(S_t,t) * S_t^2 * sigma^2 * dt)^2 ]`
        summed over the steps, which gives a standard deviation in
        `O(1/sqrt(n_rebal))` -- the -1/2 slope to test. Used ONLY as the
        theoretical justification for the expected slope; the test does not
        compute this formula, it measures the empirical slope.
    PITFALL: do not reuse the SAME `rng` for every `n_steps` in the grid
        thinking it gives you CRN -- `gbm_paths` consumes `N * n_steps`
        draws, so two calls with different `n_steps` and the same generator
        do NOT walk over the same underlying trajectories. This is not a
        problem here (the runs are not compared term by term, only their
        standard deviations), but remember it so as not to assume it
        elsewhere in the act.
    """
    mean = np.mean(errors)
    half_width = 1.96 * np.std(errors, ddof=1) / np.sqrt(errors.size)
    return mean, half_width


def bs_gamma(S: np.ndarray,
            K: float,
            sigma: float,
            r: float,
            tau: np.ndarray | float,
            q: float = 0.0) -> np.ndarray:
    """Closed-form Black-Scholes gamma, d2(price)/dS2.

    Identical for call and put (gamma is parity-invariant: d2C/dS2 = d2P/dS2
    since C - P is affine in S). Not in bs.py, which stays untouched; needed
    only for the vol-arbitrage identity of QUEST 4.3.

    Params
    ------
    S   : spot, scalar or array.
    K, sigma, r : as in bs.py.
    tau : time to maturity remaining, scalar or array broadcastable with S.
    q   : continuous dividend yield.

    Returns
    -------
    Gamma, same shape as the broadcast of S and tau.

    QUEST 4.3 -- vol arbitrage / the wrong model [50 XP]
    GOAL: write Black-Scholes' closed-form gamma -- the only missing piece to
        close 4.3's identity, since `bs.py` only has delta and vega.
    UNLOCKS: `vol_arbitrage_pnl`, below.
    VALIDATION: `test_bs_gamma_vs_difference_finie_sur_delta` -- gamma is the
        derivative of delta with respect to spot; compared against a finite
        difference of `bs.delta` (hence independent of the closed form
        itself), tolerance 1e-4 (O(h^2) error of a well-tuned centred finite
        difference).
    HINT 1 (intuition): delta moves when spot moves -- gamma measures HOW
        FAST. A high gamma means an unstable delta: this is exactly what
        makes a discrete hedge imperfect (4.2), and it is this very
        parameter that 4.3 will weight by the variance gap between hedge vol
        and realised vol.
    HINT 2 (structure): gamma only depends on d1 (not d2, not N(d1) itself)
        -- it is the gaussian DENSITY evaluated at d1, scaled by S, sigma and
        the square root of tau. A density, not a random draw:
        `scipy.stats.norm.pdf`, not `np.random.*`.
    HINT 3 (formula): gamma = exp(-q*tau)*phi(d1) / (S*sigma*sqrt(tau)),
        where phi is the N(0,1) density and
        d1 = (log(S/K) + (r - q + 0.5*sigma^2)*tau) / (sigma*sqrt(tau)) --
        the same d1 as in bs.py, with tau in place of T.
    PITFALL: gamma blows up as tau -> 0 for an at-the-money option (division
        by sqrt(tau) tending to 0) -- expected, not a bug; it already
        foreshadows the structural problem of 4.5 (delta near a barrier),
        where an exploding gamma is precisely what breaks discrete hedging.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-q * tau) * scipy.stats.norm.pdf(d1) / (S * sigma * np.sqrt(tau))



def vol_arbitrage_pnl(paths: np.ndarray,
                      K: float,
                      sigma_impl: float,
                      r: float,
                      T: float,
                      V0: float,
                      option: str = "put") -> np.ndarray:
    """Discounted P&L of hedging at sigma_impl while paths realise a different vol.

    Params
    ------
    paths : (N, n_steps+1) simulated paths -- however they were generated
        (any realised volatility; this function only hedges, it does not
        simulate).
    K, r, T, option : payoff parameters.
    sigma_impl : volatility used both to compute the delta (bs_delta_hedge_deltas)
        and to price V0 (typically bs.put_bs(S0, K, sigma_impl, r, T)).
    V0 : initial portfolio value.

    Returns
    -------
    (N,) discounted P&L, exp(-r*T) * (V_T - payoff(S_T)).

    QUEST 4.3 -- vol arbitrage / the wrong model [50 XP]
    GOAL: measure the replication P&L when the vol used to hedge is not the
        one realised by the market -- the classic interview question, "you
        mark at what vol, and the market realises a different one, what
        happens?".
    UNLOCKS: nothing downstream in the act (an independent branch from
        4.4/4.5/4.6), but this is the most-quoted quest in a structuring
        interview.
    VALIDATION: `test_vol_arbitrage_identite_gamma` -- the P&L's expectation
        (mean over N paths, compared to its 95% CI) must coincide with the
        discrete sum of
        `-exp(-r*t_i) * 0.5 * gamma(S_i, tau_i) * S_i^2 * (sigma_real^2 - sigma_impl^2) * dt`
        along the paths (MINUS sign, and `exp(-r*t_i)` INSIDE the sum -- see
        PITFALL, this factor broke the test once). This is an IDENTITY (the
        mean P&L must fall inside the reference's CI), not a sign
        inequality -- an identity is a far harder test to satisfy by
        accident than a simple "the sign is right".
    HINT 1 (intuition): delta-hedging neutralises the FIRST order (the
        spot's linear move), never the second (curvature, gamma). If you
        hedged assuming too low a vol, you are structurally UNDER-hedged in
        gamma: every spot move a bit larger than expected costs you, or
        earns you, money depending on the sign of the vol gap. This is
        "gamma scalping" in reverse.
    HINT 2 (structure): the function itself does nothing more than
        `hedging_error` (4.2), discounted -- all the subtlety lives in the
        TEST, not the implementation: it must rebuild an independent
        reference by summing `bs_gamma` along the paths, with the TRUE
        volatility of the paths (`sigma_real`, known to the test since it
        simulated the paths) and the hedge volatility (`sigma_impl`, passed
        to this function).
    HINT 3 (formula): set e_t = Pi_t - V_t (portfolio minus the price
        marked-to-model at sigma_impl). Writing Ito on e_t and substituting
        theta via the Black-Scholes PDE, the delta*dS terms cancel
        (delta = dV/dS) but NOT all the r terms -- what remains is
        `de_t = r*e_t*dt - 0.5*Gamma_t*S_t^2*(sigma_real^2 - sigma_impl^2)*dt`,
        a linear ODE in e_t (not just a term to integrate as-is: e_t
        compounds itself at rate r). With e_0 = 0 (Pi_0 = V0 by
        construction), solving gives
        `e_T = - integral_0^T exp(r*(T-s)) * 0.5*Gamma_s*S_s^2*(sigma_real^2 - sigma_impl^2) ds`,
        i.e., discounted,
        `mean discounted P&L ~= E[ - integral_0^T exp(-r*s) * 0.5*Gamma_s*S_s^2*(sigma_real^2 - sigma_impl^2) ds ]`
        where Gamma_s is evaluated at sigma_impl (the HEDGER's vol, the one
        defining his delta/gamma model -- not sigma_real, which he does not
        know).
    PITFALL: TWO sign/factor pitfalls stacked here, hit in this order while
        writing this module.
        (1) the sign -- MINUS, not PLUS. `V0` puts you on the SELLER's side
        (you collect `V0`, you owe the payoff at maturity), so your
        replicating portfolio is short gamma. If sigma_real > sigma_impl
        (the market moves more than expected), a short-gamma book LOSES
        money -- "selling vol means betting the market will not move more
        than priced in".
        (2) the `exp(-r*s)` factor MUST sit INSIDE the integral, term by
        term -- not just `exp(-r*T)` pulled out as a global factor in front
        of the whole sum. The reason: the hedging error accumulated at time
        s compounds itself at rate r up to T (that is the `r*e_t*dt` term of
        the ODE above); forgetting it leaves a systematic gap of a few
        percent that does NOT SHRINK as n_steps grows -- this is not a
        discretisation bias that fades away, it is a missing term in the
        formula, a trickier pitfall than the sign one because the result
        stays plausible (right order of magnitude, right sign) at every
        test scale.
        Third pitfall, documented in the prompt itself: the P&L is
        path-dependent (every path has its own P&L, sometimes far from
        zero), even though its EXPECTATION only depends on sigma_real and
        sigma_impl -- do not confuse "the identity holds on average" with
        "every path sits close to the reference".
    """
    delta = bs_delta_hedge_deltas(paths, K, sigma_impl, r, T, option=option)
    V_T = portfolio_terminal_value(paths, delta, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return np.exp(-r * T) * (V_T - payoff)


def turnover(deltas: np.ndarray) -> np.ndarray:
    """Total absolute share turnover along a hedge, per path.

    Params
    ------
    deltas : (N, n_steps) as in portfolio_terminal_value. delta_{-1} := 0, so
        the initial purchase of deltas[:, 0] shares counts as turnover too.

    Returns
    -------
    (N,) sum_i |delta_i - delta_{i-1}|, in shares.

    QUEST 4.4 -- transaction costs [40 XP]
    GOAL: measure, per path, how many shares in total changed hands along
        the hedge -- the base brick, reused as-is by `transaction_costs`
        ($-weighted) and by 4.5 (as a raw diagnostic, with no cost attached).
    UNLOCKS: `transaction_costs`, `hedging_error_with_costs`.
    VALIDATION: `test_turnover_deltas_constants_et_alternes` -- constant
        deltas (turnover = |delta_0|, a single purchase, never rebalanced
        afterwards) and deltas alternating sign at every step (turnover =
        sum of all the |gaps|, a case where nothing cancels): two closed
        identities, computable by hand.
    HINT 1 (intuition): turnover only looks at the SHARES traded, not their
        price -- it is `transaction_costs` that, afterwards, weights each
        trade by the price at which it happens.
    HINT 2 (structure): `deltas` has `n_steps` columns; there are `n_steps`
        trades in total -- the very first one (the initial purchase of
        `deltas[:, 0]` shares, since you started from zero shares) then
        `n_steps - 1` rebalancings between consecutive columns.
    HINT 3 (formula): `turnover = |deltas[:,0]| + sum_i |deltas[:,i+1] - deltas[:,i]|`
        for `i = 0 .. n_steps-2`.
    PITFALL: do not confuse "total turnover over the whole path" (what THIS
        function returns, a scalar per path) with "turnover at EACH date" (a
        vector per path, `(N, n_steps)`) -- `transaction_costs` needs the
        second one to weight each trade by the price AT THAT MOMENT, not the
        first one: summing in shares first and then multiplying by a global
        price makes no sense (which price would you even pick?), the order
        of operations matters.
    """
    turnover = np.abs(deltas[:, 0])  # initial purchase counts
    turnover += np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)
    return turnover


def transaction_costs(paths: np.ndarray,
                      deltas: np.ndarray,
                      cost_rate: float) -> np.ndarray:
    """Proportional transaction cost on delta turnover, in dollar terms.

    Params
    ------
    paths, deltas : as in portfolio_terminal_value.
    cost_rate : proportional cost, e.g. 0.001 for 10 bps per dollar traded.

    Returns
    -------
    (N,) total undiscounted cost paid along the path,
    cost_rate * sum_i |delta_i - delta_{i-1}| * S_i.

    QUEST 4.4 (continued) -- weighting turnover by price
    GOAL: turn a turnover in SHARES (`turnover`, above) into a cost in
        DOLLARS, by weighting each trade by the price at which it actually
        happens.
    VALIDATION: `test_transaction_costs_frequence_optimale` -- see the main
        block on `hedging_error_with_costs`.
    HINT 1 (intuition): every rebalancing happens AT A SPECIFIC PRICE, the
        spot on that date -- not an average price, not the final price. The
        cost of each trade is written BEFORE summing over dates, not after.
    HINT 2 (structure): unlike `turnover`, which immediately collapses to a
        scalar per path, here you need to keep an `(N, n_steps)` array of
        turnover PER DATE -- `|deltas[:,0]|` in column 0, then
        `|diff(deltas, axis=1)|` for the following columns (same content as
        in `turnover`, but WITHOUT summing over the date axis before
        multiplying by the price).
    HINT 3 (formula): `turnover_per_date = concatenate([|deltas[:,0:1]|, |diff(deltas, axis=1)|], axis=1)`,
        an `(N, n_steps)` array; then
        `cost = cost_rate * sum(turnover_per_date * paths[:, :-1], axis=1)`
        (`paths[:, :-1]`, NOT `paths[:, 1:]` -- the trade at date `i` settles
        at price `S_i`, the one showing AT THAT DATE, not the next one).
    PITFALL: calling `turnover(deltas)` here and multiplying the result (a
        scalar per path) by `paths[:, :-1]` (a matrix per path) does not
        work -- these are two different shapes of the same computation, the
        aggregation (summing over dates) must happen AFTER the price
        weighting, not before. `turnover` answers "how many shares in
        total?", `transaction_costs` needs "how many shares, at EACH date,
        at WHAT price?" -- two different questions, despite the similar
        name.
    """
    turnover = np.abs(deltas[:, 0])  # initial purchase counts
    turnover = turnover[:, np.newaxis]  # shape (N, 1) for broadcasting
    turnover = np.concatenate([turnover, np.abs(np.diff(deltas, axis=1))], axis=1)
    return cost_rate * np.sum(turnover * paths[:, :-1], axis=1)


def hedging_error_with_costs(paths: np.ndarray,
                             deltas: np.ndarray,
                             K: float,
                             r: float,
                             T: float,
                             V0: float,
                             cost_rate: float,
                             option: str = "put") -> np.ndarray:
    """Hedging error (as in hedging_error) net of proportional transaction costs.

    Params
    ------
    paths, deltas, K, r, T, V0, option : as in hedging_error.
    cost_rate : as in transaction_costs.

    Returns
    -------
    (N,) array, undiscounted.

    QUEST 4.4 (continued) -- the optimal frequency
    GOAL: show that an optimal `n_steps` exists, neither too rare (high
        replication error) nor too frequent (costs dominate) -- the concrete
        trade-off a desk actually makes.
    VALIDATION (the identity that closes 4.4):
        `test_transaction_costs_frequence_optimale` -- on a grid of
        increasing `n_steps` (same paths as 4.2, fixed cost_rate):
        (a) the mean transaction cost GROWS as `sqrt(n_rebal)` (log-log slope
        ~= +1/2, tested with the same tolerance as 4.2's -1/2 slope);
        (b) the hedging error's standard deviation WITHOUT costs keeps
        decreasing as in 4.2; (c) combining the two into a total error
        measure (e.g. RMS of `hedging_error_with_costs`), there is an
        intermediate `n_steps` that minimises that measure -- neither the
        smallest nor the largest in the grid.
    HINT 1 (intuition): rebalancing more often reduces the replication error
        (4.2) but every rebalancing costs something (4.4) -- the two effects
        move in opposite directions as `n_steps` grows, so somewhere in the
        middle their sum is minimal. This is Leland's (1985) argument:
        beyond an optimal frequency, hedging more often costs more than it
        gains in precision.
    HINT 2 (structure): this function only assembles what already exists --
        `portfolio_terminal_value` (4.1), the payoff (as in `hedging_error`,
        4.2), and `transaction_costs` (above) -- all the substance of the
        quest lives in the TEST, as for 4.2/4.3.
    HINT 3 (formula, independent reference): Leland (1985) adjusts the hedge
        vol by a term
        `sigma_Leland^2 = sigma^2 * (1 + sqrt(2/pi) * k / (sigma*sqrt(dt)))`,
        where `k` is the proportional cost rate -- the expected total cost
        grows as `E[turnover] * S * k`, and `E[turnover]` itself grows as
        `sqrt(n_rebal)` (a random walk of the delta, whose increments have a
        standard deviation in `sqrt(dt) = sqrt(T/n_rebal)`, summed over
        `n_rebal` independent steps). Given as a theoretical reference for
        the expected slope; the test does not compute this formula, it
        measures the empirical slope of the cost.
    PITFALL: `hedging_error_with_costs` returns V_T - payoff - costs, so a
        high `cost_rate` makes the error SYSTEMATICALLY negative (on
        average) -- it is no longer centred at zero as in 4.2, and that is
        EXPECTED: costs are a certain loss, not noise. Do not reuse 4.2's
        "mean inside the CI around zero" test here as-is.
    """
    V_T = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    costs = transaction_costs(paths, deltas, cost_rate)
    return V_T - payoff - costs


def naive_barrier_hedge_error(paths: np.ndarray,
                              K: float,
                              H: float,
                              sigma: float,
                              r: float,
                              T: float,
                              V0: float,
                              barrier: str = "do",
                              option: str = "put") -> np.ndarray:
    """Discrete hedge of a DO/DI put using the vanilla put's BS delta.

    Params
    ------
    paths : (N, n_steps+1) simulated paths, used both to hedge (BS delta at
        each column) and to read the true DO/DO payoff at maturity via
        barriers.do_put / barriers.di_put logic.
    K, H, sigma, r, T : strike, barrier, vol used for the delta, rate, maturity.
    V0 : initial portfolio value -- the true barrier option's price, not the
        vanilla's.
    barrier : "do" or "di".
    option  : "call" or "put".

    Returns
    -------
    (N,) hedging error, undiscounted.

    QUEST 4.5 -- delta near a barrier [40 XP]
    GOAL: measure what breaks when hedging a BARRIER option with the ONLY
        delta available in closed form -- the vanilla's, on the same K, T.
        The true delta of a DO/DI put (the derivative of the BARRIER PRICE
        with respect to spot) has no closed form here (discrete monitoring,
        cf. README "Not covered"); this is the mismatch a desk faces the
        moment no barrier Greek is available.
    UNLOCKS: nothing downstream (an independent branch from 4.6), but this
        is THE "real business problem" quest of the act.
    VALIDATION: `test_naive_barrier_hedge_degrade_pres_de_la_barriere` -- no
        closed-form reference (hence no identity), the test is about what
        MUST break: at fixed `n_steps`, the hedge error's standard deviation
        INCREASES sharply as `H` gets closer to `S0`, compared to an `H` far
        from spot. A test that proves a DEGRADATION is worth as much as one
        that proves convergence (cf. 4.2) -- it is only about the direction
        of a statistic, not a bare numerical threshold.
    HINT 1 (intuition): the vanilla delta never "sees" the barrier -- it
        varies smoothly with S, while the TRUE price of a DO/DI put has a
        slope that changes abruptly whenever S crosses H (the payoff itself
        is discontinuous path by path: `max(K-S_T,0)` multiplied by an
        indicator that flips between 0 and 1). The closer H is to S0, the
        more often paths cross H, the more structurally off the vanilla
        delta is at those moments.
    HINT 2 (structure): `deltas = bs_delta_hedge_deltas(paths, K, sigma, r, T, option)`
        (exactly as in 4.2, NO dependency on `H` in the delta computation --
        that is the point); `V_T = portfolio_terminal_value(paths, deltas, r, T, V0)`;
        then the TRUE barrier payoff (not the vanilla payoff) from
        `paths.min(axis=1)` and the `barrier` condition.
    HINT 3 (formula): vanilla payoff `p = maximum(K - S_T, 0)` (put) or
        `maximum(S_T - K, 0)` (call); `min_path = paths.min(axis=1)`;
        DO -- payoff = `where(min_path >= H, p, 0)` (survives as long as the
        barrier is never touched, the same strict convention as in
        `barriers.py`); DI -- payoff = `where(min_path < H, p, 0)`.
        Error = `V_T - payoff`.
    PITFALL: `V0` must be the price of the TRUE DO/DI put (typically
        `barriers.do_put(paths, K, H, r, T)[0]` or `di_put(...)`, on the
        SAME paths), not `bs.put_bs(...)` -- funding the replication with
        the wrong starting price would bias the error by a constant term
        that has nothing to do with the delta's degradation, exactly the
        same pitfall as the hard-coded `sigma` spotted in 4.2.
    """
    deltas = bs_delta_hedge_deltas(paths, K, sigma, r, T, option)
    V_T = portfolio_terminal_value(paths, deltas, r, T, V0)
    S_T = paths[:, -1]
    payoff_vanilla = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    min_path = paths.min(axis=1)
    if barrier == "do":
        payoff_barrier = np.where(min_path >= H, payoff_vanilla, 0)
    elif barrier == "di":
        payoff_barrier = np.where(min_path < H, payoff_vanilla, 0)
    else:
        raise ValueError("barrier must be 'do' or 'di'")
    return V_T - payoff_barrier

def heston_delta_bs_hedge_error(S: np.ndarray,
                                K: float,
                                sigma_hedge: float,
                                r: float,
                                T: float,
                                V0: float,
                                option: str = "put") -> np.ndarray:
    """Discrete BS-delta-only hedge of paths simulated under Heston.

    Params
    ------
    S : (N, n_steps+1) HESTON spot paths (heston.heston_paths, first array
        only -- the variance path is not used here).
    K, sigma_hedge, r, T : delta and pricing parameters, a single constant
        vol standing in for the whole (flat, wrong) hedge.
    V0 : initial portfolio value.
    option : "call" or "put".

    Returns
    -------
    (N,) hedging error, undiscounted.

    QUEST 4.6 -- delta-vega under Heston [50 XP]
    GOAL: hedge with the BS delta ALONE (a flat, constant vol) on paths
        simulated under a model that has a real smile -- show that the hedge
        is biased AND runs at high variance, for lack of any vega covered.
    UNLOCKS: `heston_delta_vega_hedge_error`, below (which adds the vega
        overlay and must bring the variance down).
    VALIDATION: `test_heston_delta_vega_reduit_la_variance` -- see the main
        block on `heston_delta_vega_hedge_error`.
    HINT 1 (intuition): the BS delta assumes a CONSTANT vol, in time and in
        space; under Heston, vol is STOCHASTIC (it has its own source of
        randomness, correlated to spot through `rho`). A hedge that only
        looks at spot lets all the risk carried by the vol's own moves run
        free -- that is exactly what a vega measures, and this hedge has
        none.
    HINT 2 (structure): structurally identical to `hedging_error` (4.2) --
        `bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)`,
        `portfolio_terminal_value(S, deltas, r, T, V0)`, payoff at maturity
        -- the only difference: `S` comes from `heston.heston_paths`, not
        `gbm_paths`.
    HINT 3 (formula): nothing new, this is simply 4.2 wired onto Heston
        paths instead of GBM.
    PITFALL: `V0` must be consistent with `sigma_hedge` -- typically
        `bs.put_bs(S0, K, sigma_hedge, r, T)`, NOT `heston.heston_put(...)`
        (which would give the model's true price, not the flat, wrong one
        the hedger believes he is using). Mixing the two would make
        EXACTLY the bias you are trying to measure disappear.
    """
    deltas = bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)
    V_T = portfolio_terminal_value(S, deltas, r, T, V0)
    S_T = S[:, -1]
    payoff = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    return V_T - payoff


def heston_delta_vega_hedge_error(S: np.ndarray,
                                  K: float,
                                  K_vega: float,
                                  sigma_hedge: float,
                                  r: float,
                                  T: float,
                                  V0: float,
                                  option: str = "put") -> np.ndarray:
    """Same hedge as heston_delta_bs_hedge_error, plus a static vega overlay.

    Params
    ------
    S : (N, n_steps+1) Heston spot paths.
    K : target option's strike.
    K_vega : strike of the vanilla instrument added to the book, held in a
        fixed quantity computed once at t=0 to neutralise vega -- a static
        overlay, not a dynamically rebalanced vega hedge.
    sigma_hedge, r, T, V0, option : as in heston_delta_bs_hedge_error.

    Returns
    -------
    (N,) hedging error of the delta+static-vega book, undiscounted.

    QUEST 4.6 (continued) -- the vega overlay
    GOAL: add, ON TOP OF the dynamic delta hedge of
        `heston_delta_bs_hedge_error`, a STATIC position (locked in at t=0,
        never rebalanced) in a vanilla on another strike, sized to cancel
        the position's vega at t=0 -- and show that the residual variance
        drops.
    VALIDATION (the identity that closes 4.6):
        `test_heston_delta_vega_reduit_la_variance` -- at identical N and
        n_steps, `Var(heston_delta_bs_hedge_error) / Var(heston_delta_vega_hedge_error) > 1`,
        with a confidence interval on that ratio (bootstrap or an
        approximate F-test) -- not just "smaller", a VARIANCE RATIO
        significantly above 1, at an equal path budget.
    HINT 1 (intuition): you cannot rebalance vega continuously as easily as
        delta (you would have to retrade the hedging option at every step,
        which has a cost and falls outside this quest's scope) -- but even a
        STATIC vega hedge, put on once at t=0 and never touched again,
        absorbs a good part of the vol risk that escaped the delta-only
        hedge.
    HINT 2 (structure): at t=0, compute
        `n_vega = bs.vega(S0, K, sigma_hedge, r, T) / bs.vega(S0, K_vega, sigma_hedge, r, T)`
        (the ratio of the two options' BS vegas, at the hedge vol -- how
        many units of the `K_vega` option are needed to match the target's
        vega). The total portfolio then holds, at EVERY date, the target's
        BS delta PLUS `n_vega` times the `K_vega` instrument's BS delta
        (both computed by `bs_delta_hedge_deltas`, same `sigma_hedge`, each
        on its own strike) -- `n_vega` is FIXED, computed once, but its
        DELTA keeps being rebalanced at every date just like the target's
        (pitfall below). The starting cash must fund the purchase of the
        `n_vega` units of the overlay, on top of replicating the target.
    HINT 3 (formula): `deltas_total = bs_delta_hedge_deltas(S,K,...) - n_vega*bs_delta_hedge_deltas(S,K_vega,...)`
        -- SUBTRACTED, not added (see PITFALL 1);
        `V0_total = V0 - n_vega*bs.put_bs(S0,K_vega,sigma_hedge,r,T)` (or
        `call_bs`, depending on `option`); `payoff_total = payoff(K) - n_vega*payoff(K_vega)`
        (you COLLECT the payoff of the overlay you hold, it offsets part of
        what you owe on the target); error =
        `portfolio_terminal_value(S, deltas_total, r, T, V0_total) - payoff_total`.
    PITFALL: two pitfalls here.
        (1) the sign of the overlay's delta -- `deltas_total` SUBTRACTS
        `n_vega*delta_Kvega`, does not add it. Reason: `bs_delta_hedge_deltas`
        returns the delta to be used directly to replicate a SHORT position
        (the convention already validated in 4.2 -- `deltas` wired as-is
        into `portfolio_terminal_value` with `V0` = premium collected). The
        overlay, however, is a LONG position (you BUY `n_vega` units of the
        `K_vega` option) -- a long position with delta `d` needs `-d` shares
        to be delta-neutral, not `+d`. Getting this sign wrong does not
        degrade the variance, it INCREASES it sharply (measured: a variance
        ratio below 1 instead of clearly above) -- if your
        `test_heston_delta_vega_reduit_la_variance` fails with a ratio well
        below 1, check this sign first.
        (2) "static" qualifies the QUANTITY `n_vega` (computed once, at
        t=0), NOT the hedging instrument's delta -- its delta still varies
        with S and tau like any BS delta, and must be rebalanced at every
        date exactly like the target's. A "truly static" overlay (delta
        never recomputed either) would let a residual delta run free and
        would invalidate the comparison: the quest only tests the variance
        reduction if the total delta stays correctly tracked at every step.
    """
    deltas_total = bs_delta_hedge_deltas(S, K, sigma_hedge, r, T, option)
    deltas_overlay = bs_delta_hedge_deltas(S, K_vega, sigma_hedge, r, T, option)
    n_vega = bs.vega(S[:, 0], K, sigma_hedge, r, T) / bs.vega(S[:, 0], K_vega, sigma_hedge, r, T)
    deltas_total -= n_vega[:, np.newaxis] * deltas_overlay
    V0_total = V0 - n_vega * bs.put_bs(S[:, 0], K_vega, sigma_hedge, r, T) if option == "put" else V0 - n_vega * bs.call_bs(S[:, 0], K_vega, sigma_hedge, r, T)
    V_T = portfolio_terminal_value(S, deltas_total, r, T, V0_total)
    S_T = S[:, -1]
    payoff_target = np.maximum(K - S_T, 0) if option == "put" else np.maximum(S_T - K, 0)
    payoff_overlay = np.maximum(K_vega - S_T, 0) if option == "put" else np.maximum(S_T - K_vega, 0)
    payoff_total = payoff_target - n_vega * payoff_overlay
    return V_T - payoff_total
