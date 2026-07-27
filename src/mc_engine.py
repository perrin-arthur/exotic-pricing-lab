import bs
import numpy as np

from scipy.stats import norm

#gbm
def gbm(S0,sigma,r,T, N=100_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return S0*np.exp((r-0.5*sigma**2)*(T)+sigma*np.sqrt(T)*rng.standard_normal(N))

#monte-carlo mtn pour avoir un pricer
def pricer_mc_call(S0, K, sigma, r, T, N=100_000, rng=None):
    ST = gbm(S0, sigma, r, T, N=N, rng=rng)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def pricer_mc_put(S0,K,sigma,r,T,N=100_000, rng=None):
    ST = gbm(S0,sigma,r,T, N=N, rng=rng)
    payoff= np.maximum(K-ST,0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96 * disc.std(ddof=1) / np.sqrt(N))

#antithetique functions
def gbm_antithetic(S0, sigma, r, T, N=50_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(N)
    drift = (r - 0.5*sigma**2)*T
    up   = S0*np.exp(drift + sigma*np.sqrt(T)*Z)
    down = S0*np.exp(drift - sigma*np.sqrt(T)*Z)
    return up, down

def pricer_mc_call_av(S0, K, sigma, r, T, N=50_000, rng=None):
    up,down = gbm_antithetic(S0, sigma, r, T, N=N, rng=rng)
    pair = 0.5*(np.maximum(up-K, 0.0) + np.maximum(down-K, 0.0))  # moyenne PAR PAIRE
    disc = np.exp(-r*T)*pair
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def delta_mc(S0,h,K,sigma,r,T,N=100_000,seed = 42):
    # Common random numbers : DEUX generateurs distincts, MEME graine -> les
    # deux pricings voient exactement les memes Z. Sans ca, la difference
    # up-down est dominee par le bruit MC (variance en 1/h^2) et le delta est
    # inutilisable. C'est le seul endroit ou une graine fixe est legitime :
    # elle fait partie de l'estimateur, ce n'est pas de la reproductibilite.
    up   = pricer_mc_call(S0+h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    down = pricer_mc_call(S0-h, K, sigma, r, T, N, rng=np.random.default_rng(seed))[0]
    return (up-down)/(2*h)

def gbm_paths(S0, sigma, r, T, n_steps, N=100_000, rng=None):
    
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


if __name__ == "__main__":
    paths = gbm_paths(100, 0.2, 0.05, 1.0, n_steps=50,N=100_000)
    ST=paths[:,-1]
    payoff = np.maximum(ST-100,0)
    prix = np.exp(-0.05*1)*payoff.mean()
    ic   = 1.96 * (np.exp(-0.05)*payoff).std(ddof=1)/np.sqrt(len(ST))
    print(prix, "+/-", ic)
