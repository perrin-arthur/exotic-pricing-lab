import bs
import numpy as np

from scipy.stats import norm

#gbm
def gbm(S0,sigma,r,T, N=100_000):
    return S0*np.exp((r-0.5*sigma**2)*(T)+sigma*np.sqrt(T)*np.random.standard_normal(N))

#monte-carlo mtn pour avoir un pricer 
def pricer_mc_call(S0, K, sigma, r, T, N=100_000):
    ST = gbm(S0, sigma, r, T, N=N)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))

def pricer_mc_put(S0,K,sigma,r,T,N=100_000):
    ST = gbm(S0,sigma,r,T, N=N)
    payoff= np.maximum(K-ST,0)
    disc = np.exp(-r*T) * payoff
    return float(disc.mean()), float(1.96 * disc.std(ddof=1) / np.sqrt(N))

#antithetique functions
def gbm_antithetic(S0, sigma, r, T, N=50_000):
    Z = np.random.standard_normal(N)
    drift = (r - 0.5*sigma**2)*T
    up   = S0*np.exp(drift + sigma*np.sqrt(T)*Z)
    down = S0*np.exp(drift - sigma*np.sqrt(T)*Z)
    return up, down

def pricer_mc_call_av(S0, K, sigma, r, T, N=50_000):
    up,down = gbm_antithetic(S0, sigma, r, T, N=N)
    pair = 0.5*(np.maximum(up-K, 0.0) + np.maximum(down-K, 0.0))  # moyenne PAR PAIRE
    disc = np.exp(-r*T)*pair
    return float(disc.mean()), float(1.96*disc.std(ddof=1)/np.sqrt(N))


def delta_mc(S0,h,K,sigma,r,T,N=100_000,seed = 42):
    np.random.seed(seed)
    up = pricer_mc_call(S0+h, K, sigma, r, T, N)[0]
    np.random.seed(seed)
    down = pricer_mc_call(S0-h, K, sigma, r, T, N)[0]
    return (up-down)/(2*h)
