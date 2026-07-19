import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

#gbm
def gbm(S0,sigma,r,T, N=100_000):
    return S0*np.exp((r-0.5*sigma**2)*(T)+sigma*np.sqrt(T)*np.random.standard_normal(N))

#BS fermé(call & put) assume qu'on commence à t=0

def call_bs(S0,K,sigma,r,T,q=0.0):
    d1= (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) -np.exp(-r*(T))*K*norm.cdf(d2)

#on utilise la parité put-call pour avoir le put

def put_bs(S0,K,sigma,r,T,q=0.0):
    return call_bs(S0,K,sigma,r,T,q=0.0)+K*np.exp(-r*T) - S0*np.exp(-q*T)

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

print(f"option call value")
print(call_bs(100, 100, 0.2, 0.05, 1,0.0))
print(f"monte carlo pricer call value")
print(pricer_mc_call(100, 100, 0.2, 0.05, 1))

#étude de la converge en 1/np.sqrt(N)

Ns = [100,1000,10000,100_000,1_000_000,10000000]
bs_ref = call_bs(100, 100, 0.2, 0.05, 1)

errors=[]

for N in Ns:
    price,_ = pricer_mc_call(100, 100, 0.2, 0.05, 1, N =N)
    errors.append(abs(price - bs_ref))
print(errors)

plt.loglog(Ns, errors, 'o-', label='|MC - BS|')
plt.loglog(Ns, [errors[0]*np.sqrt(Ns[0]/N) for N in Ns], '--', label='pente -1/2')
plt.xlabel('N paths'); plt.ylabel('erreur'); plt.legend()
plt.show()

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

print(pricer_mc_call(100, 100, 0.2, 0.05, 1, N=100_000))
print(pricer_mc_call_av(100, 100, 0.2, 0.05, 1, N=50_000))

#sanity check sur la parité call-put

np.random.seed(42)
p1 = pricer_mc_call(100, 100, 0.2, 0.05, 1)
np.random.seed(42)
p2 = pricer_mc_put(100, 100, 0.2, 0.05, 1)
print("call_mc -put_mc : ")
print(p1[0]-p2[0])
print("vs")
print(100 - 100*np.exp(-0.05))
lhs = p1[0] - p2[0]
rhs = 100 - 100*np.exp(-0.05)
assert abs(lhs - rhs) < 0.1, f"parité violée : {lhs:.4f} vs {rhs:.4f}"

#calcul des greeks

def delta(S0,h,K,sigma,r,T):
    return (call_bs(S0+h,K,sigma,r,T,q=0.0) - call_bs(S0-h,K,sigma,r,T,q=0.0))/(2*h)
print("delta théorique :", delta(100, 0.1, 100, 0.2, 0.05, 1))

def delta_mc(S0,h,K,sigma,r,T,N=100_000,seed = 42):
    np.random.seed(seed)
    up = pricer_mc_call(S0+h, K, sigma, r, T, N)[0]
    np.random.seed(seed)
    down = pricer_mc_call(S0-h, K, sigma, r, T, N)[0]
    return (up-down)/(2*h)

print("delta monte carlo:",delta_mc(100, 0.1, 100, 0.2, 0.05, 1))
