import numpy as np

from scipy.stats import norm

#BS fermé(call & put) assume qu'on commence à t=0

def call_bs(S0,K,sigma,r,T,q=0.0):
    d1= (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) -np.exp(-r*(T))*K*norm.cdf(d2)

#on utilise la parité put-call pour avoir le put

def put_bs(S0,K,sigma,r,T,q=0.0):
    return call_bs(S0,K,sigma,r,T,q=0.0)+K*np.exp(-r*T) - S0*np.exp(-q*T)


def delta(S0,h,K,sigma,r,T):
    return (call_bs(S0+h,K,sigma,r,T,q=0.0) - call_bs(S0-h,K,sigma,r,T,q=0.0))/(2*h)