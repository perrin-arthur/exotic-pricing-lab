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

def gbm_paths(S0, sigma, r, T, n_steps, N=100_000):
    # --- TODO A : signature ---------------------------------------------
    # Ajouter un parametre rng (np.random.default_rng(seed)) et le propager
    # jusqu'a gbm(). Sans ca :
    #   - pas reproductible (tu dependis du generateur global numpy) ;
    #   - impossible de faire des common random numbers SUR LES PATHS, ce qui
    #     est indispensable pour le delta pres de la barriere (etape 6 du TP).
    # np.random.seed() est l'API legacy ; default_rng est la bonne.
    dt= T/n_steps

    paths=[np.full(N,S0)]

    # --- TODO B : version vectorisee (cumsum) ----------------------------
    # La recurrence ci-dessous est EXACTE (pas d'Euler) et correcte. Mais
    # l'etape demandee est la version d'un bloc :
    #   1. Z = rng.standard_normal((N, n_steps))
    #   2. log-increments : (r - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z
    #   3. np.cumsum(..., axis=1)
    #   4. PREFIXER une colonne de zeros (np.hstack / np.concatenate)
    #   5. S0 * np.exp(...)
    # Piege classique : oublier l'etape 4 -> tes paths demarrent au premier pas
    # au lieu de S0, et le test "call europeen ~ BS" passe quand meme a peu
    # pres. Verifier paths[:, 0] == S0 EXACTEMENT.
    # Bonus : np.array(paths).T renvoie un array F-contigu -> ton futur
    # paths.min(axis=1) dans barriers.py sera lent. La version cumsum evite ca.
    for i in range(n_steps):
        ST= gbm(paths[-1],sigma,r,dt,N)
        paths.append(ST)
    return np.array(paths).T


# --- TODO C : tests de validation a ecrire dans tests/ ----------------------
# Avant de passer aux barrieres, ces trois tests doivent etre verts :
#   1. paths[:, 0] == S0 exactement (cf TODO B)
#   2. prix du call europeen sur paths[:, -1] dans l'IC autour de 10.4506
#   3. MEME prix que pricer_mc_call a N identique : la loi de S_T ne depend
#      PAS de n_steps. C'est ton test de non-regression le plus fort.
if __name__ == "__main__":
    paths = gbm_paths(100, 0.2, 0.05, 1.0, n_steps=50,N=100_000)
    ST=paths[:,-1]
    payoff = np.maximum(ST-100,0)
    prix = np.exp(-0.05*1)*payoff.mean()
    ic   = 1.96 * (np.exp(-0.05)*payoff).std(ddof=1)/np.sqrt(len(ST))
    print(prix, "+/-", ic)
