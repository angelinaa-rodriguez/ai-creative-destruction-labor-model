import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# 1. LOAD YOUR DATA (same as in your sensitivity file)
# ============================================================

df = pd.read_csv("final_merged_data.csv")
df["date"] = pd.to_datetime(df["date"])

cutoff = pd.Timestamp("2025-04-01")
df = df[df["date"] <= cutoff]

df = df.dropna(subset=["employment", "wage_real", "value_added"])

df["t"] = np.arange(len(df))

t_data = df["t"].values
E_data = df["employment"].values
W_data = df["wage_real"].values
V_data = df["value_added"].values / 1e6  # scale to millions


# ============================================================
# 2. DEFINE CONSTANTS (same as MATLAB file)
# ============================================================

A0 = 2212.1
g = 0.008129
pi_val = 0.017
L = 4400


# ============================================================
# 3. ODE MODEL (matches your MATLAB econ_ode)
# ============================================================

def econ_ode(t, y, par, V_data, A0, g, pi_val, L):
    E = max(y[0], 1)
    W = y[1]

    alpha, phi, beta, k, lam, delta, gamma, rho = par

    # Innovation curve
    A = A0 * np.exp(g * t)

    # Lookup V(t) same way MATLAB does
    idx = int(np.floor(t))
    idx = max(0, min(idx, len(V_data)-1))
    V = V_data[idx]

    dE = (
        -alpha * (E / A)
        - pi_val * E
        + phi * (L - E)
        + beta * W
    )

    dW = (
        (k * A) / (1 + lam * A)
        + delta * (V / E)
        - gamma * E
        - rho * W
    )

    return [dE, dW]


# ============================================================
# 4. solutionfunction FOR MCMC
#    (returns model-predicted Employment for the time points)
# ============================================================

def solutionfunction(par1, time_points, E0, W0, V_data, A0, g, pi_val, L):

    y0 = [E0, W0]

    sol = solve_ivp(
        lambda t, y: econ_ode(t, y, par1, V_data, A0, g, pi_val, L),
        t_span=(time_points[0], time_points[-1]),
        y0=y0,
        t_eval=time_points,
        max_step=0.1
    )

    return time_points, sol.y[0]  # return employment only


# ============================================================
# 5. MCMCparfind (Python version of professor's MATLAB code)
# ============================================================

def MCMCparfind(solutionfunction, par1guess, datalist, *args):

    par1guess = np.array(par1guess, dtype=float)
    n = len(par1guess)

    time_points = datalist[0, :]
    data_vals = datalist[1, :]

    # Initial evaluation
    _, X1 = solutionfunction(par1guess, time_points, *args)
    chi1 = np.linalg.norm((X1 - data_vals)**2)
    # print("this is chi1" + str(chi1))

    # Settings
    sd = np.array([x * 0.005 for x in par1guess])
    total = 1000
    burntime = total // 10
    sigma = 1.0

    par_samples = []
    chis = []
    accept = np.zeros(total)
    num_accept = 0

    # MCMC loop
    for j in range(total):

        # propose new parameters (Gaussian RW), enforce positivity
        par2guess = par1guess + sd * np.random.randn(n)

        # if np.any(par2guess <= 0):
        #     print('skipped')
        #     continue

        _, X2 = solutionfunction(par2guess, time_points, *args)
        chi2 = np.linalg.norm((X2 - data_vals)**2)

        # acceptance probability
        value = (-chi2 + chi1) / (2 * sigma**2)
        ratio = np.exp(min(100, value))   # cap exponent to prevent overflow
        if np.random.rand() < ratio:
            par1guess = par2guess
            # print("this is chi2" + str(chi2))
            chi1 = chi2
            num_accept += 1 
            accept[j] = 1

        if j > burntime:
            par_samples.append(par1guess.copy())
            chis.append(chi1)

    perc_accept = num_accept / total 
    print("PERCENT ACCEPTED:" + str(perc_accept))
    par_samples = np.array(par_samples)
    chis = np.array(chis)
    # print(chis)
    

    # best sample
    idx_best = np.argmin(chis)
    par_best = par_samples[idx_best]
    fval = chis[idx_best]

    return par_best, fval, par_samples


# ============================================================
# 6. PREPARE INPUTS AND RUN MCMC
# ============================================================

# Use your fitted parameters (replace with actual estimates)
ParEst = np.array([
    2.0,      # alpha
    0.35,     # phi
    0.0,      # beta
    0.004,    # k
    0.009,    # lambda
    0.5,      # delta
    0.0002,   # gamma
    0.03      # rho
])

datalist = np.vstack([t_data, E_data])

par_best, fval, par_samples = MCMCparfind(
    solutionfunction,
    ParEst,
    datalist,
    E_data[0], W_data[0], V_data, A0, g, pi_val, L
)


# ============================================================
# 7. POSTERIOR HISTOGRAMS
# ============================================================

param_names = ["alpha", "phi", "beta", "k", "lambda", "delta", "gamma", "rho"]

# print(par_samples[:, 0])

for i in range(len(param_names)):
    plt.figure()
    plt.hist(par_samples[:, i], bins=30, density=True)
    plt.title(f"Posterior distribution of {param_names[i]}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"identifiability_data/posterior_{param_names[i]}.png", dpi=300)
    plt.show()

print("Best estimates from MCMC:", par_best)
print("Final error:", fval)
