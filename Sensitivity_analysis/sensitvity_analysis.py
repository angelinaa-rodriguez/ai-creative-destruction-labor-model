import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

df = pd.read_csv("final_merged_data.csv")
df["date"] = pd.to_datetime(df["date"])

cutoff = pd.Timestamp("2025-04-01")
df = df[df["date"] <= cutoff]

# drop missing
df = df.dropna(subset=["employment", "wage_real", "value_added"])

# Time index
df["t"] = np.arange(len(df))

t_data = df["t"].values
E_data = df["employment"].values
W_data = df["wage_real"].values
V_data = (df["value_added"].values) / 1e6  # scale to millions


ParEst = np.array([
    2.0, # alpha
    0.35, # phi
    0.0, # beta
    0.004, # k
    0.009, # lambda
    0.5, # delta
    0.0002, # gamma
    0.03  # rho
])


alpha0, phi0, beta0, k0, lam0, delta0, gamma0, rho0 = ParEst


L = 4400
A0 = 2212.1
g  = 0.008129
pi_val = 0.017

# A(t)
def A(t):
    return A0 * np.exp(g * t)

# GET V(t)
def V_lookup(t):
    """
    MATLAB uses:
        idx = min(max(floor(t)+1,1), length(Vdata))
    """
    idx = int(np.floor(t))
    idx = max(0, min(idx, len(V_data)-1))
    return V_data[idx]

# ODE SYSTEM
def econ_ode(t, y, params):
    E = max(y[0], 1)
    W = y[1]

    alpha, phi, beta, k, lam, delta, gamma, rho = params

    At = A(t)
    Vt = V_lookup(t)

    dE = (
        -alpha * (E / At)
        - pi_val * E
        + phi * (L - E)
        + beta * W
    )

    dW = (
        (k * At) / (1 + lam * At)
        + delta * (Vt / E)
        - gamma * E
        - rho * W
    )

    return [dE, dW]

# RUN MODEL
def run_model(params):
    y0 = [E_data[0], W_data[0]]

    sol = solve_ivp(
        lambda t, y: econ_ode(t, y, params),
        t_span=(t_data[0], t_data[-1]),
        y0=y0,
        t_eval=t_data,
        max_step=0.1
    )

    return sol.t, sol.y[0], sol.y[1]

#### SENSITIVITY ANALYSIS

param_names = ["alpha", "phi", "k", "lambda", "gamma", "rho"]
indices     = [0, 1, 3, 4, 6, 7]

multipliers = [0.5, 1.0, 1.5]


# employment sensitivity
for pname, idx in zip(param_names, indices):

    plt.figure(figsize=(8,5))
    base_params = ParEst.copy()

    for m in multipliers:
        test_params = base_params.copy()
        test_params[idx] = base_params[idx] * m

        t, Etest, Wtest = run_model(test_params)
        plt.plot(df["date"], Etest, label=f"{pname} × {m}", linewidth=2)

    plt.title(f"Employment Sensitivity to {pname}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Employment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"employment_sensitivity_{pname}.png", dpi=300)
    plt.show()


# wage sensitivity
for pname, idx in zip(param_names, indices):

    plt.figure(figsize=(8,5))
    base_params = ParEst.copy()

    for m in multipliers:
        test_params = base_params.copy()
        test_params[idx] = base_params[idx] * m

        t, Etest, Wtest = run_model(test_params)
        plt.plot(df["date"], Wtest, label=f"{pname} × {m}", linewidth=2)

    plt.title(f"Wage Sensitivity to {pname}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Wage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"wage_sensitivity_{pname}.png", dpi=300)
    plt.show()
