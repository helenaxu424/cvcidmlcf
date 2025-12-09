"""
Controlled Simulations using Causal Forest (CF)

Implements the Section 14.1 controlled experiments from Yang et al. (2025)
with Causal Forest.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime


from cf_cv import CFModel, cross_validation_cf

random_seed = 2024
np.random.seed(random_seed)

def generate_synthetic_data(n_exp, n_obs, d, true_te, epsilon, sigma=1.0, seed=None):
    """
    Synthetic data generator exactly as in CVCI Section 14.1.

    EXPERIMENTAL:
        - Z_exp ~ N(0, I)
        - W_exp ~ Bernoulli(0.5)
        - Y_exp = Z_exp^T theta + W_exp * tau + noise

    OBSERVATIONAL:
        - Z_obs ~ N(0, I)
        - W_obs ~ Bernoulli(0.2)  (confounded)
        - Y_obs = Z_obs^T theta + W_obs * (tau + epsilon) + noise
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, size=d)

    # Experimental
    Z_exp = rng.normal(0, 1, size=(n_exp, d))
    W_exp = rng.binomial(1, 0.5, size=n_exp)
    Y_exp = Z_exp @ theta + W_exp * true_te + rng.normal(0, sigma, size=n_exp)

    # Observational
    Z_obs = rng.normal(0, 1, size=(n_obs, d))
    W_obs = rng.binomial(1, 0.2, size=n_obs)
    Y_obs = Z_obs @ theta + W_obs * (true_te + epsilon) + rng.normal(0, sigma, size=n_obs)

    return Z_exp, W_exp, Y_exp, Z_obs, W_obs, Y_obs


class FastCFModel(CFModel):
    """
    Fast CF: override forest hyperparameters without modifying the
    CFModel.fit() signature.
    """

    def __init__(self, n_estimators=200, max_depth=20, random_state=None):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _apply_fast_params(self):
        """Inject parameters into CFModel internal attributes."""
        # Adjust depending on your CFModel:

        # 1. If CFModel uses self.rf_params:
        if hasattr(self, "rf_params"):
            self.rf_params["n_estimators"] = self.n_estimators
            self.rf_params["max_depth"] = self.max_depth

        # 2. If CFModel directly stores forest kwargs:
        if hasattr(self, "forest_params"):
            self.forest_params["n_estimators"] = self.n_estimators
            self.forest_params["max_depth"] = self.max_depth

    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        # Inject hyperparameters BEFORE training
        self._apply_fast_params()

        # Call parent fit (cannot pass additional kwargs!)
        return super().fit(
            X_exp, A_exp, Y_exp,
            X_obs, A_obs, Y_obs,
            lambda_
        )


def run_single_simulation(n_exp, n_obs, d, epsilon, sigma,
                          lambda_vals, true_te, seed, use_fast=True):
    """
    Run ONE controlled simulation for given epsilon (selection bias).
    CF version of run_single_simulation in DML script.
    """

    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs = generate_synthetic_data(
        n_exp, n_obs, d, true_te, epsilon, sigma, seed
    )

    # Method 1: Exp-only
    ate_exp = Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
    mse_exp = (ate_exp - true_te) ** 2

    # Method 2: Obs-only
    model_obs = FastCFModel(random_state=seed) if use_fast else CFModel(random_state=seed)
    model_obs.fit(
        X_exp[:0], A_exp[:0], Y_exp[:0],   # no exp
        X_obs, A_obs, Y_obs,              # all observational
        lambda_=1.0                       # force obs-only
    )
    ate_obs = model_obs.predict_ate(X_obs, A_obs, Y_obs)
    mse_obs = (ate_obs - true_te) ** 2

    # Method 3: T-test baseline
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(Y_exp[A_exp==1], Y_exp[A_exp==0])
    if p_value < 0.05:
        ate_ttest = ate_exp
    else:
        Y_all = np.concatenate([Y_exp, Y_obs])
        A_all = np.concatenate([A_exp, A_obs])
        ate_ttest = Y_all[A_all==1].mean() - Y_all[A_all==0].mean()
    mse_ttest = (ate_ttest - true_te) ** 2

    # Method 4: CVCI with CF (λ cross-validation)
    Q_values, lambda_opt, model_opt = cross_validation_cf(
        X_exp, A_exp, Y_exp,
        X_obs, A_obs, Y_obs,
        lambda_vals=lambda_vals,
        k_fold=5,
        random_state=seed
    )

    ate_ours = model_opt.predict_ate(X_exp, A_exp, Y_exp)
    mse_ours = (ate_ours - true_te) ** 2

    return {
        'mse_exp': mse_exp,
        'mse_obs': mse_obs,
        'mse_ttest': mse_ttest,
        'mse_ours': mse_ours,
        'lambda_opt': lambda_opt,
        'ate_exp': ate_exp,
        'ate_obs': ate_obs,
        'ate_ttest': ate_ttest,
        'ate_ours': ate_ours
    }


def save_results_json(results, filename):
    """Convert numpy types → Python and save JSON."""

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"✓ Saved: {filename}")


def simulate_varying_epsilon(
    n_exp=50, n_obs=100, d=5, sigma=1.0,
    epsilon_vals=None, n_sims=100, lambda_bin=50,
    use_fast=True, save_dir='simulation_results_cf'
):
    """
    Main simulation loop – identical to DML version except CF inside.
    Saves incremental JSON after each epsilon.
    """

    if epsilon_vals is None:
        epsilon_vals = np.linspace(0, 0.6, 21)

    true_te = 1.0
    lambda_vals = np.linspace(0, 1, lambda_bin)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save metadata
    metadata = {
        'experiment': 'varying_epsilon_cf',
        'n_exp': n_exp,
        'n_obs': n_obs,
        'd': d,
        'sigma': sigma,
        'true_te': true_te,
        'n_sims': n_sims,
        'lambda_bin': lambda_bin,
        'epsilon_vals': epsilon_vals.tolist(),
        'timestamp': timestamp,
        'use_fast': use_fast
    }
    save_results_json(metadata, f'{save_dir}/metadata_epsilon_{timestamp}.json')

    results = {eps: [] for eps in epsilon_vals}

    print(f"Running CF epsilon simulation: n_exp={n_exp}, n_obs={n_obs}")

    for sim in range(n_sims):
        print(f"\nSimulation {sim+1}/{n_sims}")

        # Generate EXP data once per simulation (same as DML version)
        rng_exp = np.random.default_rng(random_seed + sim)
        Z_exp = rng_exp.normal(0, 1, size=(n_exp, d))
        W_exp = rng_exp.binomial(1, 0.5, size=n_exp)
        theta = rng_exp.normal(0, 1, size=d)
        Y_exp = Z_exp @ theta + W_exp * true_te + rng_exp.normal(0, sigma, size=n_exp)

        ate_exp = Y_exp[W_exp==1].mean() - Y_exp[W_exp==0].mean()
        mse_exp = (ate_exp - true_te)**2

        for eps_idx, eps in enumerate(epsilon_vals):

            rng_obs = np.random.default_rng(random_seed + sim*1000 + int(eps*10000))
            Z_obs = rng_obs.normal(0, 1, size=(n_obs, d))
            W_obs = rng_obs.binomial(1, 0.2, size=n_obs)
            Y_obs = Z_obs @ theta + W_obs*(true_te + eps) + rng_obs.normal(0, sigma, size=n_obs)

            # Obs-only CF (λ=1)
            model_obs = FastCFModel(random_state=random_seed + sim) if use_fast else CFModel(random_state=random_seed + sim)
            model_obs.fit(Z_exp[:0], W_exp[:0], Y_exp[:0], Z_obs, W_obs, Y_obs, lambda_=1.0)
            ate_obs = model_obs.predict_ate(Z_obs, W_obs, Y_obs)
            mse_obs = (ate_obs - true_te)**2

            # T-test baseline
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(Y_exp[W_exp==1], Y_exp[W_exp==0])
            if p_value < 0.05:
                ate_ttest = ate_exp
            else:
                Y_all = np.concatenate([Y_exp, Y_obs])
                W_all = np.concatenate([W_exp, W_obs])
                ate_ttest = Y_all[W_all==1].mean() - Y_all[W_all==0].mean()
            mse_ttest = (ate_ttest - true_te)**2

            # CVCI with CF
            Q_values, lambda_opt, model_opt = cross_validation_cf(
                Z_exp, W_exp, Y_exp,
                Z_obs, W_obs, Y_obs,
                lambda_vals=lambda_vals,
                k_fold=5,
                random_state=random_seed + sim
            )
            ate_ours = model_opt.predict_ate(Z_exp, W_exp, Y_exp)
            mse_ours = (ate_ours - true_te)**2

            # store
            result = {
                'mse_exp': mse_exp,
                'mse_obs': mse_obs,
                'mse_ttest': mse_ttest,
                'mse_ours': mse_ours,
                'lambda_opt': lambda_opt,
                'ate_exp': ate_exp,
                'ate_obs': ate_obs,
                'ate_ttest': ate_ttest,
                'ate_ours': ate_ours
            }

            results[eps].append(result)

        print(f"  Completed eps grid.")

    # save per-epsilon JSON
    for eps in epsilon_vals:
        eps_results = {
            'epsilon': eps,
            'simulations': results[eps],
            'n_sims_completed': len(results[eps])
        }
        save_results_json(eps_results, f'{save_dir}/epsilon_{eps:.4f}_{timestamp}.json')

    # aggregated df
    epsilon_results = []
    for eps in epsilon_vals:
        sim_results = results[eps]
        epsilon_results.append({
            'epsilon': eps,
            'mse_exp_mean': np.mean([r['mse_exp'] for r in sim_results]),
            'mse_exp_std': np.std([r['mse_exp'] for r in sim_results]),
            'mse_obs_mean': np.mean([r['mse_obs'] for r in sim_results]),
            'mse_obs_std': np.std([r['mse_obs'] for r in sim_results]),
            'mse_ttest_mean': np.mean([r['mse_ttest'] for r in sim_results]),
            'mse_ttest_std': np.std([r['mse_ttest'] for r in sim_results]),
            'mse_ours_mean': np.mean([r['mse_ours'] for r in sim_results]),
            'mse_ours_std': np.std([r['mse_ours'] for r in sim_results]),
            'lambda_mean': np.mean([r['lambda_opt'] for r in sim_results]),
            'lambda_std': np.std([r['lambda_opt'] for r in sim_results]),
        })

    df = pd.DataFrame(epsilon_results)

    csv_file = f'{save_dir}/aggregated_epsilon_{timestamp}.csv'
    json_file = f'{save_dir}/aggregated_epsilon_{timestamp}.json'

    df.to_csv(csv_file, index=False)
    save_results_json({'metadata': metadata, 'results': epsilon_results}, json_file)

    print(f"\n✓ Saved aggregated results to:")
    print(f"  {csv_file}")
    print(f"  {json_file}")

    return df, timestamp


# plotting
def load_and_plot_epsilon_results(timestamp, save_dir='simulation_results_cf'):
    """Load JSON results and plot (same as DML version)."""

    json_file = f'{save_dir}/aggregated_epsilon_{timestamp}.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    df = pd.DataFrame(data['results'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Panel 1: MSE vs epsilon
    ax = axes[0]
    ax.plot(df['epsilon'], df['mse_exp_mean'], 'g-', label='Exp-only', linewidth=2)
    ax.plot(df['epsilon'], df['mse_obs_mean'], 'r-', label='Obs-only', linewidth=2)
    ax.plot(df['epsilon'], df['mse_ttest_mean'], 'b-', label='T-test', linewidth=2)
    ax.plot(df['epsilon'], df['mse_ours_mean'], 'orange', 
            label='CVCI (CF)', linewidth=2)

    ax.fill_between(df['epsilon'],
                    df['mse_ours_mean'] - df['mse_ours_std'],
                    df['mse_ours_mean'] + df['mse_ours_std'],
                    alpha=0.2, color='orange')

    ax.set_xlabel(r'$\varepsilon$', fontsize=14)
    ax.set_ylabel('Empirical MSE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: lambda vs epsilon
    ax = axes[1]
    ax.plot(df['epsilon'], df['lambda_mean'], 'b-', linewidth=2,
            label=r'Mean $\hat{\lambda}$')

    ax.set_xlabel(r'$\varepsilon$', fontsize=14)
    ax.set_ylabel(r'$\hat{\lambda}$ Density', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f'{save_dir}/epsilon_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=300)
    print(f"\n✓ Saved plot: {plot_file}")
    plt.show()

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run CF controlled simulations')
    parser.add_argument('--mode', type=str, choices=['run', 'plot'], default='run')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--n-sims', type=int, default=100)
    parser.add_argument('--lambda-bin', type=int, default=50)
    parser.add_argument('--ultra-quick', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--medium', action='store_true')
    parser.add_argument('--use-slow', action='store_true')

    args = parser.parse_args()

    if args.mode == 'plot':
        if args.timestamp is None:
            print("ERROR: timestamp required.")
            exit(1)
        load_and_plot_epsilon_results(args.timestamp)
        exit(0)

    # run mode
    if args.ultra_quick:
        n_sims = 5
        lambda_bin = 10
        epsilon_vals = np.linspace(0, 0.6, 5)
        print("ULTRA QUICK MODE")
    elif args.quick:
        n_sims = 10
        lambda_bin = 15
        epsilon_vals = np.linspace(0, 0.6, 6)
        print("QUICK MODE")
    elif args.medium:
        n_sims = 50
        lambda_bin = 20
        epsilon_vals = np.linspace(0, 0.6, 21)
        print("MEDIUM MODE")
    else:
        n_sims = args.n_sims
        lambda_bin = args.lambda_bin
        epsilon_vals = np.linspace(0, 0.6, 21)
        print("FULL MODE")

    use_fast = not args.use_slow

    df, timestamp = simulate_varying_epsilon(
        n_exp=50,
        n_obs=100,
        d=5,
        sigma=1.0,
        epsilon_vals=epsilon_vals,
        n_sims=n_sims,
        lambda_bin=lambda_bin,
        use_fast=use_fast
    )

    print("\nSimulation complete.")
    print(f"To plot later: python controlled_simulations_cf.py --mode plot --timestamp {timestamp}")
    load_and_plot_epsilon_results(timestamp)
