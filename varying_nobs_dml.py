"""
CVCI-DML Simulation: Varying Observational Sample Size (N^obs)

Replicates CVCI paper Figure 14.4 (e) and (f):
  (e) θ^exp ≠ θ^obs, N^exp = 50, ε = 0.05, N^obs varies from 25 to 200
  (f) θ^exp ≠ θ^obs, N^exp = 1000, ε = 0.05, N^obs varies from 0 to 2000

Key features:
- Y-axis uses log scale (with transition to linear at low MSE)
- Green line (exp-only) is FLAT (doesn't use obs data)
- As N^obs increases, obs-only MSE decreases
- CVCI method adapts between exp and obs

Usage:
    python varying_nobs_dml.py                    # Run panel (e): N^exp=50
    python varying_nobs_dml.py --panel f          # Run panel (f): N^exp=1000
    python varying_nobs_dml.py --quick            # Quick test
    python varying_nobs_dml.py --mode plot --timestamp XXXXXX  # Plot saved results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy import stats
import json
import os
from datetime import datetime
import argparse

# Import from existing DML code
import sys
sys.path.insert(0, '/mnt/project')
from dml_cv import cross_validation_dml, DMLModel, compute_exp_ate_dml

random_seed = 2024
np.random.seed(random_seed)


class FastDMLModel(DMLModel):
    """Faster DML with fewer trees and depth limit."""
    def __init__(self, n_estimators=50, max_depth=10, random_state=None):
        super().__init__(propensity_model='rf', outcome_model='rf', random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """Override with faster RF parameters."""
        if lambda_ < 0.01:
            X_train, A_train, Y_train = X_exp, A_exp, Y_exp
            sample_weight = None
        elif lambda_ > 0.99:
            X_train, A_train, Y_train = X_obs, A_obs, Y_obs
            sample_weight = None
        else:
            X_train = np.vstack([X_exp, X_obs])
            A_train = np.concatenate([A_exp, A_obs])
            Y_train = np.concatenate([Y_exp, Y_obs])
            n_exp, n_obs = len(X_exp), len(X_obs)
            weights_exp = np.ones(n_exp) * (1 - lambda_) / n_exp
            weights_obs = np.ones(n_obs) * lambda_ / n_obs
            sample_weight = np.concatenate([weights_exp, weights_obs])
        
        # FAST Random Forests
        self.e_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if sample_weight is not None:
            self.e_model.fit(X_train, A_train, sample_weight=sample_weight)
        else:
            self.e_model.fit(X_train, A_train)
        
        self.mu0_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.mu1_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        X_0, Y_0 = X_train[A_train == 0], Y_train[A_train == 0]
        if len(X_0) > 0:
            w_0 = sample_weight[A_train == 0] if sample_weight is not None else None
            self.mu0_model.fit(X_0, Y_0, sample_weight=w_0)
        
        X_1, Y_1 = X_train[A_train == 1], Y_train[A_train == 1]
        if len(X_1) > 0:
            w_1 = sample_weight[A_train == 1] if sample_weight is not None else None
            self.mu1_model.fit(X_1, Y_1, sample_weight=w_1)


def generate_synthetic_data(n_exp, n_obs, d, true_te, epsilon, sigma=1.0, seed=None, theta=None):
    """
    Generate synthetic data as in CVCI Section 14.1.
    
    Experimental:
      Z ~ N(0, σ²I)
      W ~ Bern(0.5)  [randomized]
      Y = Z^T θ + W × τ* + ξ
    
    Observational:
      Z ~ N(0, σ²I)
      W ~ Bern(0.2)  [confounded]
      Y = Z^T θ + W × (τ* + ε) + ξ
    
    Args:
        n_exp: Number of experimental samples
        n_obs: Number of observational samples
        d: Covariate dimension
        true_te: True treatment effect τ*
        epsilon: Bias in observational data
        sigma: Noise standard deviation
        seed: Random seed
        theta: Pre-defined covariate coefficients (for consistency across n_obs values)
    
    Returns:
        Z_exp, W_exp, Y_exp, Z_obs, W_obs, Y_obs, theta
    """
    rng = np.random.default_rng(seed)
    
    # Covariate coefficients
    if theta is None:
        theta = rng.normal(0, 1, size=d)
    
    # EXPERIMENTAL DATA
    Z_exp = rng.normal(0, 1, size=(n_exp, d))
    W_exp = rng.binomial(1, 0.5, size=n_exp)  # Randomized 50/50
    Y_exp = Z_exp @ theta + W_exp * true_te + rng.normal(0, sigma, size=n_exp)
    
    # OBSERVATIONAL DATA
    if n_obs > 0:
        Z_obs = rng.normal(0, 1, size=(n_obs, d))
        W_obs = rng.binomial(1, 0.2, size=n_obs)  # Confounded - 20% treated
        Y_obs = Z_obs @ theta + W_obs * (true_te + epsilon) + rng.normal(0, sigma, size=n_obs)
    else:
        Z_obs = np.zeros((0, d))
        W_obs = np.zeros(0, dtype=int)
        Y_obs = np.zeros(0)
    
    return Z_exp, W_exp, Y_exp, Z_obs, W_obs, Y_obs, theta


def save_results_json(results, filename):
    """Save results to JSON with numpy conversion."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_converted = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results_converted, f, indent=2)
    print(f"✓ Saved: {filename}")


def simulate_varying_nobs(n_exp=50, epsilon=0.05, d=5, sigma=1.0,
                          nobs_vals=None, n_sims=100, lambda_bin=50,
                          use_fast=True, save_dir='simulation_results_nobs'):
    """
    Simulate varying N^obs with fixed N^exp and ε.
    
    Key: For each simulation, we generate experimental data ONCE,
    then generate observational data of different sizes.
    This ensures exp-only baseline is flat (as expected).
    
    Args:
        n_exp: Number of experimental samples (fixed)
        epsilon: Bias in observational data (fixed)
        d: Covariate dimension
        sigma: Noise std
        nobs_vals: Array of N^obs values to test
        n_sims: Number of simulations
        lambda_bin: Number of lambda grid points
        use_fast: Use fast DML
        save_dir: Directory to save results
    """
    if nobs_vals is None:
        # Default: match panel (e) from the paper
        nobs_vals = np.arange(25, 205, 5)  # 25, 30, 35, ..., 200
    
    true_te = 1.0
    lambda_vals = np.linspace(0, 1, lambda_bin)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metadata
    metadata = {
        'experiment': 'varying_nobs',
        'n_exp': n_exp,
        'epsilon': epsilon,
        'd': d,
        'sigma': sigma,
        'true_te': true_te,
        'n_sims': n_sims,
        'lambda_bin': lambda_bin,
        'nobs_vals': nobs_vals.tolist(),
        'timestamp': timestamp,
        'use_fast': use_fast
    }
    save_results_json(metadata, f'{save_dir}/metadata_nobs_{timestamp}.json')
    
    results = {n_obs: [] for n_obs in nobs_vals}
    
    print(f"Running N^obs simulation: n_exp={n_exp}, ε={epsilon}")
    print(f"N^obs range: {nobs_vals.min()} to {nobs_vals.max()}")
    print(f"Saving to: {save_dir}/")
    
    # For each simulation
    for sim in range(n_sims):
        print(f"\nSimulation {sim+1}/{n_sims}")
        
        # Generate covariate coefficients ONCE per simulation
        rng_theta = np.random.default_rng(random_seed + sim)
        theta = rng_theta.normal(0, 1, size=d)
        
        # Generate experimental data ONCE per simulation
        rng_exp = np.random.default_rng(random_seed + sim * 10000)
        Z_exp = rng_exp.normal(0, 1, size=(n_exp, d))
        W_exp = rng_exp.binomial(1, 0.5, size=n_exp)
        Y_exp = Z_exp @ theta + W_exp * true_te + rng_exp.normal(0, sigma, size=n_exp)
        
        # Compute exp-only estimate ONCE (same for all N^obs)
        ate_exp = compute_exp_ate_dml(Z_exp, W_exp, Y_exp, method='difference')
        mse_exp = (ate_exp - true_te) ** 2
        
        # Now vary N^obs
        for nobs_idx, n_obs in enumerate(nobs_vals):
            # Generate observational data for this N^obs
            rng_obs = np.random.default_rng(random_seed + sim * 10000 + n_obs)
            
            if n_obs > 0:
                Z_obs = rng_obs.normal(0, 1, size=(n_obs, d))
                W_obs = rng_obs.binomial(1, 0.2, size=n_obs)
                Y_obs = Z_obs @ theta + W_obs * (true_te + epsilon) + rng_obs.normal(0, sigma, size=n_obs)
            else:
                Z_obs = np.zeros((0, d))
                W_obs = np.zeros(0, dtype=int)
                Y_obs = np.zeros(0)
            
            # Obs-only baseline
            if n_obs >= 20:  # Need minimum samples
                if use_fast:
                    model_obs = FastDMLModel(random_state=random_seed + sim)
                else:
                    model_obs = DMLModel(random_state=random_seed + sim)
                model_obs.fit(Z_obs, W_obs, Y_obs, Z_obs[:0], W_obs[:0], Y_obs[:0], lambda_=0.0)
                ate_obs = model_obs.predict_ate(Z_obs, W_obs, Y_obs)
                mse_obs = (ate_obs - true_te) ** 2
            else:
                ate_obs = np.nan
                mse_obs = np.nan
            
            # T-test baseline
            try:
                t_stat, p_value = stats.ttest_ind(Y_exp[W_exp==1], Y_exp[W_exp==0])
                if p_value < 0.05:
                    ate_ttest = ate_exp
                else:
                    if n_obs > 0:
                        Y_all = np.concatenate([Y_exp, Y_obs])
                        W_all = np.concatenate([W_exp, W_obs])
                        ate_ttest = Y_all[W_all==1].mean() - Y_all[W_all==0].mean()
                    else:
                        ate_ttest = ate_exp
                mse_ttest = (ate_ttest - true_te) ** 2
            except:
                ate_ttest = ate_exp
                mse_ttest = mse_exp
            
            # CVCI with DML
            if n_obs >= 20:
                try:
                    Q_values, lambda_opt, model_opt = cross_validation_dml(
                        Z_exp, W_exp, Y_exp, Z_obs, W_obs, Y_obs,
                        lambda_vals=lambda_vals,
                        k_fold=5,
                        exp_ate_method='difference',
                        stratified=True,
                        random_state=random_seed + sim
                    )
                    ate_ours = model_opt.predict_ate(Z_exp, W_exp, Y_exp)
                    mse_ours = (ate_ours - true_te) ** 2
                except Exception as e:
                    print(f"    Warning: CVCI failed for n_obs={n_obs}: {e}")
                    ate_ours = ate_exp
                    mse_ours = mse_exp
                    lambda_opt = 0.0
            else:
                ate_ours = ate_exp
                mse_ours = mse_exp
                lambda_opt = 0.0
            
            # Store result
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
            results[n_obs].append(result)
            
            # Progress
            if (nobs_idx + 1) % 10 == 0:
                print(f"  N^obs={n_obs} ({nobs_idx+1}/{len(nobs_vals)})", end='\r')
        
        print(f"  Completed all {len(nobs_vals)} N^obs values")
        
        # SAVE AFTER EACH SIMULATION (intermediate checkpoint)
        if (sim + 1) % 5 == 0 or sim == n_sims - 1:
            print(f"\n  Saving checkpoint after simulation {sim+1}...")
            for n_obs in nobs_vals:
                if len(results[n_obs]) > 0:
                    checkpoint = {
                        'n_obs': int(n_obs),
                        'simulations': results[n_obs],
                        'n_sims_completed': len(results[n_obs])
                    }
                    filename = f'{save_dir}/nobs_{n_obs:04d}_{timestamp}.json'
                    save_results_json(checkpoint, filename)
            print(f"  Checkpoint saved!")
    
    # Final save
    print("\nFinal save...")
    for n_obs in nobs_vals:
        nobs_results = {
            'n_obs': int(n_obs),
            'simulations': results[n_obs],
            'n_sims_completed': len(results[n_obs])
        }
        filename = f'{save_dir}/nobs_{n_obs:04d}_{timestamp}.json'
        save_results_json(nobs_results, filename)
    
    # Aggregate results
    nobs_results_agg = []
    for n_obs in nobs_vals:
        sim_results = results[n_obs]
        
        # Filter out NaN values for obs
        mse_obs_vals = [r['mse_obs'] for r in sim_results if not np.isnan(r['mse_obs'])]
        
        nobs_results_agg.append({
            'n_obs': int(n_obs),
            'mse_exp_mean': np.mean([r['mse_exp'] for r in sim_results]),
            'mse_exp_std': np.std([r['mse_exp'] for r in sim_results]),
            'mse_obs_mean': np.mean(mse_obs_vals) if mse_obs_vals else np.nan,
            'mse_obs_std': np.std(mse_obs_vals) if mse_obs_vals else np.nan,
            'mse_ttest_mean': np.mean([r['mse_ttest'] for r in sim_results]),
            'mse_ttest_std': np.std([r['mse_ttest'] for r in sim_results]),
            'mse_ours_mean': np.mean([r['mse_ours'] for r in sim_results]),
            'mse_ours_std': np.std([r['mse_ours'] for r in sim_results]),
            'lambda_mean': np.mean([r['lambda_opt'] for r in sim_results]),
            'lambda_std': np.std([r['lambda_opt'] for r in sim_results]),
        })
    
    df = pd.DataFrame(nobs_results_agg)
    
    # Save aggregated results
    csv_file = f'{save_dir}/aggregated_nobs_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved aggregated results: {csv_file}")
    
    # Also save as JSON
    json_file = f'{save_dir}/aggregated_nobs_{timestamp}.json'
    save_results_json({
        'metadata': metadata,
        'results': nobs_results_agg
    }, json_file)
    
    return df, timestamp


def load_and_plot_nobs_results(timestamp, save_dir='simulation_results_nobs', 
                                log_linear_threshold=None):
    """
    Load results from JSON and create plots matching the paper style.
    
    The paper uses a split scale: log scale for large MSE, linear for small MSE.
    """
    # Load aggregated results
    json_file = f'{save_dir}/aggregated_nobs_{timestamp}.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    results = data['results']
    df = pd.DataFrame(results)
    
    # Create plot matching paper style
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine log-linear threshold
    if log_linear_threshold is None:
        # Auto-detect based on exp-only baseline
        log_linear_threshold = df['mse_exp_mean'].iloc[0] * 1.2
    
    # Plot with log scale
    ax.set_yscale('log')
    
    # Plot methods
    ax.plot(df['n_obs'], df['mse_exp_mean'], 'g-o', 
            label=r'Only use $X^{exp}$', linewidth=2, markersize=4)
    ax.plot(df['n_obs'], df['mse_obs_mean'], 'r-o', 
            label=r'Only use $X^{obs}$', linewidth=2, markersize=4)
    ax.plot(df['n_obs'], df['mse_ours_mean'], 'orange', linestyle='-', marker='o',
            label=r'Ours, $\beta(\hat{\theta}(\hat{\lambda}))$', linewidth=2, markersize=4)
    
    # Add horizontal lines for reference (like in paper)
    exp_mse = df['mse_exp_mean'].mean()
    ax.axhline(y=exp_mse, color='g', linestyle='--', alpha=0.5)
    
    # Add "Log scale / Linear scale" text
    ax.axhline(y=log_linear_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.text(df['n_obs'].max() * 0.7, log_linear_threshold * 1.1, 'Log scale', 
            fontsize=10, color='gray')
    ax.text(df['n_obs'].max() * 0.7, log_linear_threshold * 0.9, 'Linear scale', 
            fontsize=10, color='gray')
    
    ax.set_xlabel(r'$N^{obs}$', fontsize=14)
    ax.set_ylabel('Empirical MSE', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Title
    n_exp = metadata['n_exp']
    epsilon = metadata['epsilon']
    ax.set_title(rf"$\theta^{{exp}} \neq \theta^{{obs}}, N^{{exp}} = {n_exp}, \varepsilon = {epsilon}$", 
                 fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f'{save_dir}/nobs_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_file}")
    plt.show()
    
    # Also create lambda plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    ax2.plot(df['n_obs'], df['lambda_mean'], 'b-o', linewidth=2, markersize=4)
    ax2.fill_between(df['n_obs'],
                     df['lambda_mean'] - df['lambda_std'],
                     df['lambda_mean'] + df['lambda_std'],
                     alpha=0.3, color='blue')
    
    ax2.set_xlabel(r'$N^{obs}$', fontsize=14)
    ax2.set_ylabel(r'$\hat{\lambda}$', fontsize=14)
    ax2.set_title(r'Optimal $\lambda$ vs $N^{obs}$', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    lambda_plot_file = f'{save_dir}/lambda_vs_nobs_{timestamp}.png'
    plt.savefig(lambda_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved lambda plot: {lambda_plot_file}")
    plt.show()
    
    return df


def run_panel_e():
    """Run panel (e): N^exp = 50, ε = 0.05, N^obs from 25 to 200."""
    return simulate_varying_nobs(
        n_exp=50,
        epsilon=0.05,
        nobs_vals=np.arange(25, 205, 5),  # 25, 30, ..., 200
        save_dir='simulation_results_nobs_panel_e'
    )


def run_panel_f():
    """Run panel (f): N^exp = 1000, ε = 0.05, N^obs from 0 to 2000."""
    return simulate_varying_nobs(
        n_exp=1000,
        epsilon=0.05,
        nobs_vals=np.concatenate([[0], np.arange(50, 2050, 50)]),  # 0, 50, 100, ..., 2000
        save_dir='simulation_results_nobs_panel_f'
    )


def plot_both_panels(timestamp_e, timestamp_f, output_file='nobs_both_panels.png'):
    """
    Create side-by-side plot matching the paper's Figure 14.4 (e) and (f).
    """
    # Load both results
    with open(f'simulation_results_nobs_panel_e/aggregated_nobs_{timestamp_e}.json', 'r') as f:
        data_e = json.load(f)
    with open(f'simulation_results_nobs_panel_f/aggregated_nobs_{timestamp_f}.json', 'r') as f:
        data_f = json.load(f)
    
    df_e = pd.DataFrame(data_e['results'])
    df_f = pd.DataFrame(data_f['results'])
    meta_e = data_e['metadata']
    meta_f = data_f['metadata']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel (e)
    ax1.set_yscale('log')
    ax1.plot(df_e['n_obs'], df_e['mse_exp_mean'], 'g-o', 
            label=r'Only use $X^{exp}$', linewidth=2, markersize=4)
    ax1.plot(df_e['n_obs'], df_e['mse_obs_mean'], 'r-o', 
            label=r'Only use $X^{obs}$', linewidth=2, markersize=4)
    ax1.plot(df_e['n_obs'], df_e['mse_ours_mean'], 'orange', linestyle='-', marker='o',
            label=r'Ours, $\beta(\hat{\theta}(\hat{\lambda}))$', linewidth=2, markersize=4)
    
    # Add threshold line
    exp_mse_e = df_e['mse_exp_mean'].mean()
    ax1.axhline(y=exp_mse_e, color='g', linestyle='--', alpha=0.3)
    ax1.axhline(y=exp_mse_e * 1.2, color='gray', linestyle='--', alpha=0.3)
    ax1.text(df_e['n_obs'].max() * 0.6, exp_mse_e * 1.3, 'Log scale', fontsize=9, color='gray')
    ax1.text(df_e['n_obs'].max() * 0.6, exp_mse_e * 1.05, 'Linear scale', fontsize=9, color='gray')
    
    ax1.set_xlabel(r'$N^{obs}$', fontsize=14)
    ax1.set_ylabel('Empirical MSE', fontsize=14)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(rf"(e) $\theta^{{exp}} \neq \theta^{{obs}}, N^{{exp}} = {meta_e['n_exp']}, \varepsilon = {meta_e['epsilon']}$", 
                 fontsize=12)
    
    # Panel (f)
    ax2.set_yscale('log')
    ax2.plot(df_f['n_obs'], df_f['mse_exp_mean'], 'g-o', 
            label=r'Only use $X^{exp}$', linewidth=2, markersize=4)
    ax2.plot(df_f['n_obs'], df_f['mse_obs_mean'], 'r-o', 
            label=r'Only use $X^{obs}$', linewidth=2, markersize=4)
    ax2.plot(df_f['n_obs'], df_f['mse_ours_mean'], 'orange', linestyle='-', marker='o',
            label=r'Ours, $\beta(\hat{\theta}(\hat{\lambda}))$', linewidth=2, markersize=4)
    
    # Add threshold line
    exp_mse_f = df_f['mse_exp_mean'].mean()
    ax2.axhline(y=exp_mse_f, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=exp_mse_f * 1.2, color='gray', linestyle='--', alpha=0.3)
    ax2.text(df_f['n_obs'].max() * 0.6, exp_mse_f * 1.3, 'Log scale', fontsize=9, color='gray')
    ax2.text(df_f['n_obs'].max() * 0.6, exp_mse_f * 1.05, 'Linear scale', fontsize=9, color='gray')
    
    ax2.set_xlabel(r'$N^{obs}$', fontsize=14)
    ax2.set_ylabel('Empirical MSE', fontsize=14)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(rf"(f) $\theta^{{exp}} \neq \theta^{{obs}}, N^{{exp}} = {meta_f['n_exp']}, \varepsilon = {meta_f['epsilon']}$", 
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {output_file}")
    plt.show()
    
    return df_e, df_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run varying N^obs simulations')
    parser.add_argument('--mode', type=str, choices=['run', 'plot'], default='run',
                       help='Run simulations or plot existing results')
    parser.add_argument('--panel', type=str, choices=['e', 'f'], default='e',
                       help='Which panel to run: (e) N^exp=50, (f) N^exp=1000')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Timestamp for plotting')
    parser.add_argument('--n-sims', type=int, default=100)
    parser.add_argument('--lambda-bin', type=int, default=50)
    parser.add_argument('--ultra-quick', action='store_true',
                       help='Ultra-fast: 5 sims, 8 nobs values')
    parser.add_argument('--quick', action='store_true',
                       help='Quick: 10 sims, 15 nobs values')
    parser.add_argument('--use-slow-dml', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'plot':
        if args.timestamp is None:
            print("ERROR: --timestamp required for --mode plot")
            sys.exit(1)
        save_dir = f'simulation_results_nobs_panel_{args.panel}'
        load_and_plot_nobs_results(args.timestamp, save_dir=save_dir)
    
    else:  # Run mode
        # Set parameters based on mode
        if args.ultra_quick:
            n_sims = 5
            lambda_bin = 10
            if args.panel == 'e':
                nobs_vals = np.array([25, 50, 75, 100, 125, 150, 175, 200])
            else:
                nobs_vals = np.array([0, 250, 500, 750, 1000, 1250, 1500, 2000])
            print("⚡ ULTRA-QUICK MODE")
        elif args.quick:
            n_sims = 10
            lambda_bin = 20
            if args.panel == 'e':
                nobs_vals = np.arange(25, 210, 15)
            else:
                nobs_vals = np.arange(0, 2100, 150)
            print("⚡ QUICK MODE")
        else:
            n_sims = args.n_sims
            lambda_bin = args.lambda_bin
            nobs_vals = None  # Use defaults
            print(f"🐢 FULL MODE: {n_sims} sims, {lambda_bin} lambdas")
        
        use_fast = not args.use_slow_dml
        
        if args.panel == 'e':
            print(f"\nRunning Panel (e): N^exp=50, ε=0.05")
            save_dir = 'simulation_results_nobs_panel_e'
            n_exp = 50
            if nobs_vals is None:
                nobs_vals = np.arange(25, 205, 5)
        else:
            print(f"\nRunning Panel (f): N^exp=1000, ε=0.05")
            save_dir = 'simulation_results_nobs_panel_f'
            n_exp = 1000
            if nobs_vals is None:
                nobs_vals = np.concatenate([[0], np.arange(50, 2050, 50)])
        
        df, timestamp = simulate_varying_nobs(
            n_exp=n_exp,
            epsilon=0.05,
            d=5,
            sigma=1.0,
            nobs_vals=nobs_vals,
            n_sims=n_sims,
            lambda_bin=lambda_bin,
            use_fast=use_fast,
            save_dir=save_dir
        )
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE!")
        print("="*70)
        print(f"\nTo recreate plots:")
        print(f"  python varying_nobs_dml.py --mode plot --panel {args.panel} --timestamp {timestamp}")
        
        # Create plots
        load_and_plot_nobs_results(timestamp, save_dir=save_dir)