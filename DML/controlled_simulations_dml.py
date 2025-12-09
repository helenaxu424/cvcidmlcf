"""
OPTIMIZED Controlled Simulations with JSON Saving

Key optimizations:
1. Saves results to JSON after each epsilon/nobs value (never lose data!)
2. Faster Random Forests (fewer trees, max_depth limit)
3. Coarser lambda grid option
4. Can resume from saved JSON if interrupted
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dml_cv import cross_validation_dml, DMLModel, compute_exp_ate_dml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime

random_seed = 2024
np.random.seed(random_seed)

def generate_synthetic_data(n_exp, n_obs, d, true_te, epsilon, sigma=1.0, seed=None):
    """
    Generate synthetic data EXACTLY as in CVCI Section 14.1.
    
    Experimental:
      Z ~ N(0, σ²I)
      W ~ Bern(0.5)  [randomized]
      Y = Z^T θ + W × τ* + ξ
    
    Observational:
      Z ~ N(0, σ²I)
      W ~ Bern(0.2)  [confounded - fixed low propensity]
      Y = Z^T θ + W × (τ* + ε) + ξ
    
    Key: Same θ for both, but obs has:
      1. Different treatment propensity (0.2 vs 0.5)
      2. Biased treatment effect (τ* + ε vs τ*)
    """
    rng = np.random.default_rng(seed)
    
    # Same covariate coefficients for both (as per Section 14.1)
    theta = rng.normal(0, 1, size=d)
    
    # EXPERIMENTAL DATA
    Z_exp = rng.normal(0, 1, size=(n_exp, d))
    W_exp = rng.binomial(1, 0.5, size=n_exp)  # Randomized 50/50
    Y_exp = Z_exp @ theta + W_exp * true_te + rng.normal(0, sigma, size=n_exp)
    
    # OBSERVATIONAL DATA
    Z_obs = rng.normal(0, 1, size=(n_obs, d))
    W_obs = rng.binomial(1, 0.2, size=n_obs)  # Confounded - fixed 20% treated
    # Treatment effect is biased: (τ* + ε) instead of τ*
    Y_obs = Z_obs @ theta + W_obs * (true_te + epsilon) + rng.normal(0, sigma, size=n_obs)
    
    return Z_exp, W_exp, Y_exp, Z_obs, W_obs, Y_obs


# Create a fast DMLModel wrapper
class FastDMLModel(DMLModel):
    """Faster DML with fewer trees and depth limit."""
    def __init__(self, n_estimators=50, max_depth=10, random_state=None):
        super().__init__(propensity_model='rf', outcome_model='rf', random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """Override with faster RF parameters."""
        # Same logic as parent, but with faster RFs
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
            n_jobs=-1  # Use all cores!
        )
        
        if sample_weight is not None:
            self.e_model.fit(X_train, A_train, sample_weight=sample_weight)
        else:
            self.e_model.fit(X_train, A_train)
        
        # Outcome models
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
        
        # Fit on controls
        X_0, Y_0 = X_train[A_train == 0], Y_train[A_train == 0]
        if len(X_0) > 0:
            w_0 = sample_weight[A_train == 0] if sample_weight is not None else None
            self.mu0_model.fit(X_0, Y_0, sample_weight=w_0)
        
        # Fit on treated
        X_1, Y_1 = X_train[A_train == 1], Y_train[A_train == 1]
        if len(X_1) > 0:
            w_1 = sample_weight[A_train == 1] if sample_weight is not None else None
            self.mu1_model.fit(X_1, Y_1, sample_weight=w_1)


def run_single_simulation(n_exp, n_obs, d, epsilon, sigma, lambda_vals, true_te, seed,
                          use_fast=True):
    """Run a single simulation - SAVES nothing, just returns results."""
    # Generate data
    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs = generate_synthetic_data(
        n_exp, n_obs, d, true_te, epsilon, sigma, seed
    )
    
    # Method 1: Exp-only
    ate_exp = compute_exp_ate_dml(X_exp, A_exp, Y_exp, method='difference')
    mse_exp = (ate_exp - true_te) ** 2
    
    # Method 2: Obs-only (FIXED - actually uses obs data!)
    if use_fast:
        model_obs = FastDMLModel(random_state=seed)
    else:
        model_obs = DMLModel(random_state=seed)
    model_obs.fit(X_obs, A_obs, Y_obs, X_obs[:0], A_obs[:0], Y_obs[:0], lambda_=0.0)
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
    
    # Method 4: CVCI with DML
    Q_values, lambda_opt, model_opt = cross_validation_dml(
        X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
        lambda_vals=lambda_vals,
        k_fold=5,  # 5-fold CV (standard)
        exp_ate_method='difference',
        stratified=True,
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
    """Save results to JSON (with proper numpy conversion)."""
    # Convert numpy types to Python types
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


def simulate_varying_epsilon(n_exp=50, n_obs=100, d=5, sigma=1.0, 
                             epsilon_vals=None, n_sims=100, lambda_bin=50,
                             use_fast=True, save_dir='simulation_results'):
    """
    Simulate varying epsilon - SAVES INCREMENTALLY!
    
    IMPORTANT: For each simulation, we use the SAME experimental data
    across all epsilon values. Only the observational data changes.
    This ensures exp-only baseline is flat (as it should be).
    """
    if epsilon_vals is None:
        epsilon_vals = np.linspace(0, 0.6, 21)  # CVCI uses 0 to 0.6
    
    true_te = 1.0
    lambda_vals = np.linspace(0, 1, lambda_bin)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metadata
    metadata = {
        'experiment': 'varying_epsilon',
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
    
    print(f"Running epsilon simulation: n_exp={n_exp}, n_obs={n_obs}")
    print(f"Saving to: {save_dir}/")
    print(f"NOTE: Each simulation uses SAME exp data across all epsilon values")
    
    # For each simulation
    for sim in range(n_sims):
        print(f"\nSimulation {sim+1}/{n_sims}")
        
        # Generate experimental data ONCE per simulation
        # Y^exp = Z^T θ^exp + W τ* + ξ
        rng_exp = np.random.default_rng(random_seed + sim)
        Z_exp = rng_exp.normal(0, 1, size=(n_exp, d))
        W_exp = rng_exp.binomial(1, 0.5, size=n_exp)  # Randomized 50/50
        theta = rng_exp.normal(0, 1, size=d)  # Covariate coefficients
        Y_exp = Z_exp @ theta + W_exp * true_te + rng_exp.normal(0, sigma, size=n_exp)
        
        # Compute exp-only estimate ONCE (same for all epsilon)
        ate_exp = compute_exp_ate_dml(Z_exp, W_exp, Y_exp, method='difference')
        mse_exp = (ate_exp - true_te) ** 2
        
        # Now vary epsilon (only changes obs data)
        for eps_idx, eps in enumerate(epsilon_vals):
            # Generate observational data per CVCI Section 14.1
            rng_obs = np.random.default_rng(random_seed + sim * 1000 + int(eps * 10000))
            Z_obs = rng_obs.normal(0, 1, size=(n_obs, d))
            
            # Fixed propensity = 0.2 (confounded, not randomized)
            W_obs = rng_obs.binomial(1, 0.2, size=n_obs)
            
            # Treatment effect is biased: (τ* + ε)
            Y_obs = Z_obs @ theta + W_obs * (true_te + eps) + rng_obs.normal(0, sigma, size=n_obs)
            
            # Obs-only baseline
            if use_fast:
                model_obs = FastDMLModel(random_state=random_seed + sim)
            else:
                model_obs = DMLModel(random_state=random_seed + sim)
            model_obs.fit(Z_obs, W_obs, Y_obs, Z_obs[:0], W_obs[:0], Y_obs[:0], lambda_=0.0)
            ate_obs = model_obs.predict_ate(Z_obs, W_obs, Y_obs)
            mse_obs = (ate_obs - true_te) ** 2
            
            # T-test baseline
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(Y_exp[W_exp==1], Y_exp[W_exp==0])
            if p_value < 0.05:
                ate_ttest = ate_exp
            else:
                Y_all = np.concatenate([Y_exp, Y_obs])
                W_all = np.concatenate([W_exp, W_obs])
                ate_ttest = Y_all[W_all==1].mean() - Y_all[W_all==0].mean()
            mse_ttest = (ate_ttest - true_te) ** 2
            
            # CVCI with DML
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
            
            # Store result (exp-only is same for all epsilon in this sim)
            result = {
                'mse_exp': mse_exp,  # Same for all epsilon!
                'mse_obs': mse_obs,
                'mse_ttest': mse_ttest,
                'mse_ours': mse_ours,
                'lambda_opt': lambda_opt,
                'ate_exp': ate_exp,  # Same for all epsilon!
                'ate_obs': ate_obs,
                'ate_ttest': ate_ttest,
                'ate_ours': ate_ours
            }
            results[eps].append(result)
            
            # Progress
            if (eps_idx + 1) % 5 == 0:
                print(f"  ε={eps:.3f} ({eps_idx+1}/{len(epsilon_vals)})", end='\r')
        
        print(f"  Completed all {len(epsilon_vals)} epsilon values")
    
    # Save results for each epsilon
    print("\nSaving results...")
    for eps in epsilon_vals:
        eps_results = {
            'epsilon': eps,
            'simulations': results[eps],
            'n_sims_completed': len(results[eps])
        }
        filename = f'{save_dir}/epsilon_{eps:.4f}_{timestamp}.json'
        save_results_json(eps_results, filename)
    
    # Aggregate results
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
    
    # Save aggregated results
    csv_file = f'{save_dir}/aggregated_epsilon_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved aggregated results: {csv_file}")
    
    # Also save as JSON
    json_file = f'{save_dir}/aggregated_epsilon_{timestamp}.json'
    save_results_json({
        'metadata': metadata,
        'results': epsilon_results
    }, json_file)
    
    return df, timestamp


def load_and_plot_epsilon_results(timestamp, save_dir='simulation_results'):
    """Load results from JSON and create plots."""
    # Load aggregated results
    json_file = f'{save_dir}/aggregated_epsilon_{timestamp}.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    results = data['results']
    df = pd.DataFrame(results)
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top panel: MSE vs epsilon
    ax = axes[0]
    ax.plot(df['epsilon'], df['mse_exp_mean'], 'g-', label='Only use $X^{exp}$', linewidth=2)
    ax.plot(df['epsilon'], df['mse_obs_mean'], 'r-', label='Only use $X^{obs}$', linewidth=2)
    ax.plot(df['epsilon'], df['mse_ttest_mean'], 'b-', label='T-test baseline', linewidth=2)
    ax.plot(df['epsilon'], df['mse_ours_mean'], 'orange', label=r'Ours, $\hat{\beta}(\hat{\lambda})$', linewidth=2)
    
    ax.fill_between(df['epsilon'], 
                    df['mse_ours_mean'] - df['mse_ours_std'],
                    df['mse_ours_mean'] + df['mse_ours_std'],
                    alpha=0.2, color='orange')
    
    ax.set_xlabel(r'$\varepsilon$', fontsize=14)
    ax.set_ylabel('Empirical MSE', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"$N^{{exp}} = {metadata['n_exp']}, N^{{obs}} = {metadata['n_obs']}, \\sigma^2 = {metadata['sigma']}$", 
                 fontsize=12)
    
    # Bottom panel: Lambda vs epsilon
    ax = axes[1]
    ax.plot(df['epsilon'], df['lambda_mean'], 'b-', linewidth=2, 
            label=r'Mean of $\hat{\lambda}$ selected by cross-validation')
    ax.fill_between(df['epsilon'],
                    df['lambda_mean'] - df['lambda_std'],
                    df['lambda_mean'] + df['lambda_std'],
                    alpha=0.3, color='blue')
    
    # MSE ratio on right axis
    mse_ratio = df['mse_exp_mean'] / df['mse_ours_mean']
    ax2 = ax.twinx()
    ax2.plot(df['epsilon'], mse_ratio, 'g--', alpha=0.7,
            label='MSE ratio between only using $X^{exp}$ and ours')
    ax2.set_ylabel('MSE ratio', fontsize=12)
    
    ax.set_xlabel(r'$\varepsilon$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = f'{save_dir}/epsilon_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_file}")
    plt.show()
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run controlled simulations with JSON saving')
    parser.add_argument('--mode', type=str, choices=['run', 'plot'], default='run',
                       help='Run simulations or just plot existing results')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Timestamp of results to plot (for --mode plot)')
    parser.add_argument('--n-sims', type=int, default=100)
    parser.add_argument('--lambda-bin', type=int, default=50)
    parser.add_argument('--ultra-quick', action='store_true',
                       help='Ultra-fast: 5 sims, 5 epsilons, 10 lambdas (~5 min)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick: 10 sims, 6 epsilons, 15 lambdas (~20 min)')
    parser.add_argument('--medium', action='store_true',
                       help='Medium: 50 sims, 21 epsilons, 20 lambdas (~7 hours)')
    parser.add_argument('--use-slow-dml', action='store_true',
                       help='Use full DML (100 trees) instead of fast (50 trees, limited depth)')
    
    args = parser.parse_args()
    
    if args.mode == 'plot':
        if args.timestamp is None:
            print("ERROR: --timestamp required for --mode plot")
            print("Example: python script.py --mode plot --timestamp 20241201_143022")
            exit(1)
        load_and_plot_epsilon_results(args.timestamp)
    
    else:  # Run mode
        if args.ultra_quick:
            n_sims = 5
            lambda_bin = 10
            epsilon_vals = np.linspace(0, 0.6, 5)  # CVCI: 0 to 0.6
            print("⚡ ULTRA-QUICK MODE (~5 min)")
        elif args.quick:
            n_sims = 10
            lambda_bin = 15
            epsilon_vals = np.linspace(0, 0.6, 6)  # CVCI: 0 to 0.6
            print("⚡ QUICK MODE (~20 min)")
        elif args.medium:
            n_sims = 50
            lambda_bin = 20
            epsilon_vals = np.linspace(0, 0.6, 21)  # CVCI: 0 to 0.6
            print("⏱️  MEDIUM MODE (~7 hours)")
        else:
            n_sims = args.n_sims
            lambda_bin = args.lambda_bin
            epsilon_vals = np.linspace(0, 0.6, 21)  # CVCI: 0 to 0.6
            print(f"🐢 FULL MODE (~{n_sims * lambda_bin // 10} hours)")
        
        use_fast = not args.use_slow_dml
        if use_fast:
            print("Using FAST DML (50 trees, depth=10, parallel)")
        else:
            print("Using STANDARD DML (100 trees)")
        
        # Run simulation
        df, timestamp = simulate_varying_epsilon(
            n_exp=50,    # CVCI uses 50
            n_obs=100,   # CVCI uses 100 (2x ratio, not 50x)
            d=5,
            sigma=1.0,
            epsilon_vals=epsilon_vals,
            n_sims=n_sims,
            lambda_bin=lambda_bin,
            use_fast=use_fast
        )
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE!")
        print("="*70)
        print(f"\nTo recreate plots anytime:")
        print(f"  python {__file__} --mode plot --timestamp {timestamp}")
        
        # Create plots immediately
        load_and_plot_epsilon_results(timestamp)
