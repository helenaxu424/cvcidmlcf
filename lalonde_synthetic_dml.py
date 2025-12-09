"""
Synthetic data based on the LaLonde dataset (DML setting) with single observational group.

Uses Double/Debiased Machine Learning instead of linear models.

Usage: 
    Modify dir_path to save the checkpoint. 
    Use --group to indicate which observational group to use. Use --variables to indicate covariates. For example:
    python lalonde_synthetic_dml.py --group "psid"
    python lalonde_synthetic_dml.py --group 'psid' --variables 're75'
    python lalonde_synthetic_dml.py --group 'psid' --variables 'age' 'education' 'nodegree' 'black' 'hispanic' 'married' 're75' 'u75' 'u74' 're74' 
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from dml_cv import cross_validation_dml, DMLModel, compute_exp_ate_dml
import pandas as pd
from sklearn.linear_model import LinearRegression 
import argparse

random_seed = 2024
np.random.seed(random_seed)

parser = argparse.ArgumentParser(description='Run DML synthetic experiments on LaLonde data')
parser.add_argument("--group", type=str, default="psid", 
                   help="Group name: psid, psid2, psid3, cps, cps2, cps3")
parser.add_argument(
    "--variables",
    nargs="*",          
    type=str,
    default=[],
    help="List of variable names"
)
parser.add_argument("--n-sims", type=int, default=100,
                   help="Number of simulations (default: 100)")
parser.add_argument("--lambda-bin", type=int, default=50,
                   help="Number of lambda values to try (default: 50)")

args = parser.parse_args()
group = args.group
variables = args.variables

# Load and prepare LaLonde data
df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

n_sims = args.n_sims  # number of simulations, 5000 in the paper

# Get data dimensions
d_exp = len(variables) if len(variables) > 0 else 1  # Use 1 for no covariates case
d_obs = len(variables) if len(variables) > 0 else 1

# Helper function to get LaLonde data in DML format
def lalonde_get_data_dml(df, group, variables):
    '''
    Select the samples given group and the variables in the LaLonde dataset.
    Returns data in format needed for DML (separate X, A, Y).

    Args:
        df: LaLonde dataset
        group: observational control group
        variables: variables for DML model
    
    Return:
        X_exp, A_exp, Y_exp: experimental data
        X_obs, A_obs, Y_obs: observational data
    '''
    # Experimental data
    X_df_exp = df[df['group'].isin(['control', 'treated'])]
    if len(variables) > 0:
        X_exp = X_df_exp[variables].to_numpy()
    else:
        # No covariates case - use dummy
        X_exp = np.zeros((len(X_df_exp), 1))
    A_exp = X_df_exp['treatment'].to_numpy()
    Y_exp = X_df_exp['re78'].to_numpy()
    
    # Observational data
    X_df_obs = df[df['group'].isin(['treated', group])]
    if len(variables) > 0:
        X_obs = X_df_obs[variables].to_numpy()
    else:
        X_obs = np.zeros((len(X_df_obs), 1))
    A_obs = X_df_obs['treatment'].to_numpy()
    Y_obs = X_df_obs['re78'].to_numpy()
    
    return X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs

# Get the real data to fit models and extract residuals
X_exp, A_exp, Y_exp_real, X_obs, A_obs, Y_obs_real = lalonde_get_data_dml(df, group, variables)

# Fit experimental model to get true treatment effect and residual variance
if len(variables) > 0:
    exp_model = LinearRegression()
    exp_model.fit(np.concatenate((A_exp.reshape(-1, 1), X_exp), axis=1), Y_exp_real)
    Y_predicted = exp_model.predict(np.concatenate((A_exp.reshape(-1, 1), X_exp), axis=1))
    residuals = Y_exp_real - Y_predicted
    residuals_std = np.sqrt(residuals.var(ddof=1))
    true_te = exp_model.coef_[0]
else:
    # No covariates case - simple difference in means
    true_te = Y_exp_real[A_exp == 1].mean() - Y_exp_real[A_exp == 0].mean()
    residuals = Y_exp_real - (A_exp * Y_exp_real[A_exp == 1].mean() + 
                              (1 - A_exp) * Y_exp_real[A_exp == 0].mean())
    residuals_std = np.sqrt(residuals.var(ddof=1))
    Y_predicted = A_exp * Y_exp_real[A_exp == 1].mean() + (1 - A_exp) * Y_exp_real[A_exp == 0].mean()

n_exp = len(X_exp)

# Fit observational model to get residual variance
if len(variables) > 0:
    obs_model = LinearRegression()
    obs_model.fit(np.concatenate((A_obs.reshape(-1, 1), X_obs), axis=1), Y_obs_real)
    Y_obs_predicted = obs_model.predict(np.concatenate((A_obs.reshape(-1, 1), X_obs), axis=1))
    obs_residuals = Y_obs_real - Y_obs_predicted
    obs_residuals_std = np.sqrt(obs_residuals.var(ddof=1))
else:
    # No covariates case
    Y_obs_predicted = A_obs * Y_obs_real[A_obs == 1].mean() + (1 - A_obs) * Y_obs_real[A_obs == 0].mean()
    obs_residuals = Y_obs_real - Y_obs_predicted
    obs_residuals_std = np.sqrt(obs_residuals.var(ddof=1))

n_obs = len(X_obs)

# Storage for results
ours_cv = np.zeros(n_sims)  # our method, estimate from cross-validation
exp_only = np.zeros(n_sims)  # only using X_exp
obs_only = np.zeros(n_sims)  # only using X_obs
lambda_opt_all = np.zeros(n_sims)  # lambda values chosen by cross-validation

lambda_bin = args.lambda_bin  # number of candidate values of lambda, 50 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin)  # candidate lambda values

print(f"Running DML synthetic experiment:")
print(f"  Group: {group}")
print(f"  Variables: {variables if variables else 'None (no covariates)'}")
print(f"  True treatment effect: {true_te:.2f}")
print(f"  n_exp: {n_exp}, n_obs: {n_obs}")
print(f"  Simulations: {n_sims}")
print(f"  Lambda grid: {lambda_bin} values")
print()

for sim in range(n_sims):
    if sim % 20 == 0:
        print(f'Simulation {sim}/{n_sims}')
    
    rng = np.random.default_rng(sim)
    
    # Generate synthetic experimental data with noise
    residuals_sim = rng.normal(0, residuals_std, size=n_exp)
    Y_exp_sim = Y_predicted + residuals_sim
    
    # Estimate experimental-only ATE
    exp_only[sim] = compute_exp_ate_dml(X_exp, A_exp, Y_exp_sim, method='difference')
    
    # Generate synthetic observational data with noise
    obs_residuals_sim = rng.normal(0, obs_residuals_std, size=n_obs)
    Y_obs_sim = Y_obs_predicted + obs_residuals_sim
    
    # Estimate observational-only ATE using DML
    dml_obs_only = DMLModel(random_state=random_seed + sim)
    try:
        dml_obs_only.fit(X_obs, A_obs, Y_obs_sim, X_obs[:0], A_obs[:0], Y_obs_sim[:0], lambda_=0.0)
        obs_only[sim] = dml_obs_only.predict_ate(X_obs, A_obs, Y_obs_sim)
    except Exception as e:
        print(f"  Warning: Obs-only failed in sim {sim}: {e}")
        obs_only[sim] = np.nan
    
    # Run cross-validation with DML
    try:
        Q_values, lambda_opt, model_opt = cross_validation_dml(
            X_exp, A_exp, Y_exp_sim,
            X_obs, A_obs, Y_obs_sim,
            lambda_vals=lambda_vals,
            k_fold=5,
            exp_ate_method='difference',  # Simple mean difference
            stratified=True,
            random_state=random_seed + sim
        )
        
        lambda_opt_all[sim] = lambda_opt
        ours_cv[sim] = model_opt.predict_ate(X_exp, A_exp, Y_exp_sim)
        
    except Exception as e:
        print(f"  Warning: CV failed in sim {sim}: {e}")
        lambda_opt_all[sim] = np.nan
        ours_cv[sim] = np.nan

# Save the checkpoint
data_log = {
    'Experiment': 'lalonde_synthetic_dml',
    'Settings': {
        'group': group, 
        'n_sims': n_sims, 
        'lambda_bin': lambda_bin, 
        'random_seed': random_seed, 
        'true_te': true_te, 
        'variables': variables,
        'method': 'DML'
    },
    'ours_cv': ours_cv.tolist(),
    'exp_only': exp_only.tolist(),
    'obs_only': obs_only.tolist(),
    'lambda_opt_all': lambda_opt_all.tolist(),
}

today = str(date.today())
dir_path = f"./{today}/"
filename = data_log['Experiment'] 
filename = filename + '_' + str(data_log['Settings']['group']) + '_' + str(data_log['Settings']['variables'])

print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")

with open(dir_path + filename + '.json', 'w') as f:
    json.dump(data_log, f)
    print(f"Saved: {dir_path + filename}.json")

# Compute and write results
lambda_opt_mean = np.nanmean(lambda_opt_all)
lambda_opt_std = np.nanstd(lambda_opt_all)

# MSE and RMSE
ours_cv_mse = np.nanmean((ours_cv - true_te)**2)
exp_only_mse = np.nanmean((exp_only - true_te)**2)
obs_only_mse = np.nanmean((obs_only - true_te)**2)

ours_cv_rmse = np.sqrt(ours_cv_mse)
exp_only_rmse = np.sqrt(exp_only_mse)
obs_only_rmse = np.sqrt(obs_only_mse)

# Bias
ours_cv_bias = np.nanmean(ours_cv - true_te)
exp_only_bias = np.nanmean(exp_only - true_te)
obs_only_bias = np.nanmean(obs_only - true_te)

# Standard deviation
ours_cv_std = np.nanstd(ours_cv)
exp_only_std = np.nanstd(exp_only)
obs_only_std = np.nanstd(obs_only)

with open(dir_path + filename + ".txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("SYNTHETIC LALONDE EXPERIMENT - DML METHOD\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Settings:\n")
    f.write(f"  Group: {group}\n")
    f.write(f"  Variables: {variables if variables else 'None (no covariates)'}\n")
    f.write(f"  True treatment effect: {true_te:.2f}\n")
    f.write(f"  n_exp: {n_exp}, n_obs: {n_obs}\n")
    f.write(f"  Simulations: {n_sims}\n\n")
    
    f.write("Lambda Selection:\n")
    f.write(f"  Mean: {lambda_opt_mean:.3f}\n")
    f.write(f"  Std:  {lambda_opt_std:.3f}\n\n")
    
    f.write("Root Mean Squared Error (RMSE):\n")
    f.write(f"  Ours (CVCI-DML):  {ours_cv_rmse:.2f}\n")
    f.write(f"  Exp only:         {exp_only_rmse:.2f}\n")
    f.write(f"  Obs only:         {obs_only_rmse:.2f}\n\n")
    
    f.write("Bias:\n")
    f.write(f"  Ours (CVCI-DML):  {ours_cv_bias:.2f}\n")
    f.write(f"  Exp only:         {exp_only_bias:.2f}\n")
    f.write(f"  Obs only:         {obs_only_bias:.2f}\n\n")
    
    f.write("Standard Deviation:\n")
    f.write(f"  Ours (CVCI-DML):  {ours_cv_std:.2f}\n")
    f.write(f"  Exp only:         {exp_only_std:.2f}\n")
    f.write(f"  Obs only:         {obs_only_std:.2f}\n\n")
    
    f.write("Mean Squared Error (MSE):\n")
    f.write(f"  Ours (CVCI-DML):  {ours_cv_mse:.2f}\n")
    f.write(f"  Exp only:         {exp_only_mse:.2f}\n")
    f.write(f"  Obs only:         {obs_only_mse:.2f}\n\n")
    
    f.write("Interpretation:\n")
    if lambda_opt_mean < 0.1:
        f.write("  ⚠️  Low λ - observational data not helpful (high bias)\n")
    elif lambda_opt_mean > 0.8:
        f.write("  ✓ High λ - observational data strongly weighted\n")
    else:
        f.write("  → Moderate λ - partial use of observational data\n")
    
    if ours_cv_rmse < exp_only_rmse:
        improvement = (1 - ours_cv_rmse / exp_only_rmse) * 100
        f.write(f"  ✓ CVCI-DML improves over exp-only by {improvement:.1f}%\n")
    else:
        degradation = (ours_cv_rmse / exp_only_rmse - 1) * 100
        f.write(f"  ⚠️  CVCI-DML worse than exp-only by {degradation:.1f}%\n")

print(f"\nSaved results to: {dir_path + filename}.txt")
print("\nResults Summary:")
print("="*80)
print(f"True treatment effect: {true_te:.2f}")
print(f"Selected λ: {lambda_opt_mean:.3f} ± {lambda_opt_std:.3f}")
print(f"\nRMSE:")
print(f"  Ours (CVCI-DML): {ours_cv_rmse:.2f}")
print(f"  Exp only:        {exp_only_rmse:.2f}")
print(f"  Obs only:        {obs_only_rmse:.2f}")
print("="*80)