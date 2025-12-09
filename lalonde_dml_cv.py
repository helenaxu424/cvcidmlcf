"""
Run DML method on the LaLonde dataset. Full configurations.

Mirrors the structure of lalonde_cv.py but uses DML instead of linear models.

Usage: 
    Modify dir_path to save the checkpoint. 
    The printed "Simulation" is not in order because of the parallel computing design.
    For parallel computing, change num_workers.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json
from datetime import date
import os
from dml_cv import cross_validation_dml, DMLModel
import dask
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

random_seed = 2024
np.random.seed(random_seed)

variables_list = [[], 
                  ['age', 'age2', 'education','nodegree', 'black', 'hispanic'],
                  ['re75'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75'],
                  ['age', 'age2', 'education','nodegree', 'black', 'hispanic', 're74'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75', 're74'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74']
                 ]

group_lists = ['psid', 'psid2', 'psid3', 'cps', 'cps2', 'cps3'] # could also be 'control'

df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

lambda_bin = 5 # number of candidate values of lambda, 50 in the paper
n_sims = 100 # number of simulations, 5000 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin) # candidate lambda values
stratified_kfold = True # whether to stratify for cross-validation
k_fold = 5 # K-fold cross-validation

# storing results
ours_cv = np.zeros((len(group_lists), len(variables_list))) # our method, estimate from cross-validation
lambda_opt_all = np.zeros((len(group_lists), len(variables_list))) # lambda values chosen by cross-validation

exp_name = 'lalonde_dml_cv'

# save the checkpoint
data_log = {'Experiment': exp_name,
            'Settings': {'lambda_bin': lambda_bin, 'random_seed': random_seed, 'n_sims': n_sims,
                        'stratified_kfold': stratified_kfold, 'k_fold': k_fold, 'method': 'DML'},
            'ours_cv': ours_cv.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }
today = str(date.today())
dir_path =  f"./{today}/" 
filename = data_log['Experiment'] + '_n_sims_' + str(n_sims) + '_lambda_bin_' + str(data_log['Settings']['lambda_bin']) 
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")


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

       
def run_simulation(sim):
    print('Simulation', sim)
    res = {} # result for the current run 
    
    res["lambda_opt_all"] = np.zeros((len(group_lists), len(variables_list)))
    res["ours_cv"] = np.zeros((len(group_lists), len(variables_list)))

    for group_id, group in enumerate(group_lists):
        for variables_id, variables in enumerate(variables_list):
            print(f'  Sim {sim}: Group {group} ({group_id+1}/{len(group_lists)}), Variables {variables_id}/{len(variables_list)}')
            
            # Get data
            X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs = lalonde_get_data_dml(df, group, variables)
            
            try:
                # Run cross-validation
                Q_values, lambda_opt, model_opt = cross_validation_dml(
                    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                    lambda_vals=lambda_vals,
                    k_fold=k_fold,
                    exp_ate_method='difference',  # Simple mean difference for experimental benchmark
                    stratified=stratified_kfold,
                    random_state=random_seed + sim
                )
                
                # Get final ATE estimate
                ate_final = model_opt.predict_ate(X_exp, A_exp, Y_exp)
                
                # Store results
                lambda_opt_all[group_id][variables_id] = lambda_opt
                ours_cv[group_id][variables_id] = ate_final
                res["lambda_opt_all"][group_id][variables_id] = lambda_opt
                res["ours_cv"][group_id][variables_id] = ate_final
                
            except Exception as e:
                print(f'  ERROR in sim {sim}, group {group}, variables {variables_id}: {e}')
                # Store NaN for failed cases
                res["lambda_opt_all"][group_id][variables_id] = np.nan
                res["ours_cv"][group_id][variables_id] = np.nan
    
    return res
  
    
if __name__ == '__main__':
    # run simulations in parallel
    dask.config.set(scheduler='processes', num_workers = 1)
    compute_tasks = [dask.delayed(run_simulation)(sim) for sim in range(n_sims)]
    results_list = dask.compute(compute_tasks)[0]

    ours_cv = np.stack([res["ours_cv"] for res in results_list], axis=0)
    lambda_opt_all = np.stack([res["lambda_opt_all"] for res in results_list], axis=0)
    
    data_log = {'Experiment': exp_name,
            'Settings': {'lambda_bin': lambda_bin, 'random_seed': random_seed, 'n_sims': n_sims,
                        'stratified_kfold': stratified_kfold, 'k_fold': k_fold, 'method': 'DML'},
            'ours_cv': ours_cv.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }

    with open(dir_path + filename + '.json', 'w') as f:
        json.dump(data_log, f)
        print('saved file', filename)

    """
    Produce a table of results.
    """
    # Use nanmean/nanstd to handle any failed simulations
    lambda_mean = np.nanmean(lambda_opt_all, axis=0)
    lambda_std = np.nanstd(lambda_opt_all, axis=0)
    theta_mean = np.nanmean(ours_cv, axis=0)
    theta_std = np.nanstd(ours_cv, axis=0)
    latex_table = ""
    
    def floor_str(x):
        # floor + 0.5 and convert to str
        if np.isnan(x):
            return "N/A"
        return str(int(np.floor(x + 0.5)))
    
    for group_id in range(len(group_lists)):
        cur_latex_lambda = ""
        cur_latex_theta = ""
        for variables_id in range(len(variables_list)):
            theta_val = theta_mean[group_id][variables_id]
            theta_se = theta_std[group_id][variables_id]
            lambda_val = lambda_mean[group_id][variables_id]
            lambda_se = lambda_std[group_id][variables_id]
            
            if np.isnan(theta_val):
                cur_latex_theta += " & N/A"
                cur_latex_lambda += " & N/A"
            else:
                cur_latex_theta += " & " + floor_str(theta_val) + "$\\pm$" + floor_str(theta_se)
                cur_latex_lambda += " & (" + f"{lambda_val:.1f}" + "$\\pm$" + f"{lambda_se:.1f}" + ")"
        
        latex_table += group_lists[group_id] + cur_latex_theta + " \\\\\n"
        latex_table += cur_latex_lambda + " \\\\\n"
    
    print("\n" + "="*80)
    print("RESULTS TABLE (DML)")
    print("="*80)
    print(latex_table)
    
    with open(dir_path + filename + '.txt', 'w') as f:
        f.write(latex_table)
        f.close()
    
    print(f"\nResults saved to {dir_path}")
    print(f"  JSON: {filename}.json")
    print(f"  Table: {filename}.txt")
