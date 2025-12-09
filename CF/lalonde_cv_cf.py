"""
Run CF-based CVCI on the LaLonde dataset (full configurations).

This mirrors `lalonde_cv.py` but replaces the linear analytic CVCI
with the causal-forest-based implementation from `cf_cv.py`.

Usage:
    Modify dir_path to save the checkpoint.
    The printed "Simulation" is not in order because of the parallel computing design.
    For parallel computing, change num_workers.
"""

import numpy as np
import json
from datetime import date
import os
import dask
import pandas as pd

from causal_sim import lalonde_get_data
from cf_cv import CFModel, cross_validation_cf

random_seed = 2024
np.random.seed(random_seed)

variables_list = [[],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic'],
                  ['re75'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're74'],
                  ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75', 're74'],
                  ['age', 'education', 'nodegree', 'black', 'hispanic', 'married', 're75', 'u75', 'u74', 're74']
                 ]

group_lists = ['psid', 'psid2', 'psid3', 'cps', 'cps2', 'cps3']  # could also be 'control'

# Load real LaLonde data
df = pd.read_csv('lalonde.csv')
df['age2'] = df['age'] ** 2

lambda_bin = 5        # number of candidate values of lambda, 50 in the paper
n_sims = 100          # number of simulations, 5000 in the paper
lambda_vals = np.linspace(0, 1, lambda_bin)

# storing results (same shapes as lalonde_cv.py)
ours_cv = np.zeros((len(group_lists), len(variables_list)))       # CF-CVCI ATEs from cross-validation
lambda_opt_all = np.zeros((len(group_lists), len(variables_list)))  # lambda values chosen by cross-validation

exp_name = 'lalonde_cv_cf'

# save the checkpoint metadata
data_log = {'Experiment': exp_name,
            'Settings': {
                'lambda_bin': lambda_bin,
                'random_seed': random_seed,
                'n_sims': n_sims,
                'cf_model': 'CFModel',
                'k_fold': 5
            },
            'ours_cv': ours_cv.tolist(),
            'lambda_opt_all': lambda_opt_all.tolist(),
           }

today = str(date.today())
dir_path = f"./{today}/"
filename = data_log['Experiment'] + '_n_sims_' + str(n_sims) + '_lambda_bin_' + str(data_log['Settings']['lambda_bin'])
print('Start saving files at', dir_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")


def _split_X_to_ZAY(X, d):
    """Helper: from stacked [Z, A, Y] matrix, return (Z, A, Y)."""
    Z = X[:, :d]
    A = X[:, d]
    Y = X[:, -1]
    return Z, A, Y


def run_simulation(sim):
    """Single Monte Carlo replication using CF-based CVCI.

    Mirrors the structure of `run_simulation` in `lalonde_cv.py`, but:
      - extracts (Z, A, Y) from X_exp, X_obs
      - runs `cross_validation_cf` instead of `causal_sim.cross_validation`
      - stores CF-based ATEs via `predict_ate`.
    """
    print('Simulation', sim)
    res = {}

    res["lambda_opt_all"] = np.zeros((len(group_lists), len(variables_list)))
    res["ours_cv"] = np.zeros((len(group_lists), len(variables_list)))

    for group_id, group in enumerate(group_lists):
        for variables_id, variables in enumerate(variables_list):
            d_exp = len(variables)
            d_obs = len(variables)

            # Get stacked [Z, A, Y] matrices from the real LaLonde data
            X_exp, X_obs = lalonde_get_data(df, group, variables)

            if d_exp == 0:
                # X_* columns: [A, Y] in this case. Build Z as a column of ones.
                A_exp = X_exp[:, 0]
                Y_exp = X_exp[:, -1]
                Z_exp = np.ones((X_exp.shape[0], 1))

                A_obs = X_obs[:, 0]
                Y_obs = X_obs[:, -1]
                Z_obs = np.ones((X_obs.shape[0], 1))
            else:
                Z_exp, A_exp, Y_exp = _split_X_to_ZAY(X_exp, d_exp)
                Z_obs, A_obs, Y_obs = _split_X_to_ZAY(X_obs, d_obs)

            # CF-based CVCI cross-validation over lambda
            Q_values, lambda_opt, model_opt = cross_validation_cf(
                Z_exp, A_exp, Y_exp,
                Z_obs, A_obs, Y_obs,
                lambda_vals=lambda_vals,
                k_fold=5,
                random_state=sim
            )

            # ATE estimate from the selected CF model: average CATE on experimental data
            ate_cf = model_opt.predict_ate(Z_exp, A_exp, Y_exp)

            lambda_opt_all[group_id][variables_id] = lambda_opt
            ours_cv[group_id][variables_id] = ate_cf

            res["lambda_opt_all"][group_id][variables_id] = lambda_opt
            res["ours_cv"][group_id][variables_id] = ate_cf

    return res


if __name__ == '__main__':
    # run simulations in parallel
    dask.config.set(scheduler='processes', num_workers=1)
    compute_tasks = [dask.delayed(run_simulation)(sim) for sim in range(n_sims)]
    results_list = dask.compute(compute_tasks)[0]

    ours_cv = np.stack([res["ours_cv"] for res in results_list], axis=0)
    lambda_opt_all = np.stack([res["lambda_opt_all"] for res in results_list], axis=0)

    data_log = {'Experiment': exp_name,
                'Settings': {
                    'lambda_bin': lambda_bin,
                    'random_seed': random_seed,
                    'n_sims': n_sims,
                    'cf_model': 'CFModel',
                    'k_fold': 5
                },
                'ours_cv': ours_cv.tolist(),
                'lambda_opt_all': lambda_opt_all.tolist(),
               }

    with open(dir_path + filename + '.json', 'w') as f:
        json.dump(data_log, f)
        print('saved file', filename)

    # Produce a table of results
    lambda_mean = np.mean(lambda_opt_all, axis=0)
    lambda_std = np.std(lambda_opt_all, axis=0)
    theta_mean = np.mean(ours_cv, axis=0)
    theta_std = np.std(ours_cv, axis=0)
    latex_table = ""

    def floor_str(x):
        # floor + 0.5 and convert to str
        return str(int(np.floor(x + 0.5)))

    for group_id in range(len(group_lists)):
        cur_latex_lambda = ""
        cur_latex_theta = ""
        for variables_id in range(len(variables_list)):
            cur_latex_theta += " & " + floor_str(theta_mean[group_id][variables_id]) + "$\\pm$" + floor_str(theta_std[group_id][variables_id])
            cur_latex_lambda += " & (" + f"{lambda_mean[group_id][variables_id]:.1f}" + "$\\pm$" + f"{lambda_std[group_id][variables_id]:.1f}" + ")"
        latex_table += group_lists[group_id] + cur_latex_theta + " \\\\n"
        latex_table += cur_latex_lambda + " \\\\n"

    print(latex_table)
    with open(dir_path + filename + '.txt', 'w') as f:
        f.write(latex_table)
        f.close()
