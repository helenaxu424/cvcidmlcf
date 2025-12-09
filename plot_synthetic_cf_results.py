"""
Visualize Causal Forest (CF) synthetic results in CVCI paper style.

Creates plots matching Figure 5 from Yang et al. (2025):
- Box plots of ATE estimates across methods
- Distribution of selected lambda values

Usage:
    python plot_synthetic_cf_results.py <results_file.json> [--save-path OUTPUT_DIR]

Example:
    python plot_synthetic_cf_results.py ./2025-11-24/lalonde_synthetic_cf_psid_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys

# Set style (same as original script)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_ate_estimates(results_list, group_labels, column_labels, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'exp_only': '#90EE90',
        'ours': '#FFFFCC',
        'obs_only': '#FFB6C6',
        'pooled': '#D3D3D3'
    }

    positions, box_data, box_colors = [], [], []
    offset, spacing = 0, 1.5

    for results, label in zip(results_list, column_labels):
        exp_only = np.array(results['exp_only'])
        ours = np.array(results['ours_cv'])
        obs_only = np.array(results['obs_only'])

        exp_only = exp_only[~np.isnan(exp_only)]
        ours = ours[~np.isnan(ours)]
        obs_only = obs_only[~np.isnan(obs_only)]

        pos_exp = offset + 0
        pos_ours = offset + 0.5
        pos_obs = offset + 1.0

        positions.extend([pos_exp, pos_ours, pos_obs])
        box_data.extend([exp_only, ours, obs_only])
        box_colors.extend([
            colors['exp_only'], colors['ours'], colors['obs_only']
        ])

        offset += spacing

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.35, patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    true_te = results_list[0]['Settings']['true_te']
    ax.axhline(y=true_te, color='black', linestyle='--', linewidth=1.5)

    x_ticks = [i * spacing + 0.5 for i in range(len(results_list))]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(column_labels)

    ax.set_ylabel('ATE Estimates')
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=colors['exp_only'], ec='black',
                  label='Synthetic $\\tilde{X}^{exp}$ only, $\\lambda = 0$'),
        Rectangle((0, 0), 1, 1, fc=colors['ours'], ec='black',
                  label='Ours (CF), $\\hat{\\lambda}$'),
        Rectangle((0, 0), 1, 1, fc=colors['obs_only'], ec='black',
                  label='Synthetic $\\tilde{X}^{obs}$ only, $\\lambda = 1$'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                   label='Ground-truth ATE')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig

def plot_lambda_distribution(results_list, group_labels, column_labels, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    from scipy import stats
    for results, label, color in zip(results_list, column_labels, colors):
        lambda_vals = np.array(results['lambda_opt_all'])
        lambda_vals = lambda_vals[~np.isnan(lambda_vals)]
        
        # Check if lambda values have sufficient variance for KDE
        if len(np.unique(lambda_vals)) <= 1 or np.std(lambda_vals) < 0.05:
            # All/most lambdas are the same value - plot as a vertical line instead
            lambda_mean = np.mean(lambda_vals)
            ax.axvline(x=lambda_mean, color=color, linewidth=2, linestyle='--', 
                      label=label, alpha=0.7)
        else:
            kde = stats.gaussian_kde(lambda_vals, bw_method=0.1)
            x_eval = np.linspace(0, 1, 200)
            y_eval = kde(x_eval)

            ax.plot(x_eval, y_eval, color=color, linewidth=2, label=label)
            ax.fill_between(x_eval, 0, y_eval, color=color, alpha=0.3)

    ax.set_xlabel('$\\hat{\\lambda}$ (CF)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize CF synthetic results')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--group-label', type=str)
    args = parser.parse_args()

    results_list = []
    seen_configs = set()

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        results = load_results(filepath)

        variables = tuple(sorted(results['Settings']['variables']))
        config_key = (results['Settings']['group'], variables)

        if config_key not in seen_configs:
            seen_configs.add(config_key)
            results_list.append(results)
            print(f"Loaded: {filepath}")
        else:
            print(f"Skipping duplicate config: {filepath}")

    if len(results_list) == 0:
        print("Error: No results loaded.")
        sys.exit(1)

    results_list = sorted(results_list, key=lambda x: len(x['Settings']['variables']))

    group_label = args.group_label or results_list[0]['Settings']['group'].upper()

    column_labels = []
    group_labels = []

    print(f"\nDetected {len(results_list)} CF configuration(s):")

    for i, r in enumerate(results_list, 1):
        variables = r['Settings']['variables']
        n_vars = len(variables)

        if n_vars == 0:
            label = '{treatment}'
            col_name = 'Column 1'
        elif n_vars == 7 and 'age2' in variables:
            label = '{treatment, demographics + RE75}'
            col_name = 'Column 3'
        elif n_vars >= 9:
            label = '{treatment, all covariates}'
            col_name = 'Column 8'
        else:
            label = f'{{treatment, {n_vars} covariates}}'
            col_name = f'Column {i}'

        column_labels.append(label)
        group_labels.append(col_name)

        print(f"  {col_name}: {label}")
        print(f"    Variables ({n_vars}): {variables}")

    save_dir = Path(args.save_path) if args.save_path else \
               (Path(args.files[0]).parent / 'synthetic_plots_cf')

    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving plots to: {save_dir}")

    fig1 = plot_ate_estimates(
        results_list, group_labels, column_labels,
        save_path=save_dir / f'ate_estimates_synthetic_cf_{group_label.lower()}.png'
    )
    plt.close(fig1)

    fig2 = plot_lambda_distribution(
        results_list, group_labels, column_labels,
        save_path=save_dir / f'lambda_distribution_synthetic_cf_{group_label.lower()}.png'
    )
    plt.close(fig2)

    print("\nAll CF plots generated successfully.")

if __name__ == '__main__':
    main()
