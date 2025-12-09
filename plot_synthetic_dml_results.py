"""
Visualize DML synthetic results in CVCI paper style.

Creates plots matching Figure 5 from Yang et al. (2025):
- Box plots of ATE estimates across methods
- Distribution of selected lambda values

Usage:
    python plot_synthetic_dml_results.py <results_file.json> [--save-path OUTPUT_DIR]
    
Example:
    python plot_synthetic_dml_results.py ./2025-11-24/lalonde_synthetic_dml_psid_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys

# Set style to match CVCI paper
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_ate_estimates(results_list, group_labels, column_labels, save_path=None):
    """
    Create box plots of ATE estimates (Figure 5a/5c style).
    
    Shows raw ATE estimates (not centered) matching CVCI paper Figure 5.
    
    Args:
        results_list: List of result dictionaries (one per column configuration)
        group_labels: Labels for each column (e.g., ['Column 1', 'Column 3', 'Column 8'])
        column_labels: Descriptive labels (e.g., ['{treatment}', '{treatment, RE75}', ...])
        save_path: Path to save figure
    """
    n_cols = len(results_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors matching CVCI paper
    colors = {
        'exp_only': '#90EE90',  # Light green
        'ours': '#FFFFCC',      # Light yellow
        'obs_only': '#FFB6C6',  # Light red/pink
        'pooled': '#D3D3D3'     # Light gray
    }
    
    positions = []
    box_data = []
    box_colors = []
    labels = []
    
    offset = 0
    spacing = 1.5
    
    for col_idx, (results, label) in enumerate(zip(results_list, column_labels)):
        true_te = results['Settings']['true_te']
        
        # Get estimates
        exp_only = np.array(results['exp_only'])
        ours = np.array(results['ours_cv'])
        obs_only = np.array(results['obs_only'])
        
        # Remove NaNs
        exp_only = exp_only[~np.isnan(exp_only)]
        ours = ours[~np.isnan(ours)]
        obs_only = obs_only[~np.isnan(obs_only)]
        
        # Use raw estimates (NOT centered) - matching CVCI paper Figure 5
        exp_only_plot = exp_only
        ours_plot = ours
        obs_only_plot = obs_only
        
        # Positions for this column
        pos_exp = offset + 0
        pos_ours = offset + 0.5
        pos_obs = offset + 1.0
        
        positions.extend([pos_exp, pos_ours, pos_obs])
        box_data.extend([exp_only_plot, ours_plot, obs_only_plot])
        box_colors.extend([colors['exp_only'], colors['ours'], colors['obs_only']])
        
        offset += spacing
    
    # Create box plots
    bp = ax.boxplot(box_data, positions=positions, widths=0.35,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Add ground truth line at the true treatment effect value
    # Use the first result's true_te (they should all be the same within a group)
    true_te = results_list[0]['Settings']['true_te']
    ax.axhline(y=true_te, color='black', linestyle='--', linewidth=1.5, 
               label='Ground-truth ATE', zorder=1)
    
    # Set x-axis labels
    x_ticks = []
    x_labels_main = []
    for col_idx, label in enumerate(column_labels):
        x_ticks.append(col_idx * spacing + 0.5)
        x_labels_main.append(label)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels_main)
    
    # Add "Regress on:" label below
    ax.text(0.5, -0.15, 'Regress on:', transform=ax.transAxes,
            ha='center', va='top', fontsize=10)
    
    ax.set_ylabel('ATE Estimates', fontsize=11)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Auto-adjust y-axis limits to use space efficiently
    all_data = np.concatenate([d for d in box_data])
    y_min, y_max = all_data.min(), all_data.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
    
    # Create legend - MOVED TO BOTTOM RIGHT
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=colors['exp_only'], ec='black', label='Synthetic $\\tilde{X}^{\\mathrm{exp}}$ only, $\\lambda=0$'),
        Rectangle((0, 0), 1, 1, fc=colors['ours'], ec='black', label='Ours, $\\hat{\\lambda}$ by cross-validation'),
        Rectangle((0, 0), 1, 1, fc=colors['obs_only'], ec='black', label='Synthetic $\\tilde{X}^{\\mathrm{obs}}$ only, $\\lambda=1$'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Ground-truth ATE')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=False, 
              edgecolor='black', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def plot_lambda_distribution(results_list, group_labels, column_labels, save_path=None):
    """
    Create distribution plots of selected lambda (Figure 5b/5d style).
    
    Args:
        results_list: List of result dictionaries
        group_labels: Labels for each column
        column_labels: Descriptive labels
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors for different columns
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for col_idx, (results, label, color) in enumerate(zip(results_list, column_labels, colors)):
        lambda_vals = np.array(results['lambda_opt_all'])
        lambda_vals = lambda_vals[~np.isnan(lambda_vals)]
        
        # Create KDE
        from scipy import stats
        kde = stats.gaussian_kde(lambda_vals, bw_method=0.1)
        
        # Evaluate KDE
        x_eval = np.linspace(0, 1, 200)
        y_eval = kde(x_eval)
        
        # Plot with fill
        ax.plot(x_eval, y_eval, color=color, linewidth=2, label=label)
        ax.fill_between(x_eval, 0, y_eval, color=color, alpha=0.3)
    
    ax.set_xlabel('$\\hat{\\lambda}$ selected by cross-validation', fontsize=11)
    ax.set_ylabel('Density (log scale)', fontsize=11)  # Updated label
    ax.set_xlim(0, 1)
    ax.set_yscale('log')  # SET LOG SCALE FOR Y-AXIS
    ax.set_ylim(bottom=0.01, top=1e4)  # Set max to 10^4
    ax.grid(True, alpha=0.3, axis='both')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def create_summary_table(results_list, column_labels, save_path=None):
    """
    Create summary table matching CVCI paper format.
    
    Args:
        results_list: List of result dictionaries
        column_labels: Column labels
        save_path: Path to save table
    """
    table_lines = []
    table_lines.append("="*100)
    table_lines.append("SYNTHETIC DATA RESULTS - DML METHOD")
    table_lines.append("="*100)
    table_lines.append("")
    
    # Header
    header = f"{'Column No.':<25}"
    for i, label in enumerate(column_labels, 1):
        header += f"{str(i):>20}"
    table_lines.append(header)
    
    regress_line = f"{'Regress RE78 on:':<25}"
    for label in column_labels:
        regress_line += f"{label:>20}"
    table_lines.append(regress_line)
    table_lines.append("-"*100)
    
    # Exp-only row
    exp_line = f"{'(Î»=0, XÌƒáµ‰Ë£áµ– only) NSW(T+C), RMSE':<25}"
    for results in results_list:
        exp_only = np.array(results['exp_only'])
        exp_only = exp_only[~np.isnan(exp_only)]
        true_te = results['Settings']['true_te']
        rmse = np.sqrt(np.mean((exp_only - true_te)**2))
        exp_line += f"{rmse:>20.1f}"
    table_lines.append(exp_line)
    table_lines.append("")
    
    # Ours row
    group = results_list[0]['Settings']['group']
    ours_line1 = f"{'(Î»Ì‚, ours) XÌƒáµ‰Ë£áµ– + XÌƒáµ’áµ‡Ë¢, Xáµ‰Ë£áµ–: NSW(T+C),':<25}"
    table_lines.append(ours_line1)
    
    obs_line = f"{'Xáµ’áµ‡Ë¢ includes NSW(T) and:':<25}{group.upper():>20}"
    for _ in range(len(column_labels) - 1):
        obs_line += f"{group.upper():>20}"
    table_lines.append(obs_line)
    table_lines.append("")
    
    rmse_line = f"{'RMSE':<25}"
    for results in results_list:
        ours = np.array(results['ours_cv'])
        ours = ours[~np.isnan(ours)]
        true_te = results['Settings']['true_te']
        rmse = np.sqrt(np.mean((ours - true_te)**2))
        rmse_line += f"{rmse:>20.1f}"
    table_lines.append(rmse_line)
    
    lambda_line = f"{'Î»Ì‚ =':<25}"
    for results in results_list:
        lambda_vals = np.array(results['lambda_opt_all'])
        lambda_vals = lambda_vals[~np.isnan(lambda_vals)]
        mean_lambda = np.mean(lambda_vals)
        std_lambda = np.std(lambda_vals)
        lambda_line += f"{mean_lambda:>8.1f} Â± {std_lambda:<8.1f}"
    table_lines.append(lambda_line)
    table_lines.append("")
    
    # Obs-only row
    obs_only_line1 = f"""{"(Î»=1, XÌƒáµ’áµ‡Ë¢ only) [2]'s setting,":<25}"""
    table_lines.append(obs_only_line1)
    
    obs_line2 = f"{'Xáµ’áµ‡Ë¢ includes NSW(T) and:':<25}{group.upper():>20}"
    for _ in range(len(column_labels) - 1):
        obs_line2 += f"{group.upper():>20}"
    table_lines.append(obs_line2)
    table_lines.append("")
    
    obs_rmse_line = f"{'RMSE':<25}"
    for results in results_list:
        obs_only = np.array(results['obs_only'])
        obs_only = obs_only[~np.isnan(obs_only)]
        true_te = results['Settings']['true_te']
        rmse = np.sqrt(np.mean((obs_only - true_te)**2))
        obs_rmse_line += f"{rmse:>20.1f}"
    table_lines.append(obs_rmse_line)
    
    table_lines.append("="*100)
    
    table_text = "\n".join(table_lines)
    print("\n" + table_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_text)
        print(f"\nSaved table: {save_path}")
    
    return table_text

def main():
    parser = argparse.ArgumentParser(description='Visualize DML synthetic results')
    parser.add_argument('files', nargs='+', help='Path(s) to result JSON files')
    parser.add_argument('--save-path', type=str, default=None, 
                       help='Directory to save plots (default: same as first file)')
    parser.add_argument('--group-label', type=str, default=None,
                       help='Label for the group (e.g., "PSID" or "CPS")')
    
    args = parser.parse_args()
    
    # Load all results
    results_list = []
    seen_configs = set()  # Track unique covariate configurations
    
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        results = load_results(filepath)
        
        # Create a unique key for this configuration
        variables = tuple(sorted(results['Settings']['variables']))
        config_key = (results['Settings']['group'], variables)
        
        # Only add if we haven't seen this exact configuration
        if config_key not in seen_configs:
            results_list.append(results)
            seen_configs.add(config_key)
            print(f"Loaded: {filepath}")
        else:
            print(f"Skipping duplicate config: {filepath}")
    
    if len(results_list) == 0:
        print("Error: No results loaded")
        sys.exit(1)
    
    # Sort by number of variables (Column 1, 3, 8 order)
    results_list = sorted(results_list, 
                         key=lambda x: len(x['Settings']['variables']))
    
    # Determine group label
    if args.group_label:
        group_label = args.group_label
    else:
        group_label = results_list[0]['Settings']['group'].upper()
    
    # Create column labels based on variables
    column_labels = []
    group_labels = []
    
    print(f"\nDetected {len(results_list)} configuration(s):")
    
    for i, results in enumerate(results_list, 1):
        variables = results['Settings']['variables']
        n_vars = len(variables)
        
        # Identify standard CVCI columns by exact covariate composition
        if n_vars == 0:
            label = '{treatment}'
            col_name = 'Column 1'
        elif n_vars == 7 and 're75' in variables and 'age2' in variables and 'nodegree' in variables:
            # Standard Column 3: Demographics + RE75 (has age2, no married/u75/u74/re74)
            label = '{treatment, demographics + RE75}'
            col_name = 'Column 3'
        elif n_vars == 10 and 'married' in variables and 'u75' in variables and 'u74' in variables and 're74' in variables:
            # Standard Column 8: All covariates (has married, u75, u74, re74)
            label = '{treatment, all covariates}'
            col_name = 'Column 8'
        elif n_vars >= 8:
            # Some other "all covariates" configuration
            # Show distinguishing variables
            key_vars = []
            if 'married' in variables:
                key_vars.append('married')
            if 'u75' in variables:
                key_vars.append('u75')
            if 're74' in variables:
                key_vars.append('re74')
            
            if key_vars:
                label = f'{{treatment, {n_vars} cov. ({", ".join(key_vars[:2])})}}'
            else:
                label = f'{{treatment, {n_vars} covariates}}'
            col_name = f'Column {i}'
        elif n_vars == 7:
            # Could be Column 3 or something else - check for age2
            if 'age2' in variables and 're75' in variables:
                label = '{treatment, demographics + RE75}'
                col_name = 'Column 3'
            else:
                label = f'{{treatment, {n_vars} covariates}}'
                col_name = f'Column {i}'
        elif n_vars <= 3:
            var_str = ', '.join(variables[:2])
            if n_vars > 2:
                var_str += ', ...'
            label = f'{{treatment, {var_str}}}'
            col_name = f'Column {i}'
        else:
            # Show some key variables to distinguish
            key_vars = [v for v in ['age', 'education', 're75', 're74', 'married'] if v in variables]
            if key_vars:
                var_str = ', '.join(key_vars[:2])
                if len(key_vars) > 2:
                    var_str += ', ...'
                label = f'{{treatment, {var_str}}}'
            else:
                label = f'{{treatment, {n_vars} covariates}}'
            col_name = f'Column {i}'
        
        column_labels.append(label)
        group_labels.append(col_name)
        
        # Print what we detected
        print(f"  {col_name}: {label}")
        print(f"    Variables ({n_vars}): {variables}")
    
    # Determine save path
    if args.save_path:
        save_dir = Path(args.save_path)
    else:
        save_dir = Path(args.files[0]).parent / 'synthetic_plots'
    
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving plots to: {save_dir}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # ATE estimates box plot
    fig1 = plot_ate_estimates(
        results_list, group_labels, column_labels,
        save_path=save_dir / f'ate_estimates_synthetic_{group_label.lower()}.png'
    )
    plt.close(fig1)
    
    # Lambda distribution
    fig2 = plot_lambda_distribution(
        results_list, group_labels, column_labels,
        save_path=save_dir / f'lambda_distribution_synthetic_{group_label.lower()}.png'
    )
    plt.close(fig2)
    
    # Summary table
    create_summary_table(
        results_list, column_labels,
        save_path=save_dir / f'summary_table_synthetic_{group_label.lower()}.txt'
    )
    
    # Combined figure (2x1 layout like CVCI paper)
    print("\nCreating combined figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Recreate plots in subplots
    # Box plots
    positions = []
    box_data = []
    box_colors = []
    
    colors = {
        'exp_only': '#90EE90',
        'ours': '#FFFFCC',
        'obs_only': '#FFB6C6',
    }
    
    offset = 0
    spacing = 1.5
    
    for col_idx, results in enumerate(results_list):
        true_te = results['Settings']['true_te']
        exp_only = np.array(results['exp_only'])
        ours = np.array(results['ours_cv'])
        obs_only = np.array(results['obs_only'])
        
        # Use raw estimates (NOT centered) - matching CVCI paper
        exp_only = exp_only[~np.isnan(exp_only)]
        ours = ours[~np.isnan(ours)]
        obs_only = obs_only[~np.isnan(obs_only)]
        
        pos_exp = offset + 0
        pos_ours = offset + 0.5
        pos_obs = offset + 1.0
        
        positions.extend([pos_exp, pos_ours, pos_obs])
        box_data.extend([exp_only, ours, obs_only])
        box_colors.extend([colors['exp_only'], colors['ours'], colors['obs_only']])
        
        offset += spacing
    
    bp = ax1.boxplot(box_data, positions=positions, widths=0.35,
                     patch_artist=True, showfliers=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Add ground truth line at actual value (not 0)
    true_te = results_list[0]['Settings']['true_te']
    ax1.axhline(y=true_te, color='black', linestyle='--', linewidth=1.5, zorder=1)
    
    x_ticks = [i * spacing + 0.5 for i in range(len(results_list))]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(column_labels)
    ax1.set_ylabel('ATE Estimates', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Auto-adjust y-axis limits to use space efficiently
    all_data = np.concatenate([d for d in box_data])
    y_min, y_max = all_data.min(), all_data.max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
    
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=colors['exp_only'], ec='black', 
                 label='Synthetic $\\tilde{X}^{\\mathrm{exp}}$ only, $\\lambda=0$'),
        Rectangle((0, 0), 1, 1, fc=colors['ours'], ec='black', 
                 label='Ours, $\\hat{\\lambda}$ by cross-validation'),
        Rectangle((0, 0), 1, 1, fc=colors['obs_only'], ec='black', 
                 label='Synthetic $\\tilde{X}^{\\mathrm{obs}}$ only, $\\lambda=1$'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, 
                  label='Ground-truth ATE')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=False, edgecolor='black', fontsize=9)
    
    # Lambda distribution
    plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for col_idx, (results, label, color) in enumerate(zip(results_list, column_labels, plot_colors)):
        lambda_vals = np.array(results['lambda_opt_all'])
        lambda_vals = lambda_vals[~np.isnan(lambda_vals)]
        
        from scipy import stats
        kde = stats.gaussian_kde(lambda_vals, bw_method=0.1)
        x_eval = np.linspace(0, 1, 200)
        y_eval = kde(x_eval)
        
        ax2.plot(x_eval, y_eval, color=color, linewidth=2, label=label)
        ax2.fill_between(x_eval, 0, y_eval, color=color, alpha=0.3)
    
    ax2.set_xlabel('$\\hat{\\lambda}$ selected by cross-validation', fontsize=11)
    ax2.set_ylabel('Density (log scale)', fontsize=11)  # Updated label
    ax2.set_xlim(0, 1)
    ax2.set_yscale('log')  # SET LOG SCALE FOR Y-AXIS
    ax2.set_ylim(bottom=0.01, top=1e4)  # Set max to 10^4
    ax2.grid(True, alpha=0.3, axis='both')
    ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    
    # Add subplot labels
    ax1.text(-0.1, 1.05, '(a) ATE estimates on synthetic ' + group_label + '.',
            transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.1, 1.05, '(b) Selected $\\hat{\\lambda}$ on synthetic ' + group_label + '.',
            transform=ax2.transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    combined_path = save_dir / f'combined_synthetic_{group_label.lower()}.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure: {combined_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"Output directory: {save_dir}")
    print(f"\nFiles created:")
    print(f"  - ate_estimates_synthetic_{group_label.lower()}.png")
    print(f"  - lambda_distribution_synthetic_{group_label.lower()}.png")
    print(f"  - summary_table_synthetic_{group_label.lower()}.txt")
    print(f"  - combined_synthetic_{group_label.lower()}.png")

if __name__ == '__main__':
    main()