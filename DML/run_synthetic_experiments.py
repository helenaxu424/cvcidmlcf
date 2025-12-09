"""
Run multiple synthetic DML experiments with different covariate sets.

This script runs lalonde_synthetic_dml.py with different variable configurations
to match the columns in the CVCI paper (Column 1, 3, 8).

Usage:
    python run_synthetic_experiments.py --group psid [--n-sims 100]
"""

import subprocess
import argparse
import sys
from pathlib import Path

def run_experiment(group, variables, n_sims=100):
    """
    Run a single synthetic experiment.
    
    Args:
        group: Observational group (e.g., 'psid', 'cps')
        variables: List of variable names
        n_sims: Number of simulations
    """
    cmd = ['python', 'lalonde_synthetic_dml.py', '--group', group]
    
    if variables:
        cmd.extend(['--variables'] + variables)
    
    # Note: n_sims is hardcoded in the script, not a command line arg
    # You'd need to modify lalonde_synthetic_dml.py to accept --n-sims
    
    print(f"Running experiment: {group} with variables: {variables if variables else 'None'}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error: Experiment failed with return code {result.returncode}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run multiple synthetic DML experiments')
    parser.add_argument('--group', type=str, required=True,
                       choices=['psid', 'psid2', 'psid3', 'cps', 'cps2', 'cps3'],
                       help='Observational group')
    parser.add_argument('--n-sims', type=int, default=100,
                       help='Number of simulations (note: need to modify lalonde_synthetic_dml.py to use this)')
    
    args = parser.parse_args()
    
    # Define covariate sets matching CVCI paper columns
    experiments = [
        {
            'name': 'Column 1: No covariates',
            'variables': []
        },
        {
            'name': 'Column 3: Demographics + RE75',
            'variables': ['age', 'age2', 'education', 'nodegree', 'black', 'hispanic', 're75']
        },
        {
            'name': 'Column 8: All covariates',
            'variables': ['age', 'education', 'nodegree', 'black', 'hispanic', 
                         'married', 're75', 'u75', 'u74', 're74']
        }
    ]
    
    print(f"RUNNING SYNTHETIC DML EXPERIMENTS FOR {args.group.upper()}")
    print(f"\nWill run {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}")
    print()
    
    # Run all experiments
    results = []
    for exp in experiments:
        success = run_experiment(args.group, exp['variables'], args.n_sims)
        results.append({
            'name': exp['name'],
            'variables': exp['variables'],
            'success': success
        })
    
    # Summary
    print("EXPERIMENT SUMMARY")
    
    for result in results:
        status = "SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{status}: {result['name']}")
    
    # Provide visualization command
    print("\nTo create plots, find the JSON files in the date directory and run:")
    print("\npython plot_synthetic_dml_results.py \\")
    print(f"    ./YYYY-MM-DD/lalonde_synthetic_dml_{args.group}_*.json \\")
    print(f"    --group-label {args.group.upper()}")
    print("\nOr use glob pattern:")
    print(f"python plot_synthetic_dml_results.py ./YYYY-MM-DD/lalonde_synthetic_dml_{args.group}_*.json")

if __name__ == '__main__':
    main()
