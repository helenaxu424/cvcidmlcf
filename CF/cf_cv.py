"""
CVCI with Causal Forests (CF)

Extends the Yang et al. (2025) CVCI framework to use Causal Forests
instead of linear models or DML.

Uses econml.grf.CausalForest for treatment effect estimation.

Key differences from linear / DML CVCI:
- Model class θ is a causal forest that directly estimates CATE τ(X)
- ATE is estimated by averaging τ(X)
- Observational loss uses a simple outcome reconstruction: Y ≈ μ0 + τ(X) * A
- Hybrid loss implemented via sample weights as in the DML version
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
import pandas as pd
from econml.grf import CausalForest

random_seed = 2024
np.random.seed(random_seed)


class CFModel:
    """
    Causal Forest model for the CVCI framework.
    
    Uses econml.grf.CausalForest to estimate CATE τ(X) and then derives:
    - ATE by averaging τ(X)
    - Outcome predictions via Y ≈ μ0 + τ(X) * A, where μ0 is baseline outcome
    """
    
    def __init__(self, random_state=None, n_estimators=500, min_samples_leaf=5):
        """
        Args:
            random_state: Random seed
            n_estimators: Number of trees in the causal forest
            min_samples_leaf: Minimum leaf size
        """
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        
        # Underlying causal forest model
        self.cf_model = None
        
        # Baseline outcome for controls (used for outcome reconstruction)
        self.mu0_ = None
        
        # Store last ATE estimate
        self.ate_estimate = None
    
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """
        Fit CF model using hybrid loss (via sample weights on combined data).
        
        Strategy:
        - Combine experimental and observational data with weights
        - Fit a CausalForest on combined (X, A, Y)
        - Store baseline outcome μ0 for controls to reconstruct outcomes
        
        Args:
            X_exp, A_exp, Y_exp: Experimental covariates, treatment, outcome
            X_obs, A_obs, Y_obs: Observational covariates, treatment, outcome
            lambda_: Mixing parameter (0 = exp only, 1 = obs only)
        """
        # Combine data with weights based on lambda
        if lambda_ < 0.01:
            # Use only experimental data
            X_train = X_exp
            A_train = A_exp
            Y_train = Y_exp
            sample_weight = None
        elif lambda_ > 0.99:
            # Use only observational data
            X_train = X_obs
            A_train = A_obs
            Y_train = Y_obs
            sample_weight = None
        else:
            # Combine both datasets with weights
            X_train = np.vstack([X_exp, X_obs])
            A_train = np.concatenate([A_exp, A_obs])
            Y_train = np.concatenate([Y_exp, Y_obs])
            
            # Sample weights based on hybrid loss:
            # L(θ, λ) = (1-λ) L_exp + λ L_obs
            # Each exp sample contributes (1-λ)/n_exp to total loss
            # Each obs sample contributes λ/n_obs to total loss
            n_exp = len(X_exp)
            n_obs = len(X_obs)
            
            # Weight each experimental sample by (1-λ)/n_exp
            weights_exp = np.ones(n_exp) * (1 - lambda_) / n_exp
            
            # Weight each observational sample by λ/n_obs
            weights_obs = np.ones(n_obs) * lambda_ / n_obs
            
            sample_weight = np.concatenate([weights_exp, weights_obs])
        
        # Fit causal forest
        self.cf_model = CausalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        if sample_weight is not None:
            self.cf_model.fit(X_train, A_train, Y_train, sample_weight=sample_weight)
        else:
            self.cf_model.fit(X_train, A_train, Y_train)
        
        # Compute baseline outcome μ0 (controls) for outcome reconstruction
        mask0 = (A_train == 0)
        if np.any(mask0):
            if sample_weight is not None:
                w0 = sample_weight[mask0]
                self.mu0_ = np.average(Y_train[mask0], weights=w0)
            else:
                self.mu0_ = Y_train[mask0].mean()
        else:
            # Fallback: overall mean if no controls
            if sample_weight is not None:
                self.mu0_ = np.average(Y_train, weights=sample_weight)
            else:
                self.mu0_ = Y_train.mean()
    
    def predict_ate(self, X, A, Y):
        """
        Predict Average Treatment Effect using causal forest CATE.
        
        Args:
            X: Covariates (used for CATE prediction)
            A: Treatment indicators (unused here, kept for interface consistency)
            Y: Outcomes (unused here, kept for interface consistency)
            
        Returns:
            Scalar ATE estimate (mean CATE)
        """
        if self.cf_model is None:
            raise ValueError("Model not fitted yet")
        
        # econml.grf.CausalForest.predict returns CATE τ(X)
        tau_hat = self.cf_model.predict(X).ravel()
        self.ate_estimate = tau_hat.mean()
        return self.ate_estimate
    
    def predict_outcome(self, X, A):
        """
        Predict outcomes Y given covariates X and treatment A.
        
        We approximate:
            Y ≈ μ0 + τ(X) * A
        
        Args:
            X: Covariates
            A: Treatment indicators
            
        Returns:
            Predicted outcomes
        """
        if self.cf_model is None or self.mu0_ is None:
            raise ValueError("Model not fitted yet")
        
        tau_hat = self.cf_model.predict(X).ravel()
        Y_pred = self.mu0_ + tau_hat * A
        return Y_pred


def compute_exp_ate_cf(X_exp, A_exp, Y_exp, method='difference'):
    """
    Compute ATE estimate from experimental data for CF-based CVCI.
    
    Args:
        X_exp: Covariates
        A_exp: Treatment indicators
        Y_exp: Outcomes
        method: 
            'difference' - simple difference in means (valid under randomization)
            'cf'         - ATE via causal forest fitted on experimental data only
        
    Returns:
        Scalar ATE estimate
    """
    if method == 'difference':
        # Simple difference in means (valid under randomization)
        ate = Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
        return ate
    
    elif method == 'cf':
        # ATE via Causal Forest on experimental data only
        model = CFModel(random_state=random_seed)
        # Fit with lambda_=0.0 so only experimental data is used
        model.fit(X_exp, A_exp, Y_exp, X_exp[:0], A_exp[:0], Y_exp[:0], lambda_=0.0)
        ate = model.predict_ate(X_exp, A_exp, Y_exp)
        return ate
    
    else:
        raise ValueError(f"Unknown method: {method}")


def L_exp_cf(beta_hat, X_exp, A_exp, Y_exp, beta_exp_precompute=None, method='difference'):
    """
    Experimental loss: squared difference from experimental ATE.
    
    Args:
        beta_hat: ATE estimate from our CF model
        X_exp, A_exp, Y_exp: Experimental data
        beta_exp_precompute: Pre-computed experimental ATE (for efficiency)
        method: Method for computing experimental ATE ('difference' or 'cf')
        
    Returns:
        Scalar loss
    """
    if beta_exp_precompute is None:
        beta_exp = compute_exp_ate_cf(X_exp, A_exp, Y_exp, method=method)
    else:
        beta_exp = beta_exp_precompute
    
    return (beta_exp - beta_hat) ** 2


def L_obs_cf(model, X_obs, A_obs, Y_obs):
    """
    Observational loss: MSE of outcome predictions using CF-based reconstruction.
    
    We define:
        Y_pred = μ0 + τ(X) * A
    and use MSE(Y_obs, Y_pred).
    
    Args:
        model: Fitted CFModel
        X_obs, A_obs, Y_obs: Observational data
        
    Returns:
        Scalar loss
    """
    # Predict outcomes
    Y_pred = model.predict_outcome(X_obs, A_obs)
    
    # MSE
    mse = np.mean((Y_obs - Y_pred) ** 2)
    return mse


def cross_validation_cf(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, 
                        lambda_vals, k_fold=5, exp_ate_method='difference', 
                        stratified=True, random_state=None):
    """
    Cross-validation for CVCI with Causal Forests.
    
    Args:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        lambda_vals: Candidate mixing parameters
        k_fold: Number of CV folds
        exp_ate_method: Method for experimental ATE ('difference' or 'cf')
        stratified: Whether to stratify by treatment
        random_state: Random seed
        
    Returns:
        Q_values: CV errors for each lambda
        lambda_opt: Optimal lambda
        model_opt: Fitted CFModel with optimal lambda
    """
    # Set up cross-validator
    if k_fold is None:
        cross_validator = LeaveOneOut()
    else:
        if stratified:
            cross_validator = StratifiedKFold(
                n_splits=k_fold, shuffle=True, random_state=random_state
            )
        else:
            cross_validator = KFold(
                n_splits=k_fold, shuffle=True, random_state=random_state
            )
    
    # Pre-compute experimental ATE for efficiency
    beta_exp = compute_exp_ate_cf(X_exp, A_exp, Y_exp, method=exp_ate_method)
    
    Q_values = np.zeros(len(lambda_vals))
    
    for i, lambda_ in enumerate(lambda_vals):
        #print(f"  Testing lambda = {lambda_:.2f}")
        current_Q = 0
        n_folds = 0
        
        # Cross-validation loop
        if stratified:
            splits = cross_validator.split(X_exp, A_exp)
        else:
            splits = cross_validator.split(X_exp)
        
        for train_idx, val_idx in splits:
            # Split experimental data
            X_train = X_exp[train_idx]
            A_train = A_exp[train_idx]
            Y_train = Y_exp[train_idx]
            
            X_val = X_exp[val_idx]
            A_val = A_exp[val_idx]
            Y_val = Y_exp[val_idx]
            
            # Fit CF model on training exp + all obs
            model = CFModel(random_state=random_state)
            
            try:
                model.fit(
                    X_train, A_train, Y_train, 
                    X_obs, A_obs, Y_obs, lambda_
                )
                
                # Predict ATE on validation set
                beta_hat = model.predict_ate(X_val, A_val, Y_val)
                
                # Compute experimental loss
                loss = L_exp_cf(
                    beta_hat, X_val, A_val, Y_val, 
                    beta_exp_precompute=beta_exp,
                    method=exp_ate_method
                )
                
                current_Q += loss
                n_folds += 1
                
            except Exception as e:
                print(f"    Warning: Fold failed with error: {e}")
                continue
        
        if n_folds > 0:
            Q_values[i] = current_Q / n_folds
        else:
            Q_values[i] = np.inf
    
    # Select optimal lambda
    lambda_opt = lambda_vals[np.argmin(Q_values)]
    print(f"  Optimal lambda: {lambda_opt:.2f}")
    
    # Fit final CF model on all data
    model_opt = CFModel(random_state=random_state)
    model_opt.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_opt)
    
    return Q_values, lambda_opt, model_opt


def run_lalonde_cf(df, group='psid', variables=None, lambda_bin=11, 
                   k_fold=5, n_sims=1):
    """
    Run CVCI with Causal Forests on LaLonde data.
    
    Args:
        df: LaLonde dataframe
        group: Observational comparison group (e.g., 'psid', 'cps')
        variables: List of covariate names (None = no covariates)
        lambda_bin: Number of lambda values to try
        k_fold: Number of CV folds
        n_sims: Number of simulations (for bootstrap)
        
    Returns:
        results: Dictionary with estimates and selected lambdas
    """
    if variables is None:
        variables = []
    
    # Prepare experimental data
    X_df_exp = df[df['group'].isin(['control', 'treated'])]
    if len(variables) > 0:
        X_exp = X_df_exp[variables].to_numpy()
    else:
        X_exp = np.zeros((len(X_df_exp), 1))  # Placeholder
    A_exp = X_df_exp['treatment'].to_numpy()
    Y_exp = X_df_exp['re78'].to_numpy()
    
    # Prepare observational data
    X_df_obs = df[df['group'].isin(['treated', group])]
    if len(variables) > 0:
        X_obs = X_df_obs[variables].to_numpy()
    else:
        X_obs = np.zeros((len(X_df_obs), 1))
    A_obs = X_df_obs['treatment'].to_numpy()
    Y_obs = X_df_obs['re78'].to_numpy()
    
    # Lambda grid
    lambda_vals = np.linspace(0, 1, lambda_bin)
    
    results = {
        'ate_estimates': [],
        'lambda_opts': [],
        'Q_values_all': []
    }
    
    for sim in range(n_sims):
        print(f"Simulation {sim + 1}/{n_sims}")
        
        Q_values, lambda_opt, model_opt = cross_validation_cf(
            X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
            lambda_vals=lambda_vals,
            k_fold=k_fold,
            random_state=random_seed + sim
        )
        
        # Get final ATE estimate
        ate_final = model_opt.predict_ate(X_exp, A_exp, Y_exp)
        
        results['ate_estimates'].append(ate_final)
        results['lambda_opts'].append(lambda_opt)
        results['Q_values_all'].append(Q_values)
    
    return results


if __name__ == '__main__':
    # Example usage
    print("Testing CVCI with Causal Forests on LaLonde data")
    
    # Load data
    df = pd.read_csv('lalonde.csv')
    df['age2'] = df['age'] ** 2
    
    # Test with simple variable set
    variables = ['age', 'education', 're75']
    
    results = run_lalonde_cf(
        df, 
        group='psid',
        variables=variables,
        lambda_bin=6,  # Coarse grid for testing
        k_fold=3,
        n_sims=1
    )
    
    print("\nResults:")
    print(f"ATE estimate: {results['ate_estimates'][0]:.2f}")
    print(f"Optimal lambda: {results['lambda_opts'][0]:.2f}")
    print(f"CV errors: {results['Q_values_all'][0]}")
