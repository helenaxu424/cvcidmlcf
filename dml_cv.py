"""
CVCI with Double/Debiased Machine Learning (DML)

Extends the Yang et al. (2025) CVCI framework to use DML
instead of linear models.

Uses Random Forest for nuisance parameter estimation (no hyperparameter tuning needed).

Key differences from linear CVCI:
- Model class θ includes nuisance parameters (propensity scores, outcome models)
- Uses doubly-robust (AIPW) scores for ATE estimation
- Loss functions remain MSE-based
- Hybrid loss implemented via sample weights (corrected)
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd

random_seed = 2024
np.random.seed(random_seed)


class DMLModel:
    """
    Double/Debiased Machine Learning model for CVCI framework.
    
    Estimates ATE using doubly-robust scores with ML-based nuisance parameters.
    """
    
    def __init__(self, propensity_model='rf', outcome_model='rf', random_state=None):
        """
        Args:
            propensity_model: Model for propensity scores ('rf' or 'logistic')
            outcome_model: Model for outcome regression ('rf' or 'linear')
            random_state: Random seed
        """
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.random_state = random_state
        
        # Models will be fit during training
        self.e_model = None  # Propensity score model
        self.mu0_model = None  # Outcome model for controls
        self.mu1_model = None  # Outcome model for treated
        
        self.ate_estimate = None
    
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """
        Fit DML model using hybrid loss.
        
        Strategy:
        - Combine experimental and observational data with weights
        - Fit nuisance parameters (propensity scores, outcome models)
        - Compute ATE via doubly-robust scores
        
        Args:
            X_exp, A_exp, Y_exp: Experimental covariates, treatment, outcome
            X_obs, A_obs, Y_obs: Observational covariates, treatment, outcome
            lambda_: Mixing parameter
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
        
        # Fit propensity score model
        if self.propensity_model == 'rf':
            self.e_model = RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        elif self.propensity_model == 'logistic':
            self.e_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown propensity model: {self.propensity_model}")
        
        if sample_weight is not None:
            self.e_model.fit(X_train, A_train, sample_weight=sample_weight)
        else:
            self.e_model.fit(X_train, A_train)
        
        # Fit outcome models
        if self.outcome_model == 'rf':
            self.mu0_model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
            self.mu1_model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        else:
            self.mu0_model = LinearRegression()
            self.mu1_model = LinearRegression()
        
        # Fit on controls
        X_0 = X_train[A_train == 0]
        Y_0 = Y_train[A_train == 0]
        if len(X_0) > 0:
            if sample_weight is not None:
                w_0 = sample_weight[A_train == 0]
                self.mu0_model.fit(X_0, Y_0, sample_weight=w_0)
            else:
                self.mu0_model.fit(X_0, Y_0)
        
        # Fit on treated
        X_1 = X_train[A_train == 1]
        Y_1 = Y_train[A_train == 1]
        if len(X_1) > 0:
            if sample_weight is not None:
                w_1 = sample_weight[A_train == 1]
                self.mu1_model.fit(X_1, Y_1, sample_weight=w_1)
            else:
                self.mu1_model.fit(X_1, Y_1)
    
    def predict_ate(self, X, A, Y):
        """
        Predict Average Treatment Effect using doubly-robust scores.
        
        Args:
            X: Covariates
            A: Treatment indicators
            Y: Outcomes
            
        Returns:
            Scalar ATE estimate
        """
        if self.e_model is None:
            raise ValueError("Model not fitted yet")
        
        # Get propensity scores
        e_hat = self.e_model.predict_proba(X)[:, 1]
        # Clip for numerical stability
        e_hat = np.clip(e_hat, 0.01, 0.99)
        
        # Get outcome predictions
        mu0_hat = self.mu0_model.predict(X)
        mu1_hat = self.mu1_model.predict(X)
        
        # Compute doubly-robust scores
        psi = (A / e_hat) * (Y - mu1_hat) + mu1_hat - \
              ((1 - A) / (1 - e_hat)) * (Y - mu0_hat) - mu0_hat
        
        self.ate_estimate = np.mean(psi)
        return self.ate_estimate
    
    def predict_outcome(self, X, A):
        """
        Predict outcomes Y given covariates X and treatment A.
        
        This is needed for computing L_obs.
        
        Args:
            X: Covariates
            A: Treatment indicators
            
        Returns:
            Predicted outcomes
        """
        if self.e_model is None:
            raise ValueError("Model not fitted yet")
        
        # Predict outcomes based on treatment status
        mu0_hat = self.mu0_model.predict(X)
        mu1_hat = self.mu1_model.predict(X)
        
        # Weighted average based on treatment
        Y_pred = (1 - A) * mu0_hat + A * mu1_hat
        
        return Y_pred


def compute_exp_ate_dml(X_exp, A_exp, Y_exp, method='difference'):
    """
    Compute ATE estimate from experimental data.
    
    Args:
        X_exp: Covariates
        A_exp: Treatment indicators
        Y_exp: Outcomes
        method: 'difference' for simple mean difference, 'aipw' for doubly robust
        
    Returns:
        Scalar ATE estimate
    """
    if method == 'difference':
        # Simple difference in means (valid under randomization)
        ate = Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
        return ate
    
    elif method == 'aipw':
        # AIPW estimator using DML
        model = DMLModel(propensity_model='logistic', outcome_model='rf')
        # Fit on experimental data only
        model.fit(X_exp, A_exp, Y_exp, X_exp[:0], A_exp[:0], Y_exp[:0], lambda_=0.0)
        ate = model.predict_ate(X_exp, A_exp, Y_exp)
        return ate
    
    else:
        raise ValueError(f"Unknown method: {method}")


def L_exp_dml(beta_hat, X_exp, A_exp, Y_exp, beta_exp_precompute=None, method='difference'):
    """
    Experimental loss: squared difference from experimental ATE.
    
    Args:
        beta_hat: ATE estimate from our model
        X_exp, A_exp, Y_exp: Experimental data
        beta_exp_precompute: Pre-computed experimental ATE (for efficiency)
        method: Method for computing experimental ATE
        
    Returns:
        Scalar loss
    """
    if beta_exp_precompute is None:
        beta_exp = compute_exp_ate_dml(X_exp, A_exp, Y_exp, method=method)
    else:
        beta_exp = beta_exp_precompute
    
    return (beta_exp - beta_hat) ** 2


def L_obs_dml(model, X_obs, A_obs, Y_obs):
    """
    Observational loss: MSE of outcome predictions.
    
    Args:
        model: Fitted DMLModel
        X_obs, A_obs, Y_obs: Observational data
        
    Returns:
        Scalar loss
    """
    # Predict outcomes
    Y_pred = model.predict_outcome(X_obs, A_obs)
    
    # MSE
    mse = np.mean((Y_obs - Y_pred) ** 2)
    return mse


def cross_validation_dml(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, 
                        lambda_vals, k_fold=5, exp_ate_method='difference', 
                        stratified=True, random_state=None):
    """
    Cross-validation for CVCI with DML.
    
    Args:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        lambda_vals: Candidate mixing parameters
        k_fold: Number of CV folds
        exp_ate_method: Method for experimental ATE ('difference' or 'aipw')
        stratified: Whether to stratify by treatment
        random_state: Random seed
        
    Returns:
        Q_values: CV errors for each lambda
        lambda_opt: Optimal lambda
        model_opt: Fitted model with optimal lambda
    """
    # Set up cross-validator
    if k_fold is None:
        cross_validator = LeaveOneOut()
    else:
        if stratified:
            cross_validator = StratifiedKFold(n_splits=k_fold, shuffle=True, 
                                             random_state=random_state)
        else:
            cross_validator = KFold(n_splits=k_fold, shuffle=True, 
                                   random_state=random_state)
    
    # Pre-compute experimental ATE for efficiency
    beta_exp = compute_exp_ate_dml(X_exp, A_exp, Y_exp, method=exp_ate_method)
    
    Q_values = np.zeros(len(lambda_vals))
    
    for i, lambda_ in enumerate(lambda_vals):
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
            
            # Fit model on training exp + all obs
            model = DMLModel(random_state=random_state)
            
            try:
                model.fit(X_train, A_train, Y_train, 
                         X_obs, A_obs, Y_obs, lambda_)
                
                # Predict ATE on validation set
                beta_hat = model.predict_ate(X_val, A_val, Y_val)
                
                # Compute experimental loss
                loss = L_exp_dml(beta_hat, X_val, A_val, Y_val, 
                                beta_exp_precompute=beta_exp,
                                method=exp_ate_method)
                
                current_Q += loss
                n_folds += 1
                
            except Exception as e:
                # Skip this fold if it fails
                continue
        
        if n_folds > 0:
            Q_values[i] = current_Q / n_folds
        else:
            Q_values[i] = np.inf
    
    # Select optimal lambda
    lambda_opt = lambda_vals[np.argmin(Q_values)]
    
    # Fit final model on all data
    model_opt = DMLModel(random_state=random_state)
    model_opt.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_opt)
    
    return Q_values, lambda_opt, model_opt


if __name__ == '__main__':
    # Example usage
    print("Testing CVCI with DML on LaLonde data")
    
    # Load data
    df = pd.read_csv('lalonde.csv')
    df['age2'] = df['age'] ** 2
    
    # Test with simple variable set
    variables = ['age', 'education', 're75']
    
    # Prepare experimental data
    X_df_exp = df[df['group'].isin(['control', 'treated'])]
    X_exp = X_df_exp[variables].to_numpy()
    A_exp = X_df_exp['treatment'].to_numpy()
    Y_exp = X_df_exp['re78'].to_numpy()
    
    # Prepare observational data
    X_df_obs = df[df['group'].isin(['treated', 'psid'])]
    X_obs = X_df_obs[variables].to_numpy()
    A_obs = X_df_obs['treatment'].to_numpy()
    Y_obs = X_df_obs['re78'].to_numpy()
    
    # Run CV
    lambda_vals = np.linspace(0, 1, 6)
    Q_values, lambda_opt, model_opt = cross_validation_dml(
        X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
        lambda_vals=lambda_vals,
        k_fold=3,
        random_state=random_seed
    )
    
    # Get final ATE
    ate_final = model_opt.predict_ate(X_exp, A_exp, Y_exp)
    
    print("\nResults:")
    print(f"ATE estimate: {ate_final:.2f}")
    print(f"Optimal lambda: {lambda_opt:.2f}")
    print(f"CV errors: {Q_values}")



class DMLModel:
    """
    Double/Debiased Machine Learning model for CVCI framework.
    
    Estimates ATE using doubly-robust scores with ML-based nuisance parameters.
    """
    
    def __init__(self, propensity_model='rf', outcome_model='rf', random_state=None):
        """
        Args:
            propensity_model: Model for propensity scores ('rf' or 'logistic')
            outcome_model: Model for outcome regression ('rf' or 'linear')
            random_state: Random seed
        """
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.random_state = random_state
        
        # Models will be fit during training
        self.e_model = None  # Propensity score model
        self.mu0_model = None  # Outcome model for controls
        self.mu1_model = None  # Outcome model for treated
        
        self.ate_estimate = None
    
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """
        Fit DML model using hybrid loss.
        
        Strategy:
        - Combine experimental and observational data with weights
        - Fit nuisance parameters (propensity scores, outcome models)
        - Compute ATE via doubly-robust scores
        
        Args:
            X_exp, A_exp, Y_exp: Experimental covariates, treatment, outcome
            X_obs, A_obs, Y_obs: Observational covariates, treatment, outcome
            lambda_: Mixing parameter
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
            # Scale so sum of weights = n_total for sklearn compatibility
            n_exp = len(X_exp)
            n_obs = len(X_obs)
            n_total = n_exp + n_obs
            
            # Weight each experimental sample by (1-λ) scaled by dataset ratio
            weights_exp = np.ones(n_exp) * (1 - lambda_) * n_total / n_exp
            
            # Weight each observational sample by λ scaled by dataset ratio  
            weights_obs = np.ones(n_obs) * lambda_ * n_total / n_obs
            
            sample_weight = np.concatenate([weights_exp, weights_obs])
        
        # Fit propensity score model
        if self.propensity_model == 'rf':
            self.e_model = RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        elif self.propensity_model == 'logistic':
            self.e_model = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown propensity model: {self.propensity_model}")
        
        if sample_weight is not None:
            self.e_model.fit(X_train, A_train, sample_weight=sample_weight)
        else:
            self.e_model.fit(X_train, A_train)
        
        # Fit outcome models
        if self.outcome_model == 'rf':
            self.mu0_model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
            self.mu1_model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        else:
            from sklearn.linear_model import LinearRegression
            self.mu0_model = LinearRegression()
            self.mu1_model = LinearRegression()
        
        # Fit on controls
        X_0 = X_train[A_train == 0]
        Y_0 = Y_train[A_train == 0]
        if len(X_0) > 0:
            if sample_weight is not None:
                w_0 = sample_weight[A_train == 0]
                self.mu0_model.fit(X_0, Y_0, sample_weight=w_0)
            else:
                self.mu0_model.fit(X_0, Y_0)
        
        # Fit on treated
        X_1 = X_train[A_train == 1]
        Y_1 = Y_train[A_train == 1]
        if len(X_1) > 0:
            if sample_weight is not None:
                w_1 = sample_weight[A_train == 1]
                self.mu1_model.fit(X_1, Y_1, sample_weight=w_1)
            else:
                self.mu1_model.fit(X_1, Y_1)
    
    def predict_ate(self, X, A, Y):
        """
        Predict Average Treatment Effect using doubly-robust scores.
        
        Args:
            X: Covariates
            A: Treatment indicators
            Y: Outcomes
            
        Returns:
            Scalar ATE estimate
        """
        if self.e_model is None:
            raise ValueError("Model not fitted yet")
        
        # Get propensity scores
        e_hat = self.e_model.predict_proba(X)[:, 1]
        # Clip for numerical stability
        e_hat = np.clip(e_hat, 0.01, 0.99)
        
        # Get outcome predictions
        mu0_hat = self.mu0_model.predict(X)
        mu1_hat = self.mu1_model.predict(X)
        
        # Compute doubly-robust scores
        psi = (A / e_hat) * (Y - mu1_hat) + mu1_hat - \
              ((1 - A) / (1 - e_hat)) * (Y - mu0_hat) - mu0_hat
        
        self.ate_estimate = np.mean(psi)
        return self.ate_estimate
    
    def predict_outcome(self, X, A):
        """
        Predict outcomes Y given covariates X and treatment A.
        
        This is needed for computing L_obs.
        
        Args:
            X: Covariates
            A: Treatment indicators
            
        Returns:
            Predicted outcomes
        """
        if self.e_model is None:
            raise ValueError("Model not fitted yet")
        
        # Predict outcomes based on treatment status
        mu0_hat = self.mu0_model.predict(X)
        mu1_hat = self.mu1_model.predict(X)
        
        # Weighted average based on treatment
        Y_pred = (1 - A) * mu0_hat + A * mu1_hat
        
        return Y_pred


def compute_exp_ate_dml(X_exp, A_exp, Y_exp, method='difference'):
    """
    Compute ATE estimate from experimental data.
    
    Args:
        X_exp: Covariates
        A_exp: Treatment indicators
        Y_exp: Outcomes
        method: 'difference' for simple mean difference, 'aipw' for doubly robust
        
    Returns:
        Scalar ATE estimate
    """
    if method == 'difference':
        # Simple difference in means (valid under randomization)
        ate = Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
        return ate
    
    elif method == 'aipw':
        # AIPW estimator using DML
        model = DMLModel(propensity_model='logistic', outcome_model='rf')
        # Fit on experimental data only
        model.fit(X_exp, A_exp, Y_exp, X_exp[:0], A_exp[:0], Y_exp[:0], lambda_=0.0)
        ate = model.predict_ate(X_exp, A_exp, Y_exp)
        return ate
    
    else:
        raise ValueError(f"Unknown method: {method}")


def L_exp_dml(beta_hat, X_exp, A_exp, Y_exp, beta_exp_precompute=None, method='difference'):
    """
    Experimental loss: squared difference from experimental ATE.
    
    Args:
        beta_hat: ATE estimate from our model
        X_exp, A_exp, Y_exp: Experimental data
        beta_exp_precompute: Pre-computed experimental ATE (for efficiency)
        method: Method for computing experimental ATE
        
    Returns:
        Scalar loss
    """
    if beta_exp_precompute is None:
        beta_exp = compute_exp_ate_dml(X_exp, A_exp, Y_exp, method=method)
    else:
        beta_exp = beta_exp_precompute
    
    return (beta_exp - beta_hat) ** 2


def L_obs_dml(model, X_obs, A_obs, Y_obs):
    """
    Observational loss: MSE of outcome predictions.
    
    Args:
        model: Fitted DMLModel
        X_obs, A_obs, Y_obs: Observational data
        
    Returns:
        Scalar loss
    """
    # Predict outcomes
    Y_pred = model.predict_outcome(X_obs, A_obs)
    
    # MSE
    mse = np.mean((Y_obs - Y_pred) ** 2)
    return mse


def cross_validation_dml(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, 
                        lambda_vals, k_fold=5, exp_ate_method='difference', 
                        stratified=True, random_state=None):
    """
    Cross-validation for CVCI with DML.
    
    Args:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        lambda_vals: Candidate mixing parameters
        k_fold: Number of CV folds
        exp_ate_method: Method for experimental ATE ('difference' or 'aipw')
        stratified: Whether to stratify by treatment
        random_state: Random seed
        
    Returns:
        Q_values: CV errors for each lambda
        lambda_opt: Optimal lambda
        model_opt: Fitted model with optimal lambda
    """
    # Set up cross-validator
    if k_fold is None:
        cross_validator = LeaveOneOut()
    else:
        if stratified:
            cross_validator = StratifiedKFold(n_splits=k_fold, shuffle=True, 
                                             random_state=random_state)
        else:
            cross_validator = KFold(n_splits=k_fold, shuffle=True, 
                                   random_state=random_state)
    
    # Pre-compute experimental ATE for efficiency
    beta_exp = compute_exp_ate_dml(X_exp, A_exp, Y_exp, method=exp_ate_method)
    
    Q_values = np.zeros(len(lambda_vals))
    
    for i, lambda_ in enumerate(lambda_vals):
        print(f"  Testing lambda = {lambda_:.2f}")
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
            
            # Fit model on training exp + all obs
            model = DMLModel(random_state=random_state)
            
            try:
                model.fit(X_train, A_train, Y_train, 
                         X_obs, A_obs, Y_obs, lambda_)
                
                # Predict ATE on validation set
                beta_hat = model.predict_ate(X_val, A_val, Y_val)
                
                # Compute experimental loss
                loss = L_exp_dml(beta_hat, X_val, A_val, Y_val, 
                                beta_exp_precompute=beta_exp,
                                method=exp_ate_method)
                
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
    
    # Fit final model on all data
    model_opt = DMLModel(random_state=random_state)
    model_opt.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_opt)
    
    return Q_values, lambda_opt, model_opt


def run_lalonde_dml(df, group='psid', variables=None, lambda_bin=11, 
                   k_fold=5, n_sims=1):
    """
    Run CVCI with DML on LaLonde data.
    
    Args:
        df: LaLonde dataframe
        group: Observational comparison group
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
        
        Q_values, lambda_opt, model_opt = cross_validation_dml(
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
    print("Testing CVCI with DML on LaLonde data")
    
    # Load data
    df = pd.read_csv('lalonde.csv')
    df['age2'] = df['age'] ** 2
    
    # Test with simple variable set
    variables = ['age', 'education', 're75']
    
    results = run_lalonde_dml(
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