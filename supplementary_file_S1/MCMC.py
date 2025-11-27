import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import norm
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 24

# Bayesian Gibbs Sampling Implementation with MULTIPLE CHAINS
class BayesianLogisticRegressionMultiChainCorrected:
    def __init__(self, n_chains=5, n_iter=1050000, burn_in=50000, beta_prior_mean=0, beta_prior_precision=0.01):
        self.n_chains = n_chains
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.beta_prior_mean = beta_prior_mean
        self.beta_prior_precision = beta_prior_precision
        
    def sigmoid(self, x):
        """Sigmoid function for logistic regression"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def fit_single_chain(self, X, y, initial_beta=None, chain_id=0):
        """Fit a single MCMC chain with adaptive proposal distribution"""
        n, p = X.shape
        
        # Initialize parameters with different starting points for each chain
        if initial_beta is None:
            # More reasonable starting points
            if chain_id == 0:
                beta = np.random.normal(20, 20, p)  
            elif chain_id == 1:
                beta = np.random.normal(-20, 20, p)  
            elif chain_id == 2:
                beta = np.random.normal(-50, 10, p)  
            else:
                beta = np.random.normal(0, 100, p) 
        else:
            beta = initial_beta.copy()
        
        # Adaptive proposal variances
        proposal_sd = np.ones(p) * 0.1
        acceptance_rates = np.zeros(p)
        
        beta_chain = np.zeros((self.n_iter, p))
        log_likelihood_chain = np.zeros(self.n_iter)
        
        print(f"  Chain {chain_id + 1}: Starting with beta = {beta[:]}...")
        
        for i in range(self.n_iter):
            # Adaptive MH for each parameter
            for j in range(p):
                # Propose new value
                beta_proposal = beta.copy()
                beta_proposal[j] = np.random.normal(beta[j], proposal_sd[j])
                
                # Calculate log-likelihood
                z_current = X @ beta
                z_proposal = X @ beta_proposal
                
                # More numerically stable log-likelihood
                ll_current = np.sum(y * z_current - np.logaddexp(0, z_current))
                ll_proposal = np.sum(y * z_proposal - np.logaddexp(0, z_proposal))
                
                # Priors (normal)
                prior_current = -0.5 * self.beta_prior_precision * np.sum((beta - self.beta_prior_mean)**2)
                prior_proposal = -0.5 * self.beta_prior_precision * np.sum((beta_proposal - self.beta_prior_mean)**2)
                
                # Log posterior
                log_posterior_current = ll_current + prior_current
                log_posterior_proposal = ll_proposal + prior_proposal
                
                # Acceptance ratio
                log_accept_ratio = log_posterior_proposal - log_posterior_current
                
                # Accept or reject
                if np.log(np.random.uniform()) < min(0, log_accept_ratio):
                    beta = beta_proposal.copy()
                    acceptance_rates[j] += 1
            
            # Store samples
            beta_chain[i] = beta
            log_likelihood_chain[i] = ll_current
            
            # Adapt proposal distribution every 100 iterations
            if i % 100 == 0 and i > 100:
                for j in range(p):
                    current_rate = acceptance_rates[j] / 100
                    if current_rate < 0.15:
                        proposal_sd[j] *= 0.9
                    elif current_rate > 0.35:
                        proposal_sd[j] *= 1.1
                acceptance_rates = np.zeros(p)
        
        return beta_chain, log_likelihood_chain
        
    def fit(self, X, y):
        n, p = X.shape
        self.p = p
        
        print(f"Running {self.n_chains} independent MCMC chains for Bayesian Logistic Regression...")
        print("Using ORIGINAL DATA SCALE with adaptive MCMC")
        
        # Run multiple independent chains
        self.chains_beta = []
        self.chains_ll = []
        
        for chain_id in range(self.n_chains):
            print(f"Running chain {chain_id + 1}/{self.n_chains}...")
            beta_chain, ll_chain = self.fit_single_chain(X, y, chain_id=chain_id)
            self.chains_beta.append(beta_chain)
            self.chains_ll.append(ll_chain)
        
        # More conservative burn-in
        self.beta_post = np.vstack([chain[self.burn_in:] for chain in self.chains_beta])
        self.ll_post = np.hstack([chain[self.burn_in:] for chain in self.chains_ll])
        
        # Compute posterior statistics
        self.beta_mean = np.mean(self.beta_post, axis=0)
        self.beta_median = np.median(self.beta_post, axis=0)
        self.beta_std = np.std(self.beta_post, axis=0)
        
        # Improved convergence diagnostics
        self.r_hat_beta = self.gelman_rubin_improved(self.chains_beta)
        
        return self
    
    def gelman_rubin_improved(self, chains):
        """Improved Gelman-Rubin diagnostic with better convergence assessment"""
        n_chains = len(chains)
        n_iter = chains[0].shape[0]
        p = chains[0].shape[1]
        
        # Use only post-burn-in samples
        chains_post = [chain[self.burn_in:] for chain in chains]
        n_post = chains_post[0].shape[0]
        
        r_hat = np.zeros(p)
        
        for j in range(p):
            # Between-chain variance
            chain_means = np.array([np.mean(chain[:, j]) for chain in chains_post])
            overall_mean = np.mean(chain_means)
            B = (n_post / (n_chains - 1)) * np.sum((chain_means - overall_mean)**2)
            
            # Within-chain variance
            W = np.mean([np.var(chain[:, j], ddof=1) for chain in chains_post])
            
            # Marginal posterior variance
            var_plus = ((n_post - 1) / n_post) * W + (1 / n_post) * B
            
            # R-hat statistic
            r_hat[j] = np.sqrt(var_plus / W)
            
            # Additional check: if chains have very different means
            chain_std = np.std(chain_means)
            if chain_std > 2 * np.sqrt(W):  # Chains are too dispersed
                r_hat[j] = max(r_hat[j], 1.5)  # Force high R-hat
        
        return r_hat
    
    def predict_proba(self, X):
        """Predict probabilities using posterior mean"""
        linear_predictor = X @ self.beta_mean
        return self.sigmoid(linear_predictor)
    
    def predict(self, X, threshold=0.5):
        """Predict classes using posterior mean"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def hpd(self, chain, alpha=0.05):
        """Compute Highest Posterior Density interval"""
        sorted_chain = np.sort(chain)
        n = len(sorted_chain)
        m = int(alpha * n)
        
        if m == 0:
            return sorted_chain[0], sorted_chain[-1]
        
        intervals = sorted_chain[m:] - sorted_chain[:-m]
        min_idx = np.argmin(intervals)
        
        return sorted_chain[min_idx], sorted_chain[min_idx + m]
    
    def get_posterior_predictive(self, X):
        """Get posterior predictive distribution for each sample"""
        n_samples = len(X)
        n_post = len(self.beta_post)
        
        posterior_predictive = np.zeros((n_post, n_samples))
        
        for i in range(n_post):
            linear_predictor = X @ self.beta_post[i]
            posterior_predictive[i] = self.sigmoid(linear_predictor)
        
        return posterior_predictive


def calculate_confidence_intervals_corrected(model, X, y, alpha=0.05):
    """Calculate 95% confidence intervals for logistic regression coefficients"""
    try:
        n = X.shape[0]
        
        # Get coefficients including intercept
        if hasattr(model, 'intercept_'):
            coefficients = np.concatenate([[model.intercept_[0]], model.coef_[0]])
        else:
            coefficients = np.concatenate([[0], model.coef_[0]])
        
        # Try standard method first
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)[:, 1]
            else:
                p = model.predict(X)
            
            # Calculate Fisher information matrix
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            W = np.diag(p * (1 - p))
            fisher_info = X_design.T @ W @ X_design
            
            # Calculate standard errors
            cov_matrix = np.linalg.pinv(fisher_info)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # Calculate confidence intervals
            z_value = norm.ppf(1 - alpha/2)
            ci_lower = coefficients - z_value * std_errors
            ci_upper = coefficients + z_value * std_errors
            
            # Check for any NaN values
            if np.any(np.isnan(std_errors)) or np.any(np.isnan(ci_lower)) or np.any(np.isnan(ci_upper)):
                raise np.linalg.LinAlgError("NaN values in confidence intervals")
                
        except (np.linalg.LinAlgError, ValueError):
            # Use bootstrap for problematic cases
            print("  Using bootstrap for confidence intervals (standard method failed)")
            n_bootstrap = 500
            bootstrap_coefs = []
            
            for _ in range(n_bootstrap):
                try:
                    # Bootstrap sample
                    indices = np.random.choice(n, n, replace=True)
                    X_boot = X[indices]
                    y_boot = y[indices]
                    
                    # Fit model on bootstrap sample
                    model_boot = LogisticRegression(random_state=42, max_iter=1000)
                    model_boot.fit(X_boot, y_boot)
                    
                    # Store coefficients
                    if hasattr(model_boot, 'intercept_'):
                        coef_boot = np.concatenate([[model_boot.intercept_[0]], model_boot.coef_[0]])
                    else:
                        coef_boot = np.concatenate([[0], model_boot.coef_[0]])
                    
                    bootstrap_coefs.append(coef_boot)
                except:
                    continue
            
            if len(bootstrap_coefs) > 10:
                bootstrap_coefs = np.array(bootstrap_coefs)
                # Calculate percentile intervals
                ci_lower = np.percentile(bootstrap_coefs, (alpha/2)*100, axis=0)
                ci_upper = np.percentile(bootstrap_coefs, (1-alpha/2)*100, axis=0)
                std_errors = np.std(bootstrap_coefs, axis=0)
            else:
                # Fallback: use large standard errors
                std_errors = np.ones(len(coefficients)) * 5.0
                z_value = norm.ppf(1 - alpha/2)
                ci_lower = coefficients - z_value * std_errors
                ci_upper = coefficients + z_value * std_errors
        
        return coefficients, std_errors, ci_lower, ci_upper
        
    except Exception as e:
        print(f"Error calculating confidence intervals: {e}")
        n_params = len(model.coef_[0]) + 1
        return np.zeros(n_params), np.ones(n_params) * np.nan, np.zeros(n_params), np.ones(n_params) * np.nan


def evaluate_model(X_train, X_test, y_train, y_test, model, model_name, use_scaling=False):
    """Evaluate model performance and return metrics for both train and test"""
    try:
        # No scaling applied (as requested)
        X_train_processed = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_processed = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Predictions on test set
        y_pred_test = model.predict(X_test_processed)
        y_pred_proba_test = model.predict_proba(X_test_processed)[:, 1]
        
        # Predictions on train set
        y_pred_train = model.predict(X_train_processed)
        y_pred_proba_train = model.predict_proba(X_train_processed)[:, 1]
        
        # Metrics for test set
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)
        
        # Handle ROC curve for binary labels - convert to numeric if needed
        y_test_numeric = y_test
        if y_test.dtype == object or y_test.dtype == 'category':
            y_test_numeric = pd.Categorical(y_test).codes
        
        fpr_test, tpr_test, _ = roc_curve(y_test_numeric, y_pred_proba_test)
        roc_auc_test = auc(fpr_test, tpr_test)
        
        # Metrics for train set
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, zero_division=0)
        f1_train = f1_score(y_train, y_pred_train, zero_division=0)
        
        # Handle ROC curve for binary labels - convert to numeric if needed
        y_train_numeric = y_train
        if y_train.dtype == object or y_train.dtype == 'category':
            y_train_numeric = pd.Categorical(y_train).codes
            
        fpr_train, tpr_train, _ = roc_curve(y_train_numeric, y_pred_proba_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        
        # Calculate differences between train and test (to minimize)
        roc_auc_diff = abs(roc_auc_train - roc_auc_test)
        accuracy_diff = abs(accuracy_train - accuracy_test)
        
        # Calculate composite score
        composite_score = (roc_auc_diff * 0.7 + accuracy_diff * 0.3)
        
        # Get feature coefficients
        coefficients = model.coef_[0] if hasattr(model, 'coef_') else None
        
        return {
            'model': model,
            'accuracy_test': accuracy_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'roc_auc_test': roc_auc_test,
            'fpr_test': fpr_test,
            'tpr_test': tpr_test,
            'y_pred_test': y_pred_test,
            'y_pred_proba_test': y_pred_proba_test,
            'accuracy_train': accuracy_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'roc_auc_train': roc_auc_train,
            'fpr_train': fpr_train,
            'tpr_train': tpr_train,
            'y_pred_train': y_pred_train,  # ADD THIS LINE
            'y_pred_proba_train': y_pred_proba_train,  # ADD THIS LINE
            'roc_auc_diff': roc_auc_diff,
            'accuracy_diff': accuracy_diff,
            'composite_score': composite_score,
            'coefficients': coefficients,
            'use_scaling': use_scaling,
            'success': True
        }
    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return {'success': False}


def plot_markov_chains(bayesian_model, feature_names, base_dir):
    """Plot Markov chains for each coefficient"""
    print("Plotting Markov chains...")
    
    # Create directory for Markov chains
    chains_dir = f'{base_dir}/figures/markov_chains'
    os.makedirs(chains_dir, exist_ok=True)
    
    n_chains = len(bayesian_model.chains_beta)
    n_features = len(feature_names)
    
    # Plot each chain separately for each parameter
    for feature_idx, feature_name in enumerate(feature_names):
        plt.figure(figsize=(15, 10))
        
        # Plot each chain
        for chain_idx in range(n_chains):
            chain_data = bayesian_model.chains_beta[chain_idx][:, feature_idx]
            # Plot every 100th point to make it manageable
            plt.plot(chain_data[::100], alpha=0.7, label=f'Chain {chain_idx+1}')
        
        plt.xlabel('Iteration (every 100th)')
        plt.ylabel(f'Coefficient Value')
        plt.title(f'Markov Chains for {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save individual figure
        plt.savefig(f'{chains_dir}/markov_chain_{feature_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{chains_dir}/markov_chain_{feature_name}.svg', bbox_inches='tight')
        plt.close()
    
    print(f"Markov chains saved to {chains_dir}/")


def plot_posterior_distributions(bayesian_model, feature_names, base_dir):
    """Plot posterior distributions for each coefficient"""
    print("Plotting posterior distributions...")
    
    # Create directory for posterior distributions
    posterior_dir = f'{base_dir}/figures/posterior_distributions'
    os.makedirs(posterior_dir, exist_ok=True)
    
    n_features = len(feature_names)
    
    # Plot individual posterior distributions
    for feature_idx, feature_name in enumerate(feature_names):
        plt.figure(figsize=(12, 8))
        
        # Get posterior samples for this feature
        posterior_samples = bayesian_model.beta_post[:, feature_idx]
        
        # Plot histogram with KDE
        plt.hist(posterior_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(posterior_samples)
        x_vals = np.linspace(posterior_samples.min(), posterior_samples.max(), 1000)
        plt.plot(x_vals, kde(x_vals), 'r-', linewidth=2)
        
        # Add vertical lines for mean and HPD
        mean_val = np.mean(posterior_samples)
        hpd_lower, hpd_upper = bayesian_model.hpd(posterior_samples)
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        plt.axvline(hpd_lower, color='orange', linestyle=':', linewidth=2, label=f'HPD Lower: {hpd_lower:.4f}')
        plt.axvline(hpd_upper, color='orange', linestyle=':', linewidth=2, label=f'HPD Upper: {hpd_upper:.4f}')
        
        plt.xlabel(f'Coefficient Value')
        plt.ylabel('Density')
        plt.title(f'Posterior Distribution for {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save individual figure
        plt.savefig(f'{posterior_dir}/posterior_{feature_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{posterior_dir}/posterior_{feature_name}.svg', bbox_inches='tight')
        plt.close()
    
    print(f"Posterior distributions saved to {posterior_dir}/")


def calculate_probability_intervals(classical_model, bayesian_model, X_data, feature_names, base_dir):
    """Calculate classical and Bayesian probability estimates with intervals"""
    print("Calculating probability estimates and intervals...")
    
    # Prepare data with intercept
    X_with_intercept = np.column_stack([np.ones(len(X_data)), X_data])
    
    # Classical probability estimates (point estimates)
    classical_probs = classical_model.predict_proba(X_data)[:, 1]
    
    # Bayesian probability estimates (posterior mean)
    bayesian_probs_mean = bayesian_model.predict_proba(X_with_intercept)
    
    # Bayesian posterior predictive distribution
    posterior_predictive = bayesian_model.get_posterior_predictive(X_with_intercept)
    
    # Calculate HPD intervals for each observation
    bayesian_hpd_lower = np.zeros(len(X_data))
    bayesian_hpd_upper = np.zeros(len(X_data))
    
    for i in range(len(X_data)):
        hpd_lower, hpd_upper = bayesian_model.hpd(posterior_predictive[:, i])
        bayesian_hpd_lower[i] = hpd_lower
        bayesian_hpd_upper[i] = hpd_upper
    
    # Calculate classical confidence intervals using bootstrap
    n_bootstrap = 1000
    classical_bootstrap_probs = np.zeros((n_bootstrap, len(X_data)))
    
    # Bootstrap for classical confidence intervals
    for b in range(n_bootstrap):
        try:
            # Create bootstrap sample
            indices = np.random.choice(len(X_data), len(X_data), replace=True)
            X_boot = X_data.iloc[indices] if hasattr(X_data, 'iloc') else X_data[indices]
            
            # Fit model on bootstrap sample
            model_boot = LogisticRegression(random_state=42+b, max_iter=1000)
            model_boot.fit(X_boot, classical_model.predict(X_boot))  # Using same predictions for simplicity
            
            # Store probabilities
            classical_bootstrap_probs[b] = model_boot.predict_proba(X_data)[:, 1]
        except:
            classical_bootstrap_probs[b] = classical_probs
    
    # Calculate classical confidence intervals
    classical_ci_lower = np.percentile(classical_bootstrap_probs, 2.5, axis=0)
    classical_ci_upper = np.percentile(classical_bootstrap_probs, 97.5, axis=0)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'classical_probability': classical_probs,
        'classical_ci_lower': classical_ci_lower,
        'classical_ci_upper': classical_ci_upper,
        'classical_ci_amplitude': classical_ci_upper - classical_ci_lower,
        'bayesian_probability_mean': bayesian_probs_mean,
        'bayesian_hpd_lower': bayesian_hpd_lower,
        'bayesian_hpd_upper': bayesian_hpd_upper,
        'bayesian_hpd_amplitude': bayesian_hpd_upper - bayesian_hpd_lower
    })
    
    # Add original data
    for i, feature in enumerate(feature_names):
        results_df[feature] = X_data[feature] if hasattr(X_data, 'columns') else X_data[:, i]
    
    # Save results
    results_df.to_csv(f'{base_dir}/tables/probability_intervals.tsv', sep='\t', index=False)
    results_df.to_excel(f'{base_dir}/tables/probability_intervals.xlsx', index=False)
    
    print(f"Probability intervals saved to {base_dir}/tables/probability_intervals.tsv")
    
    return results_df


def plot_posterior_predictive_by_disease(bayesian_model, X_data, disease_names, feature_names, base_dir):
    """Plot posterior predictive distributions for each disease individually"""
    print("Plotting posterior predictive distributions by disease...")
    
    # Create directory for posterior predictive distributions
    predictive_dir = f'{base_dir}/figures/posterior_predictive'
    os.makedirs(predictive_dir, exist_ok=True)
    
    # Prepare data with intercept
    X_with_intercept = np.column_stack([np.ones(len(X_data)), X_data])
    
    # Get posterior predictive distribution
    posterior_predictive = bayesian_model.get_posterior_predictive(X_with_intercept)
    
    # Plot for each disease individually
    for i, disease_name in enumerate(disease_names):
        plt.figure(figsize=(12, 8))
        
        # Get posterior predictive samples for this disease
        disease_samples = posterior_predictive[:, i]
        
        # Plot histogram with KDE
        plt.hist(disease_samples, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(disease_samples)
        x_vals = np.linspace(disease_samples.min(), disease_samples.max(), 1000)
        plt.plot(x_vals, kde(x_vals), 'g-', linewidth=2)
        
        # Calculate statistics
        mean_prob = np.mean(disease_samples)
        hpd_lower, hpd_upper = bayesian_model.hpd(disease_samples)
        
        # Add vertical lines
        plt.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.4f}')
        plt.axvline(hpd_lower, color='orange', linestyle=':', linewidth=2, label=f'HPD Lower: {hpd_lower:.4f}')
        plt.axvline(hpd_upper, color='orange', linestyle=':', linewidth=2, label=f'HPD Upper: {hpd_upper:.4f}')
        
        # Add feature values to title
        feature_vals = ""
        for j, feature in enumerate(feature_names):
            feature_val = X_data.iloc[i][feature] if hasattr(X_data, 'iloc') else X_data[i, j]
            feature_vals += f"{feature}: {feature_val:.4f}\n"
        
        plt.xlabel('Probability of Pandemic')
        plt.ylabel('Density')
        plt.title(f'Posterior Predictive Distribution\n{disease_name}\n{feature_vals}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save individual figure
        safe_disease_name = "".join(c for c in disease_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(f'{predictive_dir}/posterior_predictive_{safe_disease_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{predictive_dir}/posterior_predictive_{safe_disease_name}.svg', bbox_inches='tight')
        plt.close()
    
    # Create summary table of posterior predictive statistics
    predictive_summary = []
    for i, disease_name in enumerate(disease_names):
        disease_samples = posterior_predictive[:, i]
        mean_prob = np.mean(disease_samples)
        hpd_lower, hpd_upper = bayesian_model.hpd(disease_samples)
        
        predictive_summary.append({
            'disease_name': disease_name,
            'mean_probability': mean_prob,
            'hpd_lower': hpd_lower,
            'hpd_upper': hpd_upper,
            'hpd_amplitude': hpd_upper - hpd_lower
        })
    
    predictive_summary_df = pd.DataFrame(predictive_summary)
    predictive_summary_df.to_csv(f'{base_dir}/tables/posterior_predictive_summary.tsv', sep='\t', index=False)
    predictive_summary_df.to_excel(f'{base_dir}/tables/posterior_predictive_summary.xlsx', index=False)
    
    print(f"Posterior predictive distributions saved to {predictive_dir}/")
    print(f"Posterior predictive summary saved to {base_dir}/tables/posterior_predictive_summary.tsv")
    
    return predictive_summary_df


def plot_separate_roc_curves(classical_results, bayesian_results, best_features, base_dir):
    """Plot separate ROC curves for classical and Bayesian models for both train and test sets"""
    print("Plotting separate ROC curves...")
    
    # Create directory for ROC curves
    roc_dir = f'{base_dir}/figures/roc_curves'
    os.makedirs(roc_dir, exist_ok=True)
    
    # Plot Classical Model ROC Curves only if classical_results is provided
    if classical_results is not None:
        plt.figure(figsize=(12, 10))
        plt.plot(classical_results['fpr_test'], classical_results['tpr_test'], color='darkorange', lw=3, 
                 label=f'Test ROC (AUC = {classical_results["roc_auc_test"]:.4f})')
        plt.plot(classical_results['fpr_train'], classical_results['tpr_train'], color='blue', lw=3, 
                 label=f'Train ROC (AUC = {classical_results["roc_auc_train"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title(f'Classical Logistic Regression - ROC Curves\nFeatures: {", ".join(best_features)}', fontsize=18)
        plt.legend(loc="lower right", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{roc_dir}/classical_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{roc_dir}/classical_roc_curves.svg', bbox_inches='tight')
        plt.close()
    
    # Plot Bayesian Model ROC Curves only if bayesian_results is provided
    if bayesian_results is not None:
        plt.figure(figsize=(12, 10))
        plt.plot(bayesian_results['fpr_test'], bayesian_results['tpr_test'], color='darkgreen', lw=3, 
                 label=f'Test ROC (AUC = {bayesian_results["roc_auc_test"]:.4f})')
        plt.plot(bayesian_results['fpr_train'], bayesian_results['tpr_train'], color='purple', lw=3, 
                 label=f'Train ROC (AUC = {bayesian_results["roc_auc_train"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title(f'Bayesian Logistic Regression - ROC Curves\nFeatures: {", ".join(best_features)}', fontsize=18)
        plt.legend(loc="lower right", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{roc_dir}/bayesian_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{roc_dir}/bayesian_roc_curves.svg', bbox_inches='tight')
        plt.close()
    
    print(f"Separate ROC curves saved to {roc_dir}/")


def run_analysis_with_fixed_features(base_dir_name):
    
    # Create directories
    base_dir = base_dir_name
    os.makedirs(f'{base_dir}/models', exist_ok=True)
    os.makedirs(f'{base_dir}/figures', exist_ok=True)
    os.makedirs(f'{base_dir}/tables', exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS WITH FIXED FEATURES")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_excel('/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/dataset/data_original.xlsx')
    print("Dataset shape:", df.shape)
    
    # Data preprocessing - handle NaN values and binary columns properly
    df_processed = df.copy()
    
    # COLUNAS FIXAS
    fixed_columns = ['disease_name', 'etiological_name', 'R0_min', 'infectious_period_average_days', 
                    'lethality_rate', 'incubation_period_days_min', 'pandemic']
    
    # Verificar se todas as colunas existem
    missing_columns = [col for col in fixed_columns if col not in df_processed.columns]
    if missing_columns:
        print(f"AVISO: Colunas faltantes: {missing_columns}")
        fixed_columns = [col for col in fixed_columns if col in df_processed.columns]
    
    # Features numéricas fixas
    numeric_features = ['R0_min', 'infectious_period_average_days', 'lethality_rate', 
                       'incubation_period_days_min']
    
    # Handle binary columns
    binary_columns = [ 'pandemic']
                       
    # Convert binary columns to numeric (1 for 'yes', 0 for 'no' or 'Not Applicable')
    for col in binary_columns:
        if col in df_processed.columns:
            # First, fill NaN values with 'no' for binary columns
            df_processed[col] = df_processed[col].fillna('no')
            
            # Convert to numeric
            df_processed[col] = df_processed[col].apply(
                lambda x: 1 if str(x).strip().lower() == 'yes' else 0
            )
    
    # Handle numeric columns - remove or impute NaN values
    for col in numeric_features:
        if col in df_processed.columns and col not in binary_columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                print(f"Imputing {nan_count} missing values in {col} with median")
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
    
    # Create final numeric dataset
    final_numeric_columns = [col for col in numeric_features if col in df_processed.columns]
    target_column = 'pandemic'
    
    df_numeric = df_processed[final_numeric_columns + [target_column]].copy()
    
    print(f"Final dataset for analysis: {df_numeric.shape}")
    print(f"Using fixed features: {final_numeric_columns}")
    
    # Check for any remaining missing values
    missing_values = df_numeric.isna().sum().sum()
    if missing_values > 0:
        print(f"WARNING: Still {missing_values} missing values in the dataset after preprocessing")
        print("Dropping rows with missing values...")
        df_numeric = df_numeric.dropna()
    
    # Prepare data - ensure no NaN values
    X = df_numeric[final_numeric_columns]
    y = df_numeric[target_column]
    
    # Remove any remaining NaN values just to be safe
    non_nan_indices = ~X.isna().any(axis=1) & ~y.isna()
    X = X[non_nan_indices]
    y = y[non_nan_indices]
    
    print(f"Final clean dataset shape: X={X.shape}, y={y.shape}")
    
    # USAR APENAS AS FEATURES FIXAS - SEM COMBINAÇÕES
    features = final_numeric_columns
    print(f"Using fixed feature set: {features}")
    
    results = []
    
    try:
        X_subset = X[features]
        
        # Ensure we have enough samples and features
        if len(X_subset) < 2 or X_subset.shape[1] < len(features):
            print("Not enough samples for analysis.")
            return None
            
        # Split data - THIS SPLIT WILL BE USED FOR BOTH CLASSICAL AND BAYESIAN MODELS
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Check if we have both classes in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print("Not enough classes in train/test split.")
            return None
        
        # Test WITHOUT scaling only (as requested)
        lr_model_unscaled = LogisticRegression(random_state=42, max_iter=10000)
        lr_results_unscaled = evaluate_model(X_train, X_test, y_train, y_test,
                                           lr_model_unscaled, 'LogisticRegression',
                                           use_scaling=False)
        
        # Store successful results for unscaled method only
        if lr_results_unscaled['success']:
            results.append({
                'features': ', '.join(features),
                'num_features': len(features),
                'model_type': 'LogisticRegression',
                'scaling_method': 'Without Scaling',
                'accuracy_test': lr_results_unscaled['accuracy_test'],
                'precision_test': lr_results_unscaled['precision_test'],
                'recall_test': lr_results_unscaled['recall_test'],
                'f1_test': lr_results_unscaled['f1_test'],
                'roc_auc_test': lr_results_unscaled['roc_auc_test'],
                'accuracy_train': lr_results_unscaled['accuracy_train'],
                'precision_train': lr_results_unscaled['precision_train'],
                'recall_train': lr_results_unscaled['recall_train'],
                'f1_train': lr_results_unscaled['f1_train'],
                'roc_auc_train': lr_results_unscaled['roc_auc_train'],
                'roc_auc_diff': lr_results_unscaled['roc_auc_diff'],
                'accuracy_diff': lr_results_unscaled['accuracy_diff'],
                'composite_score': lr_results_unscaled['composite_score'],
                'feature_list': list(features),
                'coefficients': lr_results_unscaled['coefficients'],
                'use_scaling': lr_results_unscaled['use_scaling'],
                # Store the train/test split for Bayesian model
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
            
            best_combination = {
                'features': features,
                'model_type': 'LogisticRegression',
                'scaling_method': 'Without Scaling',
                'results': lr_results_unscaled,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'composite_score': lr_results_unscaled['composite_score'],
                'coefficients': lr_results_unscaled['coefficients'],
                'use_scaling': lr_results_unscaled['use_scaling']
            }
            best_score = lr_results_unscaled['composite_score']
        else:
            print("Classical model training failed.")
            return None
    except Exception as e:
        print(f"Error with features {features}: {e}")
        return None
    
    if best_combination:
        print(f"\nFixed features analysis: {best_combination['features']}")
        print(f"Best model: {best_combination['model_type']}")
        print(f"Best scaling method: {best_combination['scaling_method']}")
        print(f"Best composite score (lower is better): {best_score:.4f}")
        print(f"ROC AUC - Train: {best_combination['results']['roc_auc_train']:.4f}, Test: {best_combination['results']['roc_auc_test']:.4f}")
        print(f"Accuracy - Train: {best_combination['results']['accuracy_train']:.4f}, Test: {best_combination['results']['accuracy_test']:.4f}")
        print(f"ROC AUC Difference: {best_combination['results']['roc_auc_diff']:.4f}")
        print(f"Accuracy Difference: {best_combination['results']['accuracy_diff']:.4f}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # CALCULAR NPPI (NOVO PANDEMIC POTENTIAL INDEX)
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def compute(row, coef):
            x = coef["intercept"] \
                + coef["R0_min"]*row["R0_min"] \
                + coef["infectious_period_average_days"]*row["infectious_period_average_days"] \
                + coef["lethality_rate"]*row["lethality_rate"] \
                + coef["incubation_period_days_min"]*row["incubation_period_days_min"]
                
            return sigmoid(x)

        # Preparar coeficientes para cálculo do NPPI
        classical_coef_dict = {
            "intercept": best_combination['results']['model'].intercept_[0],
            "R0_min": best_combination['coefficients'][0] if len(best_combination['coefficients']) > 0 else 0,
            "infectious_period_average_days": best_combination['coefficients'][1] if len(best_combination['coefficients']) > 1 else 0,
            "lethality_rate": best_combination['coefficients'][2] if len(best_combination['coefficients']) > 2 else 0,
            "incubation_period_days_min": best_combination['coefficients'][3] if len(best_combination['coefficients']) > 3 else 0
            
        }

        # Calcular NPPI para todos os dados
        df_with_nppi = df_processed[fixed_columns].copy()
        df_with_nppi['NPPI_classic'] = df_with_nppi.apply(lambda row: compute(row, classical_coef_dict), axis=1)
        
    else:
        print(f"No valid models could be trained with fixed features. Please check your data.")
        return None
    
    # Save results
    results_df.to_csv(f'{base_dir}/tables/fixed_features_results.tsv', sep='\t', index=False)
    results_df.to_excel(f'{base_dir}/tables/fixed_features_results.xlsx', index=False)
    
    # Plot results for fixed features
    if best_combination:
        best_features = best_combination['features']
        best_model = best_combination['model_type']
        best_scaling = best_combination['scaling_method']
        best_results = best_combination['results']
        
        # Plot separate ROC curves for classical model
        plot_separate_roc_curves(best_results, None, best_features, base_dir)
        
        # Confusion Matrix for test set
        cm_test = confusion_matrix(best_combination['y_test'], best_results['y_pred_test'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Test Set\n{best_model} ({best_scaling})')
        plt.savefig(f'{base_dir}/figures/fixed_features_confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/fixed_features_confusion_matrix_test.svg', bbox_inches='tight')
        plt.close()

        # Confusion Matrix for train set
        cm_train = confusion_matrix(best_combination['y_train'], best_results['y_pred_train'])  # This should work now
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Train Set\n{best_model} ({best_scaling})')
        plt.savefig(f'{base_dir}/figures/fixed_features_confusion_matrix_train.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/fixed_features_confusion_matrix_train.svg', bbox_inches='tight')
        plt.close()
        
        # Save best model
        joblib.dump(best_results['model'], f'{base_dir}/models/fixed_features_classical_model.pkl')
        
        # Calculate confidence intervals for classical model
        print("Calculating confidence intervals for classical model...")
        X_for_ci = best_combination['X_train'].values
        
        classical_coef, classical_std, classical_ci_lower, classical_ci_upper = calculate_confidence_intervals_corrected(
            best_results['model'], X_for_ci, best_combination['y_train'].values
        )
        
        # Save model summary with confidence intervals
        with open(f'{base_dir}/summary.txt', 'w') as f:
            f.write("FIXED FEATURES CLASSICAL MODEL SUMMARY\n")
            f.write("=====================================\n\n")
            f.write(f"Fixed Feature Combination: {', '.join(best_features)}\n")
            f.write(f"Best Model Type: {best_model}\n")
            f.write(f"Best Scaling Method: {best_scaling}\n")
            f.write(f"Number of Features: {len(best_features)}\n")
            f.write(f"Composite Score (lower is better): {best_combination['composite_score']:.4f}\n\n")
            
            f.write("Performance Metrics - TRAIN SET:\n")
            f.write(f"Accuracy: {best_results['accuracy_train']:.4f}\n")
            f.write(f"Precision: {best_results['precision_train']:.4f}\n")
            f.write(f"Recall: {best_results['recall_train']:.4f}\n")
            f.write(f"F1-Score: {best_results['f1_train']:.4f}\n")
            f.write(f"ROC AUC: {best_results['roc_auc_train']:.4f}\n\n")
            
            f.write("Performance Metrics - TEST SET:\n")
            f.write(f"Accuracy: {best_results['accuracy_test']:.4f}\n")
            f.write(f"Precision: {best_results['precision_test']:.4f}\n")
            f.write(f"Recall: {best_results['recall_test']:.4f}\n")
            f.write(f"F1-Score: {best_results['f1_test']:.4f}\n")
            f.write(f"ROC AUC: {best_results['roc_auc_test']:.4f}\n\n")
            
            f.write("Differences (Train - Test):\n")
            f.write(f"ROC AUC Difference: {best_results['roc_auc_diff']:.4f}\n")
            f.write(f"Accuracy Difference: {best_results['accuracy_diff']:.4f}\n\n")
            
            f.write("FEATURE IMPORTANCE ANALYSIS:\n")
            f.write("============================\n\n")
            
            f.write("Model trained WITHOUT feature scaling\n")
            f.write("Coefficients (on original features):\n")
            for feature, coef in zip(best_features, best_results['coefficients']):
                f.write(f"  {feature}: {coef:.4f}\n")
            
            f.write("\nFeature Importance Ranking (based on absolute coefficients):\n")
            importance_df = pd.DataFrame({
                'Feature': best_features,
                'Absolute_Coefficient': np.abs(best_results['coefficients'])
            }).sort_values('Absolute_Coefficient', ascending=False)
            
            for _, row in importance_df.iterrows():
                f.write(f"  {row['Feature']}: {row['Absolute_Coefficient']:.4f}\n")
        
        # Prepare data for Bayesian analysis USING THE SAME TRAIN/TEST SPLIT
        print(f"\n{'='*80}")
        print("BAYESIAN ANALYSIS USING SAME TRAIN/TEST SPLIT AS CLASSICAL MODEL")
        print(f"{'='*80}")
        print(f"Fixed features: {best_features}")
        
        # Use the EXACT SAME training and test data as the classical model
        X_bayes_train = best_combination['X_train'].values
        X_bayes_test = best_combination['X_test'].values
        y_bayes_train = best_combination['y_train'].values
        y_bayes_test = best_combination['y_test'].values
        
        print(f"Bayesian model training data shape: X_train={X_bayes_train.shape}, y_train={y_bayes_train.shape}")
        print(f"Bayesian model test data shape: X_test={X_bayes_test.shape}, y_test={y_bayes_test.shape}")
        
        # Add intercept column to training data
        X_bayes_train_with_intercept = np.column_stack([np.ones(len(X_bayes_train)), X_bayes_train])
        X_bayes_test_with_intercept = np.column_stack([np.ones(len(X_bayes_test)), X_bayes_test])
        
        # Fit Bayesian Logistic Regression model with MULTIPLE CHAINS 
        print("\nRunning Bayesian Logistic Regression with 3 independent chains...")
        print("USING THE SAME TRAIN/TEST SPLIT AS CLASSICAL MODEL")
        print("USING ORIGINAL DATA SCALE with adaptive MCMC")
        print(f"USING FIXED FEATURES: {best_features}")
        
        bayesian_model = BayesianLogisticRegressionMultiChainCorrected(
            n_chains=3, n_iter=1050000, burn_in=50000
        )
        bayesian_model.fit(X_bayes_train_with_intercept, y_bayes_train)
        
        # Make predictions on the same test set
        y_pred_bayes_test = bayesian_model.predict(X_bayes_test_with_intercept)
        y_pred_proba_bayes_test = bayesian_model.predict_proba(X_bayes_test_with_intercept)
        
        # Calculate Bayesian model performance metrics
        accuracy_bayes_test = accuracy_score(y_bayes_test, y_pred_bayes_test)
        precision_bayes_test = precision_score(y_bayes_test, y_pred_bayes_test, zero_division=0)
        recall_bayes_test = recall_score(y_bayes_test, y_pred_bayes_test, zero_division=0)
        f1_bayes_test = f1_score(y_bayes_test, y_pred_bayes_test, zero_division=0)
        
        # Handle ROC curve for binary labels
        y_bayes_test_numeric = y_bayes_test
        if y_bayes_test.dtype == object or y_bayes_test.dtype == 'category':
            y_bayes_test_numeric = pd.Categorical(y_bayes_test).codes
            
        fpr_bayes_test, tpr_bayes_test, _ = roc_curve(y_bayes_test_numeric, y_pred_proba_bayes_test)
        roc_auc_bayes_test = auc(fpr_bayes_test, tpr_bayes_test)
        
        # Make predictions on training set for comparison
        y_pred_bayes_train = bayesian_model.predict(X_bayes_train_with_intercept)
        y_pred_proba_bayes_train = bayesian_model.predict_proba(X_bayes_train_with_intercept)
        
        accuracy_bayes_train = accuracy_score(y_bayes_train, y_pred_bayes_train)
        precision_bayes_train = precision_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        recall_bayes_train = recall_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        f1_bayes_train = f1_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        
        # Handle ROC curve for binary labels
        y_bayes_train_numeric = y_bayes_train
        if y_bayes_train.dtype == object or y_bayes_train.dtype == 'category':
            y_bayes_train_numeric = pd.Categorical(y_bayes_train).codes
            
        fpr_bayes_train, tpr_bayes_train, _ = roc_curve(y_bayes_train_numeric, y_pred_proba_bayes_train)
        roc_auc_bayes_train = auc(fpr_bayes_train, tpr_bayes_train)
        
        # Calculate differences for Bayesian model
        roc_auc_diff_bayes = abs(roc_auc_bayes_train - roc_auc_bayes_test)
        accuracy_diff_bayes = abs(accuracy_bayes_train - accuracy_bayes_test)
        composite_score_bayes = (roc_auc_diff_bayes * 0.7 + accuracy_diff_bayes * 0.3)
        
        # Create bayesian results dictionary for ROC plotting
        bayesian_results = {
            'fpr_test': fpr_bayes_test,
            'tpr_test': tpr_bayes_test,
            'roc_auc_test': roc_auc_bayes_test,
            'fpr_train': fpr_bayes_train,
            'tpr_train': tpr_bayes_train,
            'roc_auc_train': roc_auc_bayes_train
        }
        
        # Plot separate ROC curves for Bayesian model
        plot_separate_roc_curves(None, bayesian_results, best_features, base_dir)
        
        # Confusion Matrix for Bayesian test set
        cm_bayes_test = confusion_matrix(y_bayes_test, y_pred_bayes_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_bayes_test, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Bayesian Test Set')
        plt.savefig(f'{base_dir}/figures/bayesian_confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/bayesian_confusion_matrix_test.svg', bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix for Bayesian train set
        cm_bayes_train = confusion_matrix(y_bayes_train, y_pred_bayes_train)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_bayes_train, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Bayesian Train Set')
        plt.savefig(f'{base_dir}/figures/bayesian_confusion_matrix_train.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/bayesian_confusion_matrix_train.svg', bbox_inches='tight')
        plt.close()
        
        # Save Bayesian model
        joblib.dump(bayesian_model, f'{base_dir}/models/fixed_features_bayesian_model.pkl')
        
        # Compute HPD intervals from combined chains
        hpd_intervals = []
        for i in range(bayesian_model.p):
            hpd_lower, hpd_upper = bayesian_model.hpd(bayesian_model.beta_post[:, i])
            hpd_intervals.append((hpd_lower, hpd_upper))
        
        # Calculate classical model coefficients on original scale for comparison
        classical_coef_full_original = np.concatenate([[best_results['model'].intercept_[0]], best_combination['coefficients']])
        
        # Calcular NPPI bayesiano
        bayesian_coef_dict = {
            "intercept": bayesian_model.beta_mean[0],
            "R0_min": bayesian_model.beta_mean[1] if len(bayesian_model.beta_mean) > 1 else 0,
            "infectious_period_average_days": bayesian_model.beta_mean[2] if len(bayesian_model.beta_mean) > 2 else 0,
            "lethality_rate": bayesian_model.beta_mean[3] if len(bayesian_model.beta_mean) > 3 else 0,
            "incubation_period_days_min": bayesian_model.beta_mean[4] if len(bayesian_model.beta_mean) > 4 else 0
            
        }
        
        df_with_nppi['NPPI_bayesian'] = df_with_nppi.apply(lambda row: compute(row, bayesian_coef_dict), axis=1)
        
        # SALVAR RESULTADOS COM NPPI
        df_with_nppi.to_csv(f'{base_dir}/tables/fixed_features_with_nppi.tsv', sep='\t', index=False)
        df_with_nppi.to_excel(f'{base_dir}/tables/fixed_features_with_nppi.xlsx', index=False)
        
        # Plot Markov chains and posterior distributions
        feature_names_bayesian = ['Intercept'] + best_features
        plot_markov_chains(bayesian_model, feature_names_bayesian, base_dir)
        plot_posterior_distributions(bayesian_model, feature_names_bayesian, base_dir)
        
        # Calculate probability intervals
        probability_intervals_df = calculate_probability_intervals(
            best_results['model'], bayesian_model, X, best_features, base_dir
        )
        
        # Plot posterior predictive distributions by disease
        predictive_summary_df = plot_posterior_predictive_by_disease(
            bayesian_model, X, df_processed['disease_name'].values, best_features, base_dir
        )
        
        # Save Bayesian results to summary with detailed comparison
        with open(f'{base_dir}/summary.txt', 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARISON: CLASSICAL vs BAYESIAN LOGISTIC REGRESSION\n")
            f.write("USING THE SAME TRAIN/TEST SPLIT AND FIXED FEATURES\n")
            f.write("="*80 + "\n\n")
            
            f.write("PERFORMANCE COMPARISON ON TEST SET\n")
            f.write("="*40 + "\n")
            f.write(f"{'Metric':<20} {'Classical':<12} {'Bayesian':<12}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Accuracy':<20} {best_results['accuracy_test']:<12.4f} {accuracy_bayes_test:<12.4f}\n")
            f.write(f"{'Precision':<20} {best_results['precision_test']:<12.4f} {precision_bayes_test:<12.4f}\n")
            f.write(f"{'Recall':<20} {best_results['recall_test']:<12.4f} {recall_bayes_test:<12.4f}\n")
            f.write(f"{'F1-Score':<20} {best_results['f1_test']:<12.4f} {f1_bayes_test:<12.4f}\n")
            f.write(f"{'ROC AUC':<20} {best_results['roc_auc_test']:<12.4f} {roc_auc_bayes_test:<12.4f}\n")
            f.write(f"{'Composite Score':<20} {best_combination['composite_score']:<12.4f} {composite_score_bayes:<12.4f}\n\n")
            
            f.write("MODEL COMPARISON - ORIGINAL SCALE COEFFICIENTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Fixed features used in both models: {', '.join(best_features)}\n\n")
            
            feature_names = ['Intercept'] + list(best_features)
            
            f.write(f"{'Parameter':<25} {'Classical Coef':<15} {'Classical 95% CI':<25} {'Bayesian Mean':<15} {'Bayesian 95% HPD':<25}\n")
            f.write("-" * 100 + "\n")
            
            for i, name in enumerate(feature_names):
                if i < len(classical_coef_full_original) and i < len(bayesian_model.beta_mean):
                    classical_coef_val = classical_coef_full_original[i]
                    classical_ci_str = f"[{classical_ci_lower[i]:.4f}, {classical_ci_upper[i]:.4f}]" if i < len(classical_ci_lower) else "[N/A, N/A]"
                    bayesian_mean_val = bayesian_model.beta_mean[i]
                    bayesian_hpd_str = f"[{hpd_intervals[i][0]:.4f}, {hpd_intervals[i][1]:.4f}]"
                    
                    f.write(f"{name:<25} {classical_coef_val:>14.4f} {classical_ci_str:<25} {bayesian_mean_val:>14.4f} {bayesian_hpd_str:<25}\n")
            
            f.write("\n\nNPPI (NOVO PANDEMIC POTENTIAL INDEX) CALCULATION\n")
            f.write("="*60 + "\n\n")
            f.write("NPPI = sigmoid(intercept + coef1*R0_min + coef2*infectious_period + coef3*lethality_rate + coef4*incubation_period)\n\n")
            f.write("Classical NPPI coefficients:\n")
            for key, value in classical_coef_dict.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nBayesian NPPI coefficients:\n")
            for key, value in bayesian_coef_dict.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"\nNPPI values saved to: {base_dir}/tables/fixed_features_with_nppi.tsv\n")
            f.write("File includes: original data + NPPI_classic + NPPI_bayesian\n")
            
            f.write(f"\nProbability intervals saved to: {base_dir}/tables/probability_intervals.tsv\n")
            f.write("Includes: classical probabilities with CI95% and Bayesian probabilities with HPD95%\n")
            
            f.write(f"\nPosterior predictive summary saved to: {base_dir}/tables/posterior_predictive_summary.tsv\n")
            f.write("Includes: mean probability and HPD intervals for each disease\n")
        
        return {
            'best_features': best_features,
            'classical_model': best_results['model'],
            'bayesian_model': bayesian_model,
            'use_scaling': False,
            'classical_performance': {
                'accuracy_test': best_results['accuracy_test'],
                'precision_test': best_results['precision_test'],
                'recall_test': best_results['recall_test'],
                'f1_test': best_results['f1_test'],
                'roc_auc_test': best_results['roc_auc_test']
            },
            'bayesian_performance': {
                'accuracy_test': accuracy_bayes_test,
                'precision_test': precision_bayes_test,
                'recall_test': recall_bayes_test,
                'f1_test': f1_bayes_test,
                'roc_auc_test': roc_auc_bayes_test
            },
            'df_with_nppi': df_with_nppi,
            'probability_intervals': probability_intervals_df,
            'predictive_summary': predictive_summary_df
        }
    return None


if __name__ == "__main__":
    print("Starting analysis with fixed features...")
    result_fixed = run_analysis_with_fixed_features('/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/MCMC_simulations')
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Results saved in:")
    print("- analysis/fixed_features/")
    print("Files generated:")
    print("  - fixed_features_with_nppi.tsv/.xlsx (NPPI values)")
    print("  - probability_intervals.tsv/.xlsx (Classical and Bayesian probabilities with intervals)")
    print("  - posterior_predictive_summary.tsv/.xlsx (Posterior predictive statistics)")
    print("  - Separate ROC curves for classical and Bayesian models in figures/roc_curves/")
    print("  - Markov chains and posterior distributions in figures/ directories")
    print(f"{'='*80}")
