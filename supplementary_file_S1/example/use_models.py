import pandas as pd
import numpy as np
import joblib
import sys
import os

# Bayesian Gibbs Sampling Implementation with MULTIPLE CHAINS
class BayesianLogisticRegressionMultiChainCorrected:
    def __init__(self, n_chains=5, n_iter=10005000, burn_in=5000, beta_prior_mean=0, beta_prior_precision=1):
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

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data"""
    print("Loading data...")
    df = pd.read_excel(file_path)
    print(f"Data loaded: {df.shape}")
    
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Preprocessing
    df_processed = df.copy()
    
    # Binary columns - convert to 0/1
    binary_columns = ['permanent_immunity', 'approved_vaccine', 'pandemic']
    
    for col in binary_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('no')
            df_processed[col] = df_processed[col].apply(
                lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
            )
    
    # Numeric columns - impute missing values with median
    numeric_columns = ['R0_min', 'R0_max', 'R0_amplitude', 'infectious_period_average_days', 
                       'lethality_rate', 'incubation_period_days_min', 'incubation_period_days_max', 
                       'incubation_period_days_amplitude']
    
    for col in numeric_columns:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                print(f"Imputed {nan_count} missing values in {col} with median: {median_val}")
    
    return df_processed

def get_available_features(df):
    """Returns available features in the dataset"""
    available_features = [
        'R0_min', 'R0_max', 'R0_amplitude', 'infectious_period_average_days',
        'lethality_rate', 'incubation_period_days_min', 'incubation_period_days_max',
        'incubation_period_days_amplitude', 'permanent_immunity', 'approved_vaccine'
    ]
    
    # Filter only features that exist in the dataframe
    available_features = [f for f in available_features if f in df.columns]
    
    print(f"Available features in dataset: {available_features}")
    return available_features

def predict_with_available_features(model, X, available_features, original_features):
    """Makes predictions using only available features"""
    
    # Create array with zeros for all original features
    X_full = np.zeros((len(X), len(original_features)))
    
    # Fill with values from available features
    for i, feature in enumerate(original_features):
        if feature in available_features:
            feature_idx = available_features.index(feature)
            X_full[:, i] = X[:, feature_idx]
        else:
            # For missing features, use default value (0 or mean)
            print(f"Missing feature: {feature} - using default value 0")
            X_full[:, i] = 0
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X_full)), X_full])
    
    # Make prediction
    probabilities = model.predict_proba(X_with_intercept)
    
    return probabilities

def calculate_pandemic_probabilities(model, df, available_features, original_features):
    """Calculates pandemic probabilities for all pathogens"""
    print(f"\nCalculating probabilities for {len(df)} pathogens...")
    
    # Prepare data for prediction
    X_pred = df[available_features].copy()
    
    # Check and handle missing values
    missing_summary = X_pred.isna().sum()
    if missing_summary.sum() > 0:
        print("Missing values found:")
        print(missing_summary)
        # Fill with medians
        for col in available_features:
            if X_pred[col].isna().sum() > 0:
                median_val = X_pred[col].median()
                X_pred[col] = X_pred[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")
    
    # Make adapted predictions
    probabilities = predict_with_available_features(
        model, 
        X_pred.values, 
        available_features, 
        original_features
    )
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'etiological_name': df['etiological_name'],
        'disease_name': df['disease_name'],
        'pandemic_actual': df['pandemic'] if 'pandemic' in df.columns else 'N/A',
        'pandemic_probability': probabilities,
        'pandemic_predicted': (probabilities >= 0.5).astype(int)
    })
    
    # Add used features for reference
    for feature in available_features:
        results[f'feature_{feature}'] = df[feature]
    
    # Add information about missing features
    results['missing_features'] = ', '.join(set(original_features) - set(available_features))
    
    return results

def analyze_results(results):
    """Analyzes and displays result statistics"""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"Total pathogens analyzed: {len(results)}")
    print(f"Average probability: {results['pandemic_probability'].mean():.4f}")
    print(f"Minimum probability: {results['pandemic_probability'].min():.4f}")
    print(f"Maximum probability: {results['pandemic_probability'].max():.4f}")
    
    # Classification by probability
    high_risk = results[results['pandemic_probability'] >= 0.7]
    medium_risk = results[(results['pandemic_probability'] >= 0.3) & (results['pandemic_probability'] < 0.7)]
    low_risk = results[results['pandemic_probability'] < 0.3]
    
    print(f"\nHigh-risk pathogens (prob >= 0.7): {len(high_risk)}")
    print(f"Medium-risk pathogens (0.3 <= prob < 0.7): {len(medium_risk)}")
    print(f"Low-risk pathogens (prob < 0.3): {len(low_risk)}")
    
    # Top 10 pathogens with highest probability
    print(f"\nTOP 10 PATHOGENS WITH HIGHEST PANDEMIC PROBABILITY:")
    print("-" * 80)
    top_10 = results.nlargest(10, 'pandemic_probability')[['etiological_name', 'disease_name', 'pandemic_probability']]
    for idx, row in top_10.iterrows():
        print(f"{row['etiological_name']:30} | {row['disease_name']:30} | {row['pandemic_probability']:.4f}")
    
    # Check SARS-CoV-2 specifically
    sars_cov2 = results[results['etiological_name'].str.contains('SARS-CoV-2', case=False, na=False)]
    if len(sars_cov2) > 0:
        print(f"\nSARS-CoV-2:")
        print(f"Pandemic probability: {sars_cov2['pandemic_probability'].iloc[0]:.4f}")
        print(f"Classification: {'PANDEMIC' if sars_cov2['pandemic_predicted'].iloc[0] == 1 else 'NON-PANDEMIC'}")

def main():
    # Configuration
    data_file = '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/example/test_data.xlsx'
    model_file = '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S2/classical_bayesian_models/models/best_bayesian_model.pkl'
    output_file = '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/example/pandemic_probabilities_all_pathogens.csv'
    
    try:
        # 1. Load model
        print("Loading Bayesian model...")
        model = joblib.load(model_file)
        print("Model loaded successfully!")
        
        # Check model information
        print(f"\nModel information:")
        print(f"Number of parameters: {model.p}")
        print(f"Chain size: {len(model.beta_post)}")
        
        # 2. Load and preprocess data
        df = load_and_preprocess_data(data_file)
        
        # 3. Define original model features (based on previous error)
        original_features = ['R0_max', 'assintomatic_transmission', 'infectious_period_average_days', 'zoonotic_transmission', 'lethality_rate']
        
        # 4. Get available features
        available_features = get_available_features(df)
        
        # 5. Adjust to use only available features
        # We'll use available features that correspond to original ones
        usable_features = [f for f in original_features if f in available_features]
        
        print(f"\nUsable features: {usable_features}")
        print(f"Unavailable features: {set(original_features) - set(usable_features)}")
        
        if len(usable_features) == 0:
            print("ERROR: No model features are available in the data!")
            return
        
        # 6. Calculate probabilities
        results = calculate_pandemic_probabilities(model, df, usable_features, original_features)
        
        # 7. Analyze results
        analyze_results(results)
        
        # 8. Save results
        results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved in: {output_file}")
        
        # 9. Save summary version
        summary_cols = ['etiological_name', 'disease_name', 'pandemic_actual', 
                       'pandemic_probability', 'pandemic_predicted', 'missing_features']
        results[summary_cols].to_csv(output_file.replace('.csv', '_summary.csv'), 
                                   index=False, encoding='utf-8')
        print(f"Summary saved in: {output_file.replace('.csv', '_summary.csv')}")
        
        # 10. Save detailed analysis
        with open(output_file.replace('.csv', '_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("DETAILED ANALYSIS - PANDEMIC PROBABILITIES\n")
            f.write("="*50 + "\n\n")
            
            f.write("WARNING: Some original model features were not available\n")
            f.write(f"Original model features: {original_features}\n")
            f.write(f"Available features: {available_features}\n")
            f.write(f"Missing features: {list(set(original_features) - set(available_features))}\n\n")
            
            f.write(f"Total pathogens analyzed: {len(results)}\n")
            f.write(f"Average probability: {results['pandemic_probability'].mean():.4f}\n")
            f.write(f"Standard deviation: {results['pandemic_probability'].std():.4f}\n\n")
            
            f.write("RISK DISTRIBUTION:\n")
            high_risk = len(results[results['pandemic_probability'] >= 0.7])
            medium_risk = len(results[(results['pandemic_probability'] >= 0.3) & (results['pandemic_probability'] < 0.7)])
            low_risk = len(results[results['pandemic_probability'] < 0.3])
            
            f.write(f"High risk (≥0.7): {high_risk} pathogens ({high_risk/len(results)*100:.1f}%)\n")
            f.write(f"Medium risk (0.3-0.7): {medium_risk} pathogens ({medium_risk/len(results)*100:.1f}%)\n")
            f.write(f"Low risk (<0.3): {low_risk} pathogens ({low_risk/len(results)*100:.1f}%)\n\n")
            
            f.write("TOP 15 PATHOGENS WITH HIGHEST PANDEMIC RISK:\n")
            f.write("-" * 80 + "\n")
            top_15 = results.nlargest(15, 'pandemic_probability')
            for idx, row in top_15.iterrows():
                f.write(f"{row['etiological_name']:35} | {row['disease_name']:35} | {row['pandemic_probability']:.4f}\n")
        
        print(f"Detailed analysis saved in: {output_file.replace('.csv', '_analysis.txt')}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
