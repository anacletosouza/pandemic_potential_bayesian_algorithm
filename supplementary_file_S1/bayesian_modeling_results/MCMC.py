import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2_contingency, norm
import itertools
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 32

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
                    model_boot = LogisticRegression(random_state=42, max_iter=100000)
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

# Exhaustive feature combination testing
def evaluate_model(X_train, X_test, y_train, y_test, model, model_name, scaler=None, use_scaling=False):
    """Evaluate model performance and return metrics for both train and test"""
    try:
        # Apply scaling if specified
        if use_scaling and scaler is not None:
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_processed = X_test.values if hasattr(X_test, 'values') else X_test
            scaler = None  # No scaling applied
        
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
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc_test = auc(fpr_test, tpr_test)
        
        # Metrics for train set
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, zero_division=0)
        f1_train = f1_score(y_train, y_pred_train, zero_division=0)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        
        # Calculate differences between train and test (to minimize)
        roc_auc_diff = abs(roc_auc_train - roc_auc_test)
        accuracy_diff = abs(accuracy_train - accuracy_test)
        
        # Calculate composite score
        composite_score = (roc_auc_diff * 0.7 + accuracy_diff * 0.3)
        
        # Get feature coefficients and calculate real weights if scaling was used
        coefficients = model.coef_[0] if hasattr(model, 'coef_') else None
        real_weights = None
        
        if coefficients is not None and use_scaling and scaler is not None:
            # Convert coefficients back to original scale
            if hasattr(scaler, 'scale_'):
                real_weights = coefficients / scaler.scale_
            elif hasattr(scaler, 'mean_'):
                # For StandardScaler
                real_weights = coefficients / scaler.scale_ if hasattr(scaler, 'scale_') else coefficients
        
        return {
            'model': model,
            'scaler': scaler,
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
            'roc_auc_diff': roc_auc_diff,
            'accuracy_diff': accuracy_diff,
            'composite_score': composite_score,
            'coefficients': coefficients,
            'real_weights': real_weights,
            'use_scaling': use_scaling,
            'success': True
        }
    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return {'success': False}

def detect_column_types(df):
    """Detect numeric and categorical columns automatically"""
    numeric_columns = []
    categorical_columns = []
    
    for col in df.columns:
        # Skip the target column if it exists
        if col == 'pandemic':
            continue
            
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    
    print(f"Detected numeric columns ({len(numeric_columns)}): {numeric_columns}")
    print(f"Detected categorical columns ({len(categorical_columns)}): {categorical_columns}")
    
    return numeric_columns, categorical_columns

def run_pca_tsne_analysis(dir_base="analysis", 
                          df_file='/home/anacleto/projects/andressa/data_original.xlsx',
                          columns_numeric_selected=None, 
                          columns_categorical_selected=None):
    
    # Set font size
    plt.rcParams['font.size'] = 12

    # Criar todos os diretórios necessários
    base_dirs = [
        f'{dir_base}/PCA/tables',
        f'{dir_base}/PCA/figures', 
        f'{dir_base}/PCA/analysis',
        f'{dir_base}/tSNE/tables',
        f'{dir_base}/tSNE/figures',
        f'{dir_base}/tSNE/analysis',
        f'{dir_base}/cat/tables',
        f'{dir_base}/analysis',
        f'{dir_base}/figures', 
        f'{dir_base}/tables'    
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Diretório criado/verificado: {dir_path}")

    # Read the data
    try:
        df = pd.read_excel(df_file)
        print(f"Dados carregados: {df.shape}")
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        return

    # Display basic info about the data
    print("Data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())


    df_processed = df.copy()
    

    binary_columns = ['zoonotic_transmission', 'assintomatic_transmission', 
                      'permanent_immunity', 'approved_treatment', 
                      'approved_vaccine', 'pandemic']
    

    for col in binary_columns:
        if col in df_processed.columns:


            df_processed[col] = df_processed[col].fillna('no')

            df_processed[col] = df_processed[col].apply(
                lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
            )
    
    # Detect column types automatically
    numeric_columns, categorical_columns = detect_column_types(df_processed)
    
    # Use detected columns if not provided
    if columns_numeric_selected is None:
        columns_numeric_selected = numeric_columns
    if columns_categorical_selected is None:
        columns_categorical_selected = categorical_columns

    print(f"Using numeric columns: {columns_numeric_selected}")
    print(f"Using categorical columns: {columns_categorical_selected}")

    # 1. Create contingency tables and perform chi-squared tests
    categorical_columns_for_test = ['etiological_name', 'microrganism', 
                                    'feature', 'assintomatic_transmission', 
                                    'permanent_immunity', 'approved_treatment',  
                                    'approved_vaccine'
                                    ]
                          
    # Initialize summary file
    with open(f'{dir_base}/analysis/summary_PCA_tSNE.txt', 'w') as f:
        f.write("CHI-SQUARED TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")

    # Create contingency tables and perform chi-squared tests
    for col in categorical_columns_for_test:
        if col in df_processed.columns:
            # Create contingency table
            contingency_table = pd.crosstab(df_processed[col], df_processed['pandemic'])
            
            # Save tables
            try:
                contingency_table.to_excel(f'{dir_base}/cat/tables/contingency_{col}.xlsx')
                contingency_table.to_csv(f'{dir_base}/cat/tables/contingency_{col}.tsv', sep='\t')
            except Exception as e:
                print(f"Error with {col}: {e}")
            
            # Perform chi-squared test
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Save results to summary
                with open(f'{dir_base}/analysis/summary_PCA_tSNE.txt', 'a') as f:
                    f.write(f"Variable: {col}\n")
                    f.write(f"Chi-squared statistic: {chi2:.4f}\n")
                    f.write(f"P-value: {p_value:.4f}\n")
                    f.write(f"Degrees of freedom: {dof}\n")
                    f.write("Contingency Table:\n")
                    f.write(contingency_table.to_string())
                    f.write("\n\n" + "="*50 + "\n\n")
            except Exception as e:
                print(f"Error in chi-squared test to {col}: {e}")

    # 2. PCA Analysis with BEST FEATURES ONLY
    print(f"\nRunning PCA with best features: {columns_numeric_selected}")
    
    # CORREÇÃO: Preparar dados numéricos de forma robusta
    df_numeric = df_processed[columns_numeric_selected].copy()
    
    # Converter todas as colunas para numérico, forçando erros para NaN
    for col in columns_numeric_selected:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    
    # Verificar valores missing
    print(f"\nMissing values em cada feature:")
    missing_summary = df_numeric.isna().sum()
    print(missing_summary)
    
    # CORREÇÃO: Remover linhas com valores missing
    df_numeric_clean = df_numeric.dropna()
    original_rows = len(df_numeric)
    clean_rows = len(df_numeric_clean)
        
    try:

        valid_indices = df_numeric_clean.index
        df_filtered = df_processed.loc[valid_indices]
        
        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric_clean)

        # Perform PCA
        pca = PCA()
        pca_scores = pca.fit_transform(df_scaled)

        # Criar PCA dataframe
        pca_df = pd.DataFrame({
            'PC1': pca_scores[:, 0],
            'PC2': pca_scores[:, 1],
            'PC3': pca_scores[:, 2] if pca_scores.shape[1] > 2 else np.zeros(len(pca_scores)),
            'PC4': pca_scores[:, 3] if pca_scores.shape[1] > 3 else np.zeros(len(pca_scores)),
            'PC5': pca_scores[:, 4] if pca_scores.shape[1] > 4 else np.zeros(len(pca_scores)),
            'etiological_name': df_filtered['etiological_name'],
            'disease_name': df_filtered['disease_name'],
            'pandemic': df_filtered['pandemic']
        })

        # Save PCA scores
        pca_df.to_csv(f'{dir_base}/PCA/analysis/pca_scores.tsv', sep='\t', index=False)

        # Create PCA plot
        plt.figure(figsize=(15, 10))

        # Create color map for pandemic status
        colors = {1: 'blue', 0: 'red'}  # 1 = pandemic, 0 = no pandemic
        pca_df['color'] = pca_df['pandemic'].map(colors)

        # Create scatter plot
        for pandemic_status in [1, 0]:
            mask = pca_df['pandemic'] == pandemic_status
            if len(pca_df[mask]) > 0:
                color = 'blue' if pandemic_status == 1 else 'red'
                label = 'Pandemic' if pandemic_status == 1 else 'No Pandemic'
                plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                           c=color, edgecolor='black', s=100, alpha=0.7, linewidth=1, label=label)

        # Add labels for each point
        for i, row in pca_df.iterrows():
            plt.annotate(row['etiological_name'], 
                        (row['PC1'], row['PC2']),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=12)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.title(f'PCA: PC1 vs PC2 Scores\n{columns_numeric_selected}\n(Blue=Pandemic, Red=No Pandemic)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        
        try:
            plt.savefig(f'{dir_base}/PCA/figures/pca_plot.svg', format='svg', dpi=300, bbox_inches='tight')
            plt.savefig(f'{dir_base}/PCA/figures/pca_plot.png', format='png', dpi=300, bbox_inches='tight')
            print("Figuras PCA salvas com sucesso")
        except Exception as e:
            print(f"Erro ao salvar figuras PCA: {e}")
            
        plt.show()

        # Save PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=columns_numeric_selected
        )
        loadings.to_csv(f'{dir_base}/PCA/analysis/pca_loadings.tsv', sep='\t')
        
        print("Análise PCA concluída com sucesso!")

    except Exception as e:
        print(f"Erro durante análise PCA: {e}")
        import traceback
        traceback.print_exc()



    # 3. Correlation Matrix with Best Features
    try:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df_numeric_clean.corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Plot correlation matrix
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={'shrink': 0.8})

        plt.title('Correlation Matrix of Best Features', pad=20)
        plt.tight_layout()
        
        # Salvar matriz de correlação
        try:
            plt.savefig(f'{dir_base}/figures/correlation_matrix.svg', format='svg', dpi=300, bbox_inches='tight')
            plt.savefig(f'{dir_base}/figures/correlation_matrix.png', format='png', dpi=300, bbox_inches='tight')
            print("Matriz de correlação salva com sucesso")
        except Exception as e:
            print(f"Erro ao salvar matriz de correlação: {e}")
            
        plt.show()

        # Save correlation matrix
        correlation_matrix.to_excel(f'{dir_base}/tables/correlation_matrix.xlsx')
        correlation_matrix.to_csv(f'{dir_base}/tables/correlation_matrix.tsv', sep='\t')

    except Exception as e:
        print(f"Erro durante criação da matriz de correlação: {e}")

    print("\nAnálise completa! Verifique os diretórios criados para os resultados.")


def run_analysis_for_n_features(n_features, base_dir_name):
    
    # Create directories
    base_dir = base_dir_name
    os.makedirs(f'{base_dir}/models', exist_ok=True)
    os.makedirs(f'{base_dir}/figures', exist_ok=True)
    os.makedirs(f'{base_dir}/tables', exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR {n_features} FEATURES")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_excel('/home/anacleto/projects/andressa/data_original.xlsx')
    print("Dataset shape:", df.shape)
    
    # Data preprocessing - handle NaN values and binary columns properly
    df_processed = df.copy()
    
    # Identify and handle binary columns
    binary_columns = ['zoonotic_transmission', 'assintomatic_transmission', 
                      'permanent_immunity', 'approved_treatment', 
                      'approved_vaccine', 'pandemic']
                       
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
    numeric_columns = ['R0_min', 'R0_max', 'R0_amplitude', 'infectious_period_average_days', 
                       'lethality_rate', 'incubation_period_days_min', 'incubation_period_days_max', 
                       'incubation_period_days_amplitude', 'average_transmission_rate']
    
    # Check for NaN in numeric columns and impute with median
    for col in numeric_columns:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                print(f"Imputing {nan_count} missing values in {col} with median")
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
    
    # Create final numeric dataset - include both numeric and binary columns
    final_numeric_columns = [col for col in numeric_columns + binary_columns 
                            if col in df_processed.columns and col != 'pandemic']
    target_column = 'pandemic'
    
    df_numeric = df_processed[final_numeric_columns + [target_column]].copy()
    
    print(f"Final dataset for analysis: {df_numeric.shape}")
    
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
    
    # Get all possible feature combinations for the specified number of features
    all_features = X.columns.tolist()
    feature_combinations = []
    
    # Generate combinations only for the specified number of features
    if len(all_features) >= n_features:
        feature_combinations.extend(list(itertools.combinations(all_features, n_features)))
    
    print(f"Testing {len(feature_combinations)} feature combinations ({n_features} features)...")
    
    results = []
    best_combination = None
    best_score = float('inf')  # Now we want to MINIMIZE the difference
    
    # Test each combination with and without scaling
    for i, features in enumerate(feature_combinations):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(feature_combinations)}")
        
        try:
            X_subset = X[list(features)]
            
            # Ensure we have enough samples and features
            if len(X_subset) < 2 or X_subset.shape[1] < n_features:
                continue
                
            # Split data - THIS SPLIT WILL BE USED FOR BOTH CLASSICAL AND BAYESIAN MODELS
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Check if we have both classes in train and test
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            # Test WITHOUT scaling only
            lr_model_unscaled = LogisticRegression(random_state=42, max_iter=1000000)
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
                    'real_weights': lr_results_unscaled['real_weights'],
                    'use_scaling': lr_results_unscaled['use_scaling'],
                    # Store the train/test split for Bayesian model
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                })
                
                # Update best combination based on MINIMUM composite score (smallest differences)
                if lr_results_unscaled['composite_score'] < best_score:
                    best_score = lr_results_unscaled['composite_score']
                    best_combination = {
                        'features': features,
                        'model_type': 'LogisticRegression',
                        'scaling_method': 'Without Scaling',
                        'results': lr_results_unscaled,
                        'scaler': lr_results_unscaled['scaler'],
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'composite_score': lr_results_unscaled['composite_score'],
                        'coefficients': lr_results_unscaled['coefficients'],
                        'real_weights': lr_results_unscaled['real_weights'],
                        'use_scaling': lr_results_unscaled['use_scaling']
                    }
        except Exception as e:
            print(f"Error with features {features}: {e}")
            continue
    

    if not results and len(all_features) >= n_features:
        print("No valid feature combinations found. Trying with random features...")

        features = tuple(np.random.choice(all_features, n_features, replace=False))
        X_subset = X[list(features)]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42, stratify=y)
        

        lr_model_unscaled = LogisticRegression(random_state=42, max_iter=1000000)
        lr_results_unscaled = evaluate_model(X_train, X_test, y_train, y_test,
                                           lr_model_unscaled, 'LogisticRegression',
                                           use_scaling=False)
        
        if lr_results_unscaled['success']:
            best_combination = {
                'features': features,
                'model_type': 'LogisticRegression',
                'scaling_method': 'Without Scaling',
                'results': lr_results_unscaled,
                'scaler': lr_results_unscaled['scaler'],
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'composite_score': lr_results_unscaled['composite_score'],
                'coefficients': lr_results_unscaled['coefficients'],
                'real_weights': lr_results_unscaled['real_weights'],
                'use_scaling': lr_results_unscaled['use_scaling']
            }
            best_score = lr_results_unscaled['composite_score']
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
                'real_weights': lr_results_unscaled['real_weights'],
                'use_scaling': lr_results_unscaled['use_scaling'],
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
    
    if best_combination:
        print(f"\nBest combination for {n_features} features: {best_combination['features']}")
        print(f"Best model: {best_combination['model_type']}")
        print(f"Best scaling method: {best_combination['scaling_method']}")
        print(f"Best composite score (lower is better): {best_score:.4f}")
        print(f"ROC AUC - Train: {best_combination['results']['roc_auc_train']:.4f}, Test: {best_combination['results']['roc_auc_test']:.4f}")
        print(f"Accuracy - Train: {best_combination['results']['accuracy_train']:.4f}, Test: {best_combination['results']['accuracy_test']:.4f}")
        print(f"ROC AUC Difference: {best_combination['results']['roc_auc_diff']:.4f}")
        print(f"Accuracy Difference: {best_combination['results']['accuracy_diff']:.4f}")
        

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df.query('roc_auc_test > 0.7 and roc_auc_train > 0.7 and accuracy_test > 0.7 and accuracy_train > 0.7 and precision_test > 0.7 and precision_train > 0.7').sort_values(by=['roc_auc_test', 'roc_auc_train', 'accuracy_test', 'accuracy_train', 'precision_test', 'precision_train'], ascending=False).head(50)

            results_df = results_df.sort_values('composite_score', ascending=True)  
        else:

            results_df = pd.DataFrame([{
                'features': ', '.join(best_combination['features']),
                'num_features': len(best_combination['features']),
                'model_type': best_combination['model_type'],
                'scaling_method': best_combination['scaling_method'],
                'accuracy_test': best_combination['results']['accuracy_test'],
                'precision_test': best_combination['results']['precision_test'],
                'recall_test': best_combination['results']['recall_test'],
                'f1_test': best_combination['results']['f1_test'],
                'roc_auc_test': best_combination['results']['roc_auc_test'],
                'accuracy_train': best_combination['results']['accuracy_train'],
                'precision_train': best_combination['results']['precision_train'],
                'recall_train': best_combination['results']['recall_train'],
                'f1_train': best_combination['results']['f1_train'],
                'roc_auc_train': best_combination['results']['roc_auc_train'],
                'roc_auc_diff': best_combination['results']['roc_auc_diff'],
                'accuracy_diff': best_combination['results']['accuracy_diff'],
                'composite_score': best_combination['composite_score'],
                'feature_list': list(best_combination['features']),
                'coefficients': best_combination['coefficients'],
                'real_weights': best_combination['real_weights'],
                'use_scaling': best_combination['use_scaling'],
                'X_train': best_combination['X_train'],
                'X_test': best_combination['X_test'],
                'y_train': best_combination['y_train'],
                'y_test': best_combination['y_test']
            }])
    else:
        print(f"No valid models could be trained for {n_features} features. Please check your data.")

        results_df = pd.DataFrame()
        return None
    
    # Save results
    results_df.to_csv(f'{base_dir}/tables/feature_combination_results.tsv', sep='\t', index=False)
    results_df.to_excel(f'{base_dir}/tables/feature_combination_results.xlsx', index=False)
    
    # Plot results for best combination if available
    if best_combination:
        best_features = best_combination['features']
        best_model = best_combination['model_type']
        best_scaling = best_combination['scaling_method']
        best_results = best_combination['results']
        
        # ROC Curve for both train and test
        plt.figure(figsize=(10, 8))
        plt.plot(best_results['fpr_test'], best_results['tpr_test'], color='darkorange', lw=2, 
                 label=f'Test ROC (AUC = {best_results["roc_auc_test"]:.2f})')
        plt.plot(best_results['fpr_train'], best_results['tpr_train'], color='blue', lw=2, 
                 label=f'Train ROC (AUC = {best_results["roc_auc_train"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model} ({best_scaling})\nFeatures: {", ".join(best_features)}\nROC AUC Diff: {best_results["roc_auc_diff"]:.4f}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{base_dir}/figures/best_model_roc.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/best_model_roc.svg', bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix for test set
        cm_test = confusion_matrix(best_combination['y_test'], best_results['y_pred_test'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Test Set\n{best_model} ({best_scaling})')
        plt.savefig(f'{base_dir}/figures/best_model_confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/best_model_confusion_matrix_test.svg', bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix for training set
        cm_train = confusion_matrix(best_combination['y_train'], best_results['model'].predict(best_combination['X_train']))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Pandemic', 'Pandemic'],
                    yticklabels=['Non-Pandemic', 'Pandemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Training Set\n{best_model} ({best_scaling})')
        plt.savefig(f'{base_dir}/figures/best_model_confusion_matrix_train.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/best_model_confusion_matrix_train.svg', bbox_inches='tight')
        plt.close()
        
        # Training vs Test distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training distribution
        train_counts = best_combination['y_train'].value_counts()
        ax1.bar(['Non-Pandemic', 'Pandemic'], train_counts, color='green', alpha=0.7)
        ax1.set_title('Training Set Distribution')
        ax1.set_ylabel('Count')
        
        # Test distribution
        test_counts = best_combination['y_test'].value_counts()
        ax2.bar(['Non-Pandemic', 'Pandemic'], test_counts, color='gray', alpha=0.7)
        ax2.set_title('Test Set Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{base_dir}/figures/train_test_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/train_test_distribution.svg', bbox_inches='tight')
        plt.close()
        
        # Run PCA 
        # Only run if we have complete data for the selected features
        selected_features_complete = True
        for feature in best_combination['features']:
            if feature in binary_columns:
                # Binary features are already complete after preprocessing
                continue
            if feature in numeric_columns and df[feature].isna().any():
                selected_features_complete = False
                print(f"Skipping PCA/t-SNE for feature {feature} due to missing values")
                break
        
        if selected_features_complete:
            run_pca_tsne_analysis(dir_base=base_dir, 
                                  df_file='/home/anacleto/projects/andressa/data_original.xlsx',
                                  columns_numeric_selected=list(best_combination['features']), 
                                  columns_categorical_selected=binary_columns)
        else:
            print("Skipping PCA analysis due to missing values in selected features")
        
        # Save best model
        joblib.dump(best_results['model'], f'{base_dir}/models/best_classical_model.pkl')
        
        # Calculate confidence intervals for classical model
        print("Calculating confidence intervals for classical model...")
        X_for_ci = best_combination['X_train'].values
        
        classical_coef, classical_std, classical_ci_lower, classical_ci_upper = calculate_confidence_intervals_corrected(
            best_results['model'], X_for_ci, best_combination['y_train'].values
        )
        
        # Save model summary with real weights calculation and confidence intervals
        with open(f'{base_dir}/summary.txt', 'w') as f:
            f.write("CLASSICAL MODEL SUMMARY\n")
            f.write("======================\n\n")
            f.write(f"Best Feature Combination: {', '.join(best_features)}\n")
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
        print("BAYESIAN ANALYSIS USING SAME TRAIN/TEST SPLIT AS BEST CLASSICAL MODEL")
        print(f"{'='*80}")
        print(f"Best features from classical model: {best_features}")
        
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
        print(f"USING SAME FEATURES AS BEST CLASSICAL MODEL: {best_features}")
        
        bayesian_model = BayesianLogisticRegressionMultiChainCorrected(
            n_chains=3, n_iter=1050000, burn_in=50000, beta_prior_precision=1
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
        fpr_bayes_test, tpr_bayes_test, _ = roc_curve(y_bayes_test, y_pred_proba_bayes_test)
        roc_auc_bayes_test = auc(fpr_bayes_test, tpr_bayes_test)
        
        # Make predictions on training set for comparison
        y_pred_bayes_train = bayesian_model.predict(X_bayes_train_with_intercept)
        y_pred_proba_bayes_train = bayesian_model.predict_proba(X_bayes_train_with_intercept)
        
        accuracy_bayes_train = accuracy_score(y_bayes_train, y_pred_bayes_train)
        precision_bayes_train = precision_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        recall_bayes_train = recall_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        f1_bayes_train = f1_score(y_bayes_train, y_pred_bayes_train, zero_division=0)
        fpr_bayes_train, tpr_bayes_train, _ = roc_curve(y_bayes_train, y_pred_proba_bayes_train)
        roc_auc_bayes_train = auc(fpr_bayes_train, tpr_bayes_train)
        
        # Calculate differences for Bayesian model
        roc_auc_diff_bayes = abs(roc_auc_bayes_train - roc_auc_bayes_test)
        accuracy_diff_bayes = abs(accuracy_bayes_train - accuracy_bayes_test)
        composite_score_bayes = (roc_auc_diff_bayes * 0.7 + accuracy_diff_bayes * 0.3)
        
        # Save Bayesian model
        joblib.dump(bayesian_model, f'{base_dir}/models/best_bayesian_model.pkl')
        
        # Compute HPD intervals from combined chains
        hpd_intervals = []
        for i in range(bayesian_model.p):
            hpd_lower, hpd_upper = bayesian_model.hpd(bayesian_model.beta_post[:, i])
            hpd_intervals.append((hpd_lower, hpd_upper))
        
        # Calculate classical model coefficients on original scale for comparison
        classical_coef_full_original = np.concatenate([[best_results['model'].intercept_[0]], best_combination['coefficients']])
        
        # Save Bayesian results to summary with detailed comparison
        with open(f'{base_dir}/summary.txt', 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARISON: CLASSICAL vs BAYESIAN LOGISTIC REGRESSION\n")
            f.write("USING THE SAME TRAIN/TEST SPLIT AND FEATURES\n")
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
            f.write(f"Features used in both models: {', '.join(best_features)}\n\n")
            
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
            
            f.write("\n\nBAYESIAN LOGISTIC REGRESSION DETAILS (MULTIPLE CHAINS)\n")
            f.write("="*60 + "\n\n")
            f.write("IMPORTANT: Model fitted on THE SAME TRAINING DATA as classical model\n")
            f.write("Coefficients are directly interpretable in original units\n")
            f.write(f"USING SAME FEATURES AS BEST CLASSICAL MODEL: {', '.join(best_features)}\n\n")
            
            f.write("Convergence Diagnostics:\n")
            f.write(f"Number of chains: {bayesian_model.n_chains}\n")
            f.write(f"Number of iterations per chain: {bayesian_model.n_iter}\n")
            f.write(f"Burn-in period: {bayesian_model.burn_in}\n")
            f.write(f"Effective sample size (combined): {len(bayesian_model.beta_post)}\n\n")
            
            f.write("Gelman-Rubin Diagnostics (R-hat):\n")
            f.write("Values close to 1.0 indicate good convergence\n")
            for i, name in enumerate(feature_names):
                if i < len(bayesian_model.r_hat_beta):
                    f.write(f"  {name}: {bayesian_model.r_hat_beta[i]:.4f}\n")
            f.write("\n")
            
            f.write("Detailed Bayesian Parameter Estimates (combined chains) - ORIGINAL SCALE:\n")
            for i, name in enumerate(feature_names):
                if i < len(bayesian_model.beta_mean):
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {bayesian_model.beta_mean[i]:.4f}\n")
                    f.write(f"  Median: {bayesian_model.beta_median[i]:.4f}\n")
                    f.write(f"  Std: {bayesian_model.beta_std[i]:.4f}\n")
                    f.write(f"  95% HPD: [{hpd_intervals[i][0]:.4f}, {hpd_intervals[i][1]:.4f}]\n")
                    f.write("\n")
        
        # Plot comparison ROC curves
        plt.figure(figsize=(10, 8))
        # Classical model
        plt.plot(best_results['fpr_test'], best_results['tpr_test'], color='darkorange', lw=2, 
                 label=f'Classical Test ROC (AUC = {best_results["roc_auc_test"]:.2f})')
        # Bayesian model
        plt.plot(fpr_bayes_test, tpr_bayes_test, color='green', lw=2, 
                 label=f'Bayesian Test ROC (AUC = {roc_auc_bayes_test:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison - Same Test Set\nFeatures: {", ".join(best_features)}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{base_dir}/figures/classical_vs_bayesian_roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/classical_vs_bayesian_roc_comparison.svg', bbox_inches='tight')
        plt.close()
        
        # Plot Markov chains and distributions for MULTIPLE CHAINS
        feature_names = ['Intercept'] + list(best_features)
        
        # Create subplots for chains (showing all chains) and distributions
        fig, axes = plt.subplots(len(feature_names), 2, figsize=(15, 5*len(feature_names)))
        
        # Colors for different chains
        chain_colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, name in enumerate(feature_names):
            if i >= len(feature_names):
                break
                
            # Markov chains from all chains
            for chain_id in range(bayesian_model.n_chains):
                axes[i, 0].plot(bayesian_model.chains_beta[chain_id][:, i], 
                               alpha=0.7, color=chain_colors[chain_id % len(chain_colors)],
                               label=f'Chain {chain_id+1}' if i == 0 else "")
            
            axes[i, 0].axvline(x=bayesian_model.burn_in, color='black', linestyle='--', label='Burn-in')
            axes[i, 0].set_title(f'Markov Chains: {name}\nR-hat: {bayesian_model.r_hat_beta[i]:.4f}')
            axes[i, 0].set_xlabel('Iteration')
            axes[i, 0].set_ylabel('Parameter Value')
            if i == 0:
                axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Posterior distribution (combined chains)
            axes[i, 1].hist(bayesian_model.beta_post[:, i], bins=50, density=True, alpha=0.7, color='skyblue')
            axes[i, 1].axvline(bayesian_model.beta_mean[i], color='red', linestyle='-', label='Mean')
            axes[i, 1].axvline(bayesian_model.beta_median[i], color='orange', linestyle='--', label='Median')
            axes[i, 1].axvline(hpd_intervals[i][0], color='green', linestyle=':', label='95% HPD')
            axes[i, 1].axvline(hpd_intervals[i][1], color='green', linestyle=':', label='_nolegend_')
            axes[i, 1].set_title(f'Posterior Distribution: {name}\n(Combined Chains - Original Scale)')
            axes[i, 1].set_xlabel('Parameter Value (Original Scale)')
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{base_dir}/figures/bayesian_chains_distributions_original_scale.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_dir}/figures/bayesian_chains_distributions_original_scale.svg', bbox_inches='tight')
        plt.close()
        
        # Create comparison matrix for coefficients
        comparison_data = []
        feature_names = ['Intercept'] + list(best_features)
        
        for i, name in enumerate(feature_names):
            if i < len(classical_coef_full_original) and i < len(bayesian_model.beta_mean):
                comparison_data.append({
                    'Parameter': name,
                    'Classical_Coefficient': round(classical_coef_full_original[i], 4),
                    'Classical_CI95_Lower': round(classical_ci_lower[i], 4) if i < len(classical_ci_lower) else 'N/A',
                    'Classical_CI95_Upper': round(classical_ci_upper[i], 4) if i < len(classical_ci_upper) else 'N/A',
                    'Bayesian_Mean': round(bayesian_model.beta_mean[i], 4),
                    'Bayesian_HPD95_Lower': round(hpd_intervals[i][0], 4),
                    'Bayesian_HPD95_Upper': round(hpd_intervals[i][1], 4)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison matrix
        comparison_df.to_csv(f'{base_dir}/tables/coefficient_comparison_matrix.tsv', sep='\t', index=False)
        comparison_df.to_excel(f'{base_dir}/tables/coefficient_comparison_matrix.xlsx', index=False)
        
        # Create formatted comparison matrix for summary
        with open(f'{base_dir}/tables/formatted_comparison_matrix.tsv', 'w') as f:
            f.write("Parameter\tClassical_Coefficient\tClassical_CI95\tBayesian_Mean\tBayesian_HPD95\n")
            for _, row in comparison_df.iterrows():
                classical_ci = f"[{row['Classical_CI95_Lower']}, {row['Classical_CI95_Upper']}]"
                bayesian_hpd = f"[{row['Bayesian_HPD95_Lower']}, {row['Bayesian_HPD95_Upper']}]"
                f.write(f"{row['Parameter']}\t{row['Classical_Coefficient']}\t{classical_ci}\t{row['Bayesian_Mean']}\t{bayesian_hpd}\n")
        
        return {
            'best_features': best_features,
            'classical_model': best_results['model'],
            'bayesian_model': bayesian_model,
            'scaler': None,  # No scaling used
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
            }
        }
    return None

if __name__ == "__main__":
            
    # Executar para 5 features
    print("Starting analysis for 5 features...")
    result_5 = run_analysis_for_n_features(5, 'analysis/five_best_descriptors')
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Results saved in:")
    print("- analysis/five_best_descriptors/")
    print(f"{'='*80}")
