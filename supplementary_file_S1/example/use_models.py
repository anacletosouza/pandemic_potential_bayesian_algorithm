import pandas as pd
import numpy as np
import joblib
import warnings
import sys
warnings.filterwarnings('ignore')

# Adicionar o caminho onde está o arquivo MCMC.py ao Python path
sys.path.append('/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/bayesian_modeling_results')

from MCMC import BayesianLogisticRegressionMultiChainCorrected


def load_bayesian_model_and_predict(test_file_path, model_path, output_file='predictions_results.xlsx'):
    """
    Load saved Bayesian model and make predictions on new strains
    """
    
    print("Loading Bayesian model...")
    try:
        bayesian_model = joblib.load(model_path)
        print(" Bayesian model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        return None
    
    print("\nLoading test data...")
    try:
        test_df = pd.read_excel(test_file_path)
        print(f" Test data loaded: {test_df.shape}")
        print(f"Columns: {test_df.columns.tolist()}")
    except Exception as e:
        print(f" Error loading test data: {e}")
        return None
    
    # DEFINIR AS FEATURES EXATAS USADAS NO TREINAMENTO
    expected_features = [
        'R0_min',  # R0 (min)
        'infectious_period_average_days',  # Infectious period (average days)
        'lethality_rate',  # Lethality rate
        'incubation_period_days_min',  # Incubation period (days minimum)
        'permanent_immunity'  # Permanent immunity
    ]
    
    print(f"\nUsing EXACT features from model training:")
    for i, feature in enumerate(expected_features, 1):
        print(f"  {i}. {feature}")
    
    print("\nPreprocessing test data...")
    test_data = test_df.copy()
    
    # Verificar features faltantes
    missing_features = [f for f in expected_features if f not in test_data.columns]
    if missing_features:
        print(f" ERROR: Missing required features: {missing_features}")
        print(f" Available features: {test_data.columns.tolist()}")
        return None
    
    # Processar coluna binária permanent_immunity
    if 'permanent_immunity' in test_data.columns:
        test_data['permanent_immunity'] = test_data['permanent_immunity'].fillna('no')
        test_data['permanent_immunity'] = test_data['permanent_immunity'].apply(
            lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true', 'y'] else 0
        )
        print(f" Processed binary column: permanent_immunity")
    
    # Preencher valores NaN nas colunas numéricas
    numeric_columns = ['R0_min', 'infectious_period_average_days', 'lethality_rate', 'incubation_period_days_min']
    
    for col in numeric_columns:
        if col in test_data.columns:
            nan_count = test_data[col].isna().sum()
            if nan_count > 0:
                median_val = test_data[col].median()
                test_data[col] = test_data[col].fillna(median_val)
                print(f" Imputed {nan_count} missing values in {col} with median: {median_val:.4f}")
    
    # Selecionar APENAS as features usadas no treinamento
    X_test = test_data[expected_features]
    print(f" Final feature set shape: {X_test.shape}")
    
    # Verificar se temos os valores esperados
    print(f"\nFeature values summary:")
    for feature in expected_features:
        if feature in X_test.columns:
            print(f"  {feature}: min={X_test[feature].min():.4f}, max={X_test[feature].max():.4f}, mean={X_test[feature].mean():.4f}")
    
    # Preparar dados para o modelo Bayesiano (adicionar intercept manualmente)
    X_test_with_intercept = np.column_stack([np.ones(X_test.shape[0]), X_test.values])
    print(f" Data with intercept shape: {X_test_with_intercept.shape}")
    print(f" Expected: 6 features total (1 intercept + 5 features)")
    
    print("\nMaking predictions...")
    
    try:
        # Fazer predições usando o modelo Bayesiano
        # Para modelos Bayesianos personalizados, geralmente usamos multiplicação manual
        if hasattr(bayesian_model, 'beta_mean'):
            print("Using Bayesian model coefficients for prediction...")
            
            # Verificar dimensões dos coeficientes
            print(f"Model coefficients shape: {bayesian_model.beta_mean.shape}")
            print(f"Model coefficients: {bayesian_model.beta_mean}")
            
            # Calcular logits: X * beta
            logits = X_test_with_intercept @ bayesian_model.beta_mean
            print(f"Logits calculated: {logits}")
            
            # Converter logits para probabilidades (função sigmoid)
            probabilities = 1 / (1 + np.exp(-logits))
            predictions = (probabilities > 0.5).astype(int)
            
            print(" Predictions completed successfully using manual Bayesian method")
            
        else:
            print("Using standard predict methods...")
            probabilities = bayesian_model.predict_proba(X_test_with_intercept)
            predictions = bayesian_model.predict(X_test_with_intercept)
        
        # Criar dataframe de resultados
        results_df = test_data.copy()
        results_df['pandemic_probability'] = probabilities
        results_df['pandemic_prediction'] = predictions
        
        # Garantir que probabilidades sejam array 1D
        if len(results_df['pandemic_probability'].shape) > 1:
            results_df['pandemic_probability'] = results_df['pandemic_probability'][:, 1]
        
        # Categorizar risco
        results_df['pandemic_risk_category'] = results_df['pandemic_probability'].apply(
            lambda x: 'High Risk' if x >= 0.7 else 'Medium Risk' if x >= 0.3 else 'Low Risk'
        )
        
        results_df['interpretation'] = results_df.apply(
            lambda row: f"Pandemic potential: {row['pandemic_probability']:.1%} - {row['pandemic_risk_category']}",
            axis=1
        )
        
        # Mostrar resumo
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total strains analyzed: {len(results_df)}")
        print(f"High risk strains (≥70%): {len(results_df[results_df['pandemic_probability'] >= 0.7])}")
        print(f"Medium risk strains (30-70%): {len(results_df[(results_df['pandemic_probability'] >= 0.3) & (results_df['pandemic_probability'] < 0.7)])}")
        print(f"Low risk strains (<30%): {len(results_df[results_df['pandemic_probability'] < 0.3])}")
        
        # Mostrar predições detalhadas
        print(f"\n{'='*60}")
        print("DETAILED PREDICTIONS")
        print(f"{'='*60}")
        
        display_columns = []
        name_columns = ['etiological_name', 'disease_name']
        for col in name_columns:
            if col in results_df.columns:
                display_columns.append(col)
                break
        
        display_columns.extend(['pandemic_probability', 'pandemic_risk_category'])
        
        # Header
        header_line = ""
        for col in display_columns:
            if col == 'pandemic_probability':
                header_line += f"{'Probability':<15} "
            elif col == 'pandemic_risk_category':
                header_line += f"{'Risk':<12} "
            else:
                header_line += f"{str(col):<25} "
        print(header_line)
        print("-" * len(header_line))
        
        # Data
        for _, row in results_df.iterrows():
            data_line = ""
            for col in display_columns:
                if col == 'pandemic_probability':
                    data_line += f"{row[col]:<15.3f} "
                else:
                    data_line += f"{str(row[col]):<25} "
            print(data_line)
        
        # Salvar resultados
        try:
            results_df.to_excel(output_file, index=False)
            print(f"\n Predictions saved to: {output_file}")
            
            # Salvar versão simplificada
            simplified_cols = display_columns.copy()
            results_df[simplified_cols].to_excel('simplified_predictions.xlsx', index=False)
            print(f" Simplified results saved to: simplified_predictions.xlsx")
            
        except Exception as e:
            print(f" Error saving results: {e}")
        
        return results_df
        
    except Exception as e:
        print(f" Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_feature_importance(model_path):
    """Mostrar importância das features do modelo"""
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'beta_mean'):
            print(f"\n{'='*60}")
            print("MODEL FEATURE IMPORTANCE")
            print(f"{'='*60}")
            
            features = ['Intercept', 'R0_min', 'infectious_period_average_days', 
                       'lethality_rate', 'incubation_period_days_min', 'permanent_immunity']
            
            for i, (feature, coef) in enumerate(zip(features, model.beta_mean)):
                effect = "increases" if coef > 0 else "decreases"
                print(f"  {feature:<30}: {coef:>8.4f} ({effect} pandemic risk)")
                
    except Exception as e:
        print(f" Could not display feature importance: {e}")

if __name__ == "__main__":
    TEST_FILE_PATH = "/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/example/test_data.xlsx"
    MODEL_PATH = "/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S2/classical_bayesian_models/models/best_bayesian_model.pkl"
    
    print("BAYESIAN MODEL PREDICTION FOR NEW STRAINS")
    print("=" * 50)
    
    # Mostrar importância das features primeiro
    show_feature_importance(MODEL_PATH)
    
    # Fazer predições
    results = load_bayesian_model_and_predict(TEST_FILE_PATH, MODEL_PATH)
    
    if results is not None:
        print(f"\n{'='*60}")
        print("PROCESS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("PROCESS FAILED")
        print(f"{'='*60}")
