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

def calculate_sars_cov2_probability(model, features, feature_list, use_scaling=False, scaler=None):
    """Calculate the probability of SARS-CoV-2 being pandemic using the model"""
    try:
        # Create a row for SARS-CoV-2 with the selected features
        sars_cov2_data = {}
        
        # These are example values for SARS-CoV-2 - you should replace with actual values from your dataset
        sars_cov2_characteristics = {
            'R0_min': 2.5, 'R0_max': 3.0, 'R0_amplitude': 0.5,
            'infectious_period_average_days': 10, 'lethality_rate': 0.02,
            'incubation_period_days_min': 5, 'incubation_period_days_max': 14,
            'incubation_period_days_amplitude': 9, 'average_transmission_rate': 0.3,
            'zoonotic_transmission': 1, 'assintomatic_transmission': 1,
            'permanent_immunity': 0, 'approved_treatment': 0, 'approved_vaccine': 1
        }
        
        for feature in feature_list:
            if feature in sars_cov2_characteristics:
                sars_cov2_data[feature] = sars_cov2_characteristics[feature]
            else:
                sars_cov2_data[feature] = 0  # Default value for missing features
        
        # Create DataFrame for SARS-CoV-2
        sars_cov2_df = pd.DataFrame([sars_cov2_data])
        
        # Apply scaling if the model was trained with scaling
        if use_scaling and scaler is not None:
            sars_cov2_processed = scaler.transform(sars_cov2_df)
        else:
            sars_cov2_processed = sars_cov2_df.values
        
        # Predict probability
        probability = model.predict_proba(sars_cov2_processed)[0, 1]
        
        return probability
    except Exception as e:
        print(f"Error calculating SARS-CoV-2 probability: {e}")
        return 0.0

def prepare_features_with_dummies(X, required_features):
    """
    Prepare feature matrix with required features, creating dummy values for missing features
    """
    available_features = [f for f in required_features if f in X.columns]
    missing_features = [f for f in required_features if f not in X.columns]
    
    print(f"Available features: {available_features}")
    print(f"Missing features: {missing_features}")
    
    # Create DataFrame with available features
    X_prepared = X[available_features].copy()
    
    # Add dummy columns for missing features (filled with 0)
    for feature in missing_features:
        X_prepared[feature] = 0
        print(f"Added dummy column for {feature} with value 0")
    
    # Ensure the columns are in the correct order
    X_prepared = X_prepared[required_features]
    
    return X_prepared, available_features, missing_features

def load_and_test_model(model_path, data_file, feature_list):
    """
    Load a pre-trained model and test it on the original dataset
    """
    print(f"Loading model from: {model_path}")
    
    # Load the pre-trained model
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # Check model type and expected features
        if hasattr(model, 'coef_'):
            print(f"Model type: Classical Logistic Regression")
            expected_features = len(model.coef_[0])
        elif hasattr(model, 'beta_mean'):
            print(f"Model type: Bayesian Logistic Regression")
            expected_features = len(model.beta_mean) - 1  # Subtract intercept
        else:
            print(f"Model type: Unknown")
            expected_features = len(feature_list)
            
        print(f"Model expects {expected_features} features")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load and preprocess the original dataset
    print("Loading and preprocessing original dataset...")
    df = pd.read_excel(data_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
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
    
    print(f"Final dataset for inference: {df_numeric.shape}")
    
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
    
    # Prepare features with dummy values for missing ones
    X_prepared, available_features, missing_features = prepare_features_with_dummies(X, feature_list)
    
    print(f"Final prepared feature matrix shape: {X_prepared.shape}")
    print(f"Feature order: {X_prepared.columns.tolist()}")
    
    # Make predictions using the loaded model
    print("Making predictions with loaded model...")
    
    try:
        # For classical models
        if hasattr(model, 'predict_proba'):
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_prepared.values)[:, 1]
            y_pred = model.predict(X_prepared.values)
        # For Bayesian models
        elif hasattr(model, 'beta_mean'):
            # Bayesian models need intercept added
            X_with_intercept = np.column_stack([np.ones(len(X_prepared)), X_prepared.values])
            y_pred_proba = model.predict_proba(X_with_intercept)
            y_pred = model.predict(X_with_intercept)
        else:
            raise ValueError("Unknown model type - cannot make predictions")
        
        # Calculate performance metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate SARS-CoV-2 probability
        sars_cov2_prob = calculate_sars_cov2_probability(
            model, 
            feature_list, 
            feature_list,
            use_scaling=False,
            scaler=None
        )
        
        # Print results
        print("\n" + "="*80)
        print("MODEL INFERENCE RESULTS")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Required features: {feature_list}")
        print(f"Available features: {available_features}")
        print(f"Missing features (set to 0): {missing_features}")
        print(f"Dataset size: {X_prepared.shape[0]} samples")
        print(f"Performance Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  SARS-CoV-2 Pandemic Probability: {sars_cov2_prob:.4f}")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'actual': y.values,
            'predicted': y_pred,
            'probability': y_pred_proba
        })
        
        # Add feature values for analysis
        for feature in feature_list:
            if feature in X_prepared.columns:
                results_df[feature] = X_prepared[feature].values
            else:
                results_df[feature] = 0  # For missing features
        
        return {
            'model': model,
            'features': feature_list,
            'available_features': available_features,
            'missing_features': missing_features,
            'X': X_prepared,
            'y': y,
            'predictions': results_df,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'sars_cov2_probability': sars_cov2_prob
            }
        }
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_inference_with_loaded_models():
    """
    Main function to load pre-trained models and run inference
    """
    # Define model paths and their corresponding features
    models_config = {
        'classical_model': {
            'path': '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S2/classical_bayesian_models/models/best_classical_model.pkl',
            'features': ['R0_min', 'infectious_period_average_days', 'lethality_rate', 'zoonotic_transmission', 'assintomatic_transmission']
        },
        'bayesian_model': {
            'path': '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S2/classical_bayesian_models/models/best_bayesian_model.pkl', 
            'features': ['R0_min', 'infectious_period_average_days', 'lethality_rate', 'zoonotic_transmission', 'assintomatic_transmission']
        }
    }
    
    data_file = '/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/dataset/data_original.xlsx'
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"LOADING AND TESTING {model_name.upper()}")
        print(f"{'='*80}")
        
        result = load_and_test_model(
            model_path=config['path'],
            data_file=data_file,
            feature_list=config['features']
        )
        
        if result is not None:
            results[model_name] = result
            
            # Save predictions to file
            output_file = f"/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S2/classical_bayesian_models/models/{model_name}_predictions.csv"
            result['predictions'].to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")
    
    # Compare models if both were loaded successfully
    if 'classical_model' in results and 'bayesian_model' in results:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        classical_metrics = results['classical_model']['metrics']
        bayesian_metrics = results['bayesian_model']['metrics']
        
        print(f"{'Metric':<25} {'Classical':<12} {'Bayesian':<12}")
        print(f"{'-'*50}")
        print(f"{'Accuracy':<25} {classical_metrics['accuracy']:<12.4f} {bayesian_metrics['accuracy']:<12.4f}")
        print(f"{'Precision':<25} {classical_metrics['precision']:<12.4f} {bayesian_metrics['precision']:<12.4f}")
        print(f"{'Recall':<25} {classical_metrics['recall']:<12.4f} {bayesian_metrics['recall']:<12.4f}")
        print(f"{'F1-Score':<25} {classical_metrics['f1']:<12.4f} {bayesian_metrics['f1']:<12.4f}")
        print(f"{'ROC AUC':<25} {classical_metrics['roc_auc']:<12.4f} {bayesian_metrics['roc_auc']:<12.4f}")
        print(f"{'SARS-CoV-2 Probability':<25} {classical_metrics['sars_cov2_probability']:<12.4f} {bayesian_metrics['sars_cov2_probability']:<12.4f}")
        
        # Print feature availability information
        print(f"\nFeature Availability:")
        print(f"Available features: {results['classical_model']['available_features']}")
        print(f"Missing features (set to 0): {results['classical_model']['missing_features']}")
    
    return results

if __name__ == "__main__":
    # Run inference with loaded models
    results = run_inference_with_loaded_models()
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE!")
    print(f"{'='*80}")
