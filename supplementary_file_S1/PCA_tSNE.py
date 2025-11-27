import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 32

def run_pca_tsne_analysis(dir_base="/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/multivariable_analysis", 
                          df_file='/home/anacleto/projects/andressa/pandemic_potential_bayesian_algorithm/supplementary_file_S1/dataset/data_original.xlsx'):
    
    # Set font size
    plt.rcParams['font.size'] = 24

    # Create necessary directories
    base_dirs = [
        f'{dir_base}/PCA/figures', 
        f'{dir_base}/PCA/analysis',
        f'{dir_base}/tSNE/figures',
        f'{dir_base}/tSNE/analysis',
        f'{dir_base}/analysis',
        f'{dir_base}/figures', 
        f'{dir_base}/tables'    
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created/verified: {dir_path}")

    # Read the data
    try:
        df = pd.read_excel(df_file)
        print(f"Data loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Display basic info about the data
    print("Data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # FIXED COLUMNS SPECIFIED
    fixed_columns = ['disease_name', 'etiological_name', 'R0_min', 'infectious_period_average_days', 
                    'lethality_rate', 'incubation_period_days_min', 'permanent_immunity', 'pandemic']
    
    # Verify if all columns exist in the dataframe
    columns_numeric_selected = ['R0_min', 'infectious_period_average_days', 'lethality_rate', 
                               'incubation_period_days_min']
    
    print(f"Using fixed columns: {fixed_columns}")
    print(f"Using numeric columns: {columns_numeric_selected}")

    # 1. PCA Analysis with FIXED FEATURES
    print(f"\nRunning PCA with fixed features: {columns_numeric_selected}")
    
    # Filter out non-numeric values and handle missing values more robustly
    df_numeric = df[columns_numeric_selected].copy()
    
    # Convert to numeric, forcing errors to NaN
    for col in columns_numeric_selected:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Check for missing values and handle them
    print(f"Missing values in each feature:")
    for col in columns_numeric_selected:
        missing_count = df_numeric[col].isna().sum()
        print(f"  {col}: {missing_count} missing values")
    
    # Remove rows with any missing values in the selected features
    df_numeric_clean = df_numeric.dropna()
    print(f"Original data shape: {df_numeric.shape}")
    print(f"Clean data shape (after removing missing values): {df_numeric_clean.shape}")
    
    # Check if we have enough data for PCA
    if len(df_numeric_clean) < 2:
        print(f"WARNING: Not enough samples ({len(df_numeric_clean)}) for PCA analysis.")
        print("Skipping PCA and t-SNE analysis due to insufficient data.")
        return
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric_clean)

        # Perform PCA
        pca = PCA()
        pca_scores = pca.fit_transform(df_scaled)

        # Create PCA dataframe with all necessary columns
        pca_df = pd.DataFrame({
            'PC1': pca_scores[:, 0],
            'PC2': pca_scores[:, 1],
            'PC3': pca_scores[:, 2] if pca_scores.shape[1] > 2 else np.zeros(len(pca_scores)),
            'PC4': pca_scores[:, 3] if pca_scores.shape[1] > 3 else np.zeros(len(pca_scores)),
            'PC5': pca_scores[:, 4] if pca_scores.shape[1] > 4 else np.zeros(len(pca_scores)),
            'etiological_name': df.loc[df_numeric_clean.index, 'etiological_name'],
            'disease_name': df.loc[df_numeric_clean.index, 'disease_name'],
            'pandemic': df.loc[df_numeric_clean.index, 'pandemic']
        })

        # Save PCA scores
        pca_df.to_csv(f'{dir_base}/PCA/analysis/pca_scores.tsv', sep='\t', index=False)

        # Create PCA plot with color coding for pandemic status and black borders
        plt.figure(figsize=(15, 10))

        # Create color map for pandemic status
        colors = {'yes': 'blue', 'no': 'red'}
        pca_df['color'] = pca_df['pandemic'].map(colors)

        # Create scatter plot with different colors for pandemic vs no pandemic and black borders
        for pandemic_status in ['yes', 'no']:
            mask = pca_df['pandemic'] == pandemic_status
            if len(pca_df[mask]) > 0:  # Only plot if there are samples
                plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                           c=pca_df.loc[mask, 'color'], edgecolor='black', s=100, alpha=0.7, linewidth=1)

        # Add labels for each point
        for i, row in pca_df.iterrows():
            plt.annotate(row['etiological_name'], 
                        (row['PC1'], row['PC2']),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=16)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.title(f'PCA: PC1 vs PC2 Scores using Fixed Features')
        plt.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='blue', markersize=10, label='Pandemic', markeredgecolor='black'),
                          plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', markersize=10, label='No Pandemic', markeredgecolor='black')]
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        
        # Save figures
        try:
            plt.savefig(f'{dir_base}/PCA/figures/pca_plot.svg', format='svg', dpi=300, bbox_inches='tight')
            print("PCA SVG figure saved successfully")
        except Exception as e:
            print(f"Error saving PCA SVG: {e}")
        
        try:
            plt.savefig(f'{dir_base}/PCA/figures/pca_plot.png', format='png', dpi=300, bbox_inches='tight')
            print("PCA PNG figure saved successfully")
        except Exception as e:
            print(f"Error saving PCA PNG: {e}")
            
        plt.show()

        # Save PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=columns_numeric_selected
        )
        loadings.to_csv(f'{dir_base}/PCA/analysis/pca_loadings.tsv', sep='\t')

    except Exception as e:
        print(f"Error during PCA analysis: {e}")

    # 2. t-SNE Analysis with FIXED FEATURES
    print(f"\nRunning t-SNE with fixed features: {columns_numeric_selected}")
    
    try:
        # Prepare data for t-SNE using the same cleaned data
        df_top = df_numeric_clean[columns_numeric_selected]
        
        # Standardize the data for t-SNE
        scaler_tsne = StandardScaler()
        df_top_scaled = scaler_tsne.fit_transform(df_top)
        
        # Run t-SNE with appropriate perplexity
        perplexity_val = min(5, len(df_top_scaled) - 1)
        if perplexity_val < 1:
            print("Not enough samples for t-SNE. Skipping t-SNE analysis.")
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
            tsne_results = tsne.fit_transform(df_top_scaled)
            
            # Create t-SNE dataframe with all necessary columns
            tsne_df = pd.DataFrame({
                'tSNE1': tsne_results[:, 0],
                'tSNE2': tsne_results[:, 1],
                'etiological_name': df.loc[df_top.index, 'etiological_name'],
                'disease_name': df.loc[df_top.index, 'disease_name'],
                'pandemic': df.loc[df_top.index, 'pandemic']
            })
            
            # Save t-SNE results
            tsne_df.to_csv(f'{dir_base}/tSNE/analysis/tsne_fixed_features_scores.tsv', sep='\t', index=False)
            
            # Create t-SNE plot with black borders
            plt.figure(figsize=(15, 10))
            
            # Create color map for pandemic status
            colors = {'yes': 'blue', 'no': 'red'}
            tsne_df['color'] = tsne_df['pandemic'].map(colors)
            
            # Create scatter plot with black borders
            for pandemic_status in ['yes', 'no']:
                mask = tsne_df['pandemic'] == pandemic_status
                if len(tsne_df[mask]) > 0:  # Only plot if there are samples
                    plt.scatter(tsne_df.loc[mask, 'tSNE1'], tsne_df.loc[mask, 'tSNE2'], 
                               c=tsne_df.loc[mask, 'color'], edgecolor='black', s=100, alpha=0.7, linewidth=1)
            
            # Add labels for each point
            for i, row in tsne_df.iterrows():
                plt.annotate(row['etiological_name'], 
                            (row['tSNE1'], row['tSNE2']),
                            xytext=(5, 5), 
                            textcoords='offset points',
                            fontsize=16)
            
            plt.xlabel('tSNE1')
            plt.ylabel('tSNE2')
            plt.title(f't-SNE with Fixed Features')
            plt.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='blue', markersize=10, label='Pandemic', markeredgecolor='black'),
                              plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='red', markersize=10, label='No Pandemic', markeredgecolor='black')]
            plt.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            
            # Save t-SNE figures
            try:
                plt.savefig(f'{dir_base}/tSNE/figures/tsne_fixed_features_plot.svg', format='svg', dpi=300, bbox_inches='tight')
                print("t-SNE SVG figure saved successfully")
            except Exception as e:
                print(f"Error saving t-SNE SVG: {e}")
            
            try:
                plt.savefig(f'{dir_base}/tSNE/figures/tsne_fixed_features_plot.png', format='png', dpi=300, bbox_inches='tight')
                print("t-SNE PNG figure saved successfully")
            except Exception as e:
                print(f"Error saving t-SNE PNG: {e}")
                
            plt.show()

    except Exception as e:
        print(f"Error during t-SNE analysis: {e}")

    # 3. Correlation Matrix with Fixed Features
    try:
        plt.rcParams['font.size'] = 12
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

        plt.title('Correlation Matrix of Fixed Features', pad=20)
        plt.tight_layout()
        
        # Use correct paths for correlation matrix
        try:
            plt.savefig(f'{dir_base}/figures/correlation_matrix.svg', format='svg', dpi=300, bbox_inches='tight')
            print("Correlation matrix SVG saved successfully")
        except Exception as e:
            print(f"Error saving correlation matrix SVG: {e}")
        
        try:
            plt.savefig(f'{dir_base}/figures/correlation_matrix.png', format='png', dpi=300, bbox_inches='tight')
            print("Correlation matrix PNG saved successfully")
        except Exception as e:
            print(f"Error saving correlation matrix PNG: {e}")
            
        plt.show()

        # Save correlation matrix
        correlation_matrix.to_excel(f'{dir_base}/tables/correlation_matrix.xlsx')
        correlation_matrix.to_csv(f'{dir_base}/tables/correlation_matrix.tsv', sep='\t')

    except Exception as e:
        print(f"Error during correlation matrix creation: {e}")

    # 4. Additional Analysis for Pandemic Potential
    print("\n" + "="*50)
    print("KEY INSIGHTS FOR PANDEMIC POTENTIAL")
    print("="*50)

    try:
        # Calculate correlations with pandemic status
        df_analysis = df.copy()
        df_analysis['pandemic_numeric'] = df_analysis['pandemic'].map({'yes': 1, 'no': 0})
        
        pandemic_correlations = {}
        for col in columns_numeric_selected:
            if col in df_analysis.columns:
                # Use only complete cases for correlation
                temp_df = df_analysis[[col, 'pandemic_numeric']].dropna()
                if len(temp_df) > 1:  # Need at least 2 samples for correlation
                    corr = temp_df[col].corr(temp_df['pandemic_numeric'])
                    pandemic_correlations[col] = corr

        pandemic_corr_df = pd.DataFrame.from_dict(pandemic_correlations, 
                                                 orient='index', 
                                                 columns=['correlation_with_pandemic'])
        pandemic_corr_df = pandemic_corr_df.sort_values('correlation_with_pandemic', 
                                                      key=abs, ascending=False)

        # Save pandemic correlation analysis
        pandemic_corr_df.to_excel(f'{dir_base}/analysis/pandemic_correlations.xlsx')
        pandemic_corr_df.to_csv(f'{dir_base}/analysis/pandemic_correlations.tsv', sep='\t')

        # 5. Summary Statistics by Pandemic Status
        # Use the cleaned numeric data and merge with pandemic status
        pandemic_summary_data = df_numeric_clean.copy()
        pandemic_summary_data['pandemic'] = df.loc[df_numeric_clean.index, 'pandemic']
        pandemic_summary = pandemic_summary_data.groupby('pandemic')[columns_numeric_selected].mean()
        print("\nAverage values by pandemic status:")
        print(pandemic_summary)

        pandemic_summary.to_excel(f'{dir_base}/analysis/pandemic_summary_stats.xlsx')
        pandemic_summary.to_csv(f'{dir_base}/analysis/pandemic_summary_stats.tsv', sep='\t')
        
        # 6. Create comprehensive summary
        with open(f'{dir_base}/analysis/summary_PCA_tSNE.txt', 'a') as f:
            f.write("\n\nPCA ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Features used: {columns_numeric_selected}\n")
            f.write(f"Number of features: {len(columns_numeric_selected)}\n")
            f.write(f"Original samples: {len(df_numeric)}\n")
            f.write(f"Clean samples (after removing missing values): {len(df_numeric_clean)}\n")
            f.write(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%\n")
            f.write(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%\n")
            f.write(f"Total explained variance (PC1+PC2): {sum(pca.explained_variance_ratio_[:2])*100:.2f}%\n\n")
            
            f.write("PCA Loadings (PC1):\n")
            for feature, loading in zip(columns_numeric_selected, pca.components_[0]):
                f.write(f"  {feature}: {loading:.3f}\n")
            
            f.write("\nPCA Loadings (PC2):\n")
            for feature, loading in zip(columns_numeric_selected, pca.components_[1]):
                f.write(f"  {feature}: {loading:.3f}\n")
            
            f.write("\n\nCORRELATION ANALYSIS FOR PANDEMIC POTENTIAL\n")
            f.write("=" * 50 + "\n")
            f.write("Variables correlated with pandemic status:\n")
            f.write(pandemic_corr_df.to_string())
            
            f.write("\n\nSUMMARY STATISTICS BY PANDEMIC STATUS\n")
            f.write("=" * 50 + "\n")
            f.write(pandemic_summary.to_string())

    except Exception as e:
        print(f"Error during additional analysis: {e}")

    print("\nAnalysis complete! All files have been saved in organized directories.")
    print("Generated files:")
    print("- tables/: Correlation matrices in .xlsx and .tsv formats")
    print("- figures/: PCA plot, t-SNE plots and correlation matrix in .svg and .png formats")
    print("- analysis/: PCA scores, loadings, correlations, and summary.txt")

if __name__ == "__main__":
    print("Starting PCA, t-SNE and Correlation analysis...")
    run_pca_tsne_analysis()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Results saved!")
    print(f"{'='*80}")
