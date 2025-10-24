# shap_table_analysis.py
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Your trait colors (adding a color for 'non-trait')
trait_colors = {
    'Figurative_Speech': '#1f77b4',  # Blue
    'Epithet': '#ff7f0e',           # Orange
    'Neologism': '#2ca02c',         # Green
    'Irony/Sarcasm': '#d62728',     # Red
    'Hyperbolic_Language': '#9467bd',  # Purple
    'Loaded_language': '#8c564b',   # Brown
    'non-trait': '#7f7f7f'          # Gray for non-trait
}

# Function to parse shap_values string into numpy array
def parse_shap_values(s, row_idx, df_name):
    """
    Parse SHAP values string into numpy array
    """
    try:
        # Clean string: remove brackets, replace newlines with spaces, strip
        s = s.strip('[]').replace('\n', ' ').strip()
        # Split by spaces and clean each value
        values = [v.strip(',').strip() for v in s.split() if v.strip(',')]
        # Convert to float, handle invalid values
        parsed_values = []
        for v in values:
            try:
                parsed_values.append(float(v))
            except ValueError:
                print(f"Warning: Could not convert value '{v}' to float in row {row_idx} of {df_name}")
                continue
        return np.array(parsed_values)
    except Exception as e:
        print(f"Error parsing shap_values in row {row_idx} of {df_name}: {e}")
        return np.array([])  # Return empty array to skip row

# Function to get predicted label from label_pred string
def get_predicted_label(label_pred_str, row_idx, df_name):
    """
    Extract predicted label from label_pred string
    """
    try:
        preds = ast.literal_eval(label_pred_str)
        max_entry = max(preds, key=lambda x: x['score'])
        max_label = max_entry['label']
        return 0 if max_label == 'LABEL_0' else 1
    except Exception as e:
        print(f"Error parsing label_pred in row {row_idx} of {df_name}: {e}")
        return -1  # Error flag

def extract_unique_traits(dfs):
    """
    Extract unique traits from all dataframes
    """
    all_traits = set()
    for df, df_name in dfs:
        for i, mapping_str in enumerate(df['token_trait_mapping']):
            try:
                mapping = ast.literal_eval(mapping_str)
                for traits in mapping:
                    for t in traits:
                        all_traits.add(t)
            except Exception as e:
                print(f"Error parsing token_trait_mapping in row {i} of {df_name}: {e}")
                continue

    trait_list = sorted(all_traits)
    print("Unique traits found:", trait_list)
    return trait_list

def compute_shap_statistics(dfs, trait_list):
    """
    Compute SHAP statistics for multiple model configurations
    
    Parameters:
    dfs: list of tuples (dataframe, model_name)
    trait_list: list of unique traits
    
    Returns:
    List of dictionaries with SHAP statistics for each model
    """
    all_results = []
    
    for df, model_name in dfs:
        print(f"\nProcessing {model_name}...")
        
        # Initialize storage for SHAP values by trait and label
        shap_by_trait_label = {trait: {'label_0': [], 'label_1': []} for trait in trait_list}
        sample_counts = {'label_0': 0, 'label_1': 0}
        total_shap_values = []
        
        for i in range(len(df)):
            try:
                # Get predicted label
                pred = get_predicted_label(df['label_pred'][i], i, model_name)
                if pred == -1:
                    continue
                
                # Parse SHAP values and trait mapping
                shap_vals = parse_shap_values(df['shap_values'][i], i, model_name)
                mapping = ast.literal_eval(df['token_trait_mapping'][i])
                
                if len(shap_vals) == 0 or len(shap_vals) != len(mapping):
                    continue
                
                # Aggregate SHAP values by trait for this sample
                trait_shap_contributions = {trait: 0 for trait in trait_list}
                trait_counts = {trait: 0 for trait in trait_list}
                
                for j, traits in enumerate(mapping):
                    for trait in traits:
                        if trait in trait_list:
                            trait_shap_contributions[trait] += shap_vals[j]
                            trait_counts[trait] += 1
                            total_shap_values.append(shap_vals[j])
                
                # Store in appropriate label bucket
                label_key = 'label_0' if pred == 0 else 'label_1'
                sample_counts[label_key] += 1
                
                for trait in trait_list:
                    if trait_counts[trait] > 0:
                        avg_contribution = trait_shap_contributions[trait] / trait_counts[trait]
                        shap_by_trait_label[trait][label_key].append(avg_contribution)
                    
            except Exception as e:
                print(f"Error processing row {i} in {model_name}: {e}")
                continue
        
        # Compute comprehensive SHAP statistics
        model_results = {
            'Model': model_name,
            'Sample_Counts': sample_counts,
            'Total_Samples': sample_counts['label_0'] + sample_counts['label_1']
        }
        
        # Add overall SHAP statistics
        if total_shap_values:
            total_shap_array = np.array(total_shap_values)
            model_results['Overall_SHAP_Mean'] = np.mean(total_shap_array)
            model_results['Overall_SHAP_Std'] = np.std(total_shap_array)
            model_results['Overall_SHAP_Abs_Mean'] = np.mean(np.abs(total_shap_array))
        else:
            model_results['Overall_SHAP_Mean'] = 0
            model_results['Overall_SHAP_Std'] = 0
            model_results['Overall_SHAP_Abs_Mean'] = 0
        
        # Average SHAP values by trait and label
        for trait in trait_list:
            # For label_0 (Neutral)
            if shap_by_trait_label[trait]['label_0']:
                model_results[f'{trait}_Neutral_avg'] = np.mean(shap_by_trait_label[trait]['label_0'])
                model_results[f'{trait}_Neutral_std'] = np.std(shap_by_trait_label[trait]['label_0'])
            else:
                model_results[f'{trait}_Neutral_avg'] = 0
                model_results[f'{trait}_Neutral_std'] = 0
            
            # For label_1 (HP)
            if shap_by_trait_label[trait]['label_1']:
                model_results[f'{trait}_HP_avg'] = np.mean(shap_by_trait_label[trait]['label_1'])
                model_results[f'{trait}_HP_std'] = np.std(shap_by_trait_label[trait]['label_1'])
            else:
                model_results[f'{trait}_HP_avg'] = 0
                model_results[f'{trait}_HP_std'] = 0
            
            # Overall absolute importance
            all_shap_vals = (shap_by_trait_label[trait]['label_0'] + 
                           shap_by_trait_label[trait]['label_1'])
            if all_shap_vals:
                model_results[f'{trait}_Abs_avg'] = np.mean(np.abs(all_shap_vals))
                model_results[f'{trait}_Abs_std'] = np.std(np.abs(all_shap_vals))
            else:
                model_results[f'{trait}_Abs_avg'] = 0
                model_results[f'{trait}_Abs_std'] = 0
        
        all_results.append(model_results)
    
    return all_results

def create_comparison_tables(results, trait_list):
    """
    Create various comparison tables from the results
    """
    # Table 1: Detailed SHAP values by model and label
    detailed_data = []
    for result in results:
        row = {'Configuration': result['Model']}
        for trait in trait_list:
            row[f'{trait}_Neutral'] = result[f'{trait}_Neutral_avg']
            row[f'{trait}_HP'] = result[f'{trait}_HP_avg']
        detailed_data.append(row)
    
    detailed_table = pd.DataFrame(detailed_data)
    
    # Table 2: Feature importance (absolute SHAP values)
    importance_data = []
    for result in results:
        row = {'Configuration': result['Model']}
        for trait in trait_list:
            row[trait] = result[f'{trait}_Abs_avg']
        importance_data.append(row)
    
    importance_table = pd.DataFrame(importance_data)
    
    # Table 3: Sample counts and overall statistics
    sample_data = []
    for result in results:
        row = {
            'Configuration': result['Model'],
            'Neutral_Samples': result['Sample_Counts']['label_0'],
            'HP_Samples': result['Sample_Counts']['label_1'],
            'Total_Samples': result['Total_Samples'],
            'Overall_SHAP_Mean': result['Overall_SHAP_Mean'],
            'Overall_SHAP_Abs_Mean': result['Overall_SHAP_Abs_Mean']
        }
        sample_data.append(row)
    
    sample_table = pd.DataFrame(sample_data)
    
    return detailed_table, importance_table, sample_table

def print_formatted_tables(detailed_table, importance_table, sample_table, trait_list):
    """
    Print nicely formatted tables to console
    """
    print("=" * 120)
    print("SAMPLE COUNTS AND OVERALL STATISTICS")
    print("=" * 120)
    display_sample_table = sample_table.copy()
    for col in ['Overall_SHAP_Mean', 'Overall_SHAP_Abs_Mean']:
        display_sample_table[col] = display_sample_table[col].round(4)
    print(display_sample_table.to_string(index=False))
    print()
    
    print("=" * 120)
    print("DETAILED AVERAGE SHAP VALUES BY CONFIGURATION AND LABEL")
    print("=" * 120)
    
    # Create a more readable detailed table
    readable_detailed = detailed_table.copy()
    # Round values for display
    for col in readable_detailed.columns:
        if col != 'Configuration':
            readable_detailed[col] = readable_detailed[col].round(4)
    
    print(readable_detailed.to_string(index=False))
    print()
    
    print("=" * 120)
    print("FEATURE IMPORTANCE (AVERAGE ABSOLUTE SHAP VALUES)")
    print("=" * 120)
    
    # Create a more readable importance table
    readable_importance = importance_table.copy()
    # Round values
    for col in readable_importance.columns:
        if col != 'Configuration':
            readable_importance[col] = readable_importance[col].round(4)
    
    # Calculate overall importance ranking
    trait_importance = {}
    for trait in trait_list:
        trait_importance[trait] = np.mean(readable_importance[trait])
    
    # Sort traits by overall importance
    sorted_traits = sorted(trait_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Overall Feature Importance Ranking:")
    for i, (trait, importance) in enumerate(sorted_traits, 1):
        print(f"  {i:2d}. {trait}: {importance:.4f}")
    print()
    
    print(readable_importance.to_string(index=False))
    
    return sorted_traits

def create_latex_tables(detailed_table, importance_table, sample_table, trait_list):
    """
    Generate LaTeX code for paper tables
    """
    print("\n" + "=" * 100)
    print("LaTeX TABLES FOR PAPER")
    print("=" * 100)
    
    # LaTeX Table 1: Feature Importance
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Average Absolute SHAP Values by Configuration}")
    print("\\label{tab:shap_importance}")
    print("\\begin{tabular}{l" + "c" * len(trait_list) + "}")
    print("\\toprule")
    header = "Configuration & " + " & ".join([trait.replace('_', '\\_') for trait in trait_list]) + " \\\\"
    print(header)
    print("\\midrule")
    
    for _, row in importance_table.iterrows():
        config = row['Configuration']
        values = [f"{row[trait]:.4f}" for trait in trait_list]
        print(f"{config} & " + " & ".join(values) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # LaTeX Table 2: Sample Counts
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Sample Counts by Configuration}")
    print("\\label{tab:sample_counts}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Configuration & Neutral & HP & Total & Avg Abs SHAP \\\\")
    print("\\midrule")
    
    for _, row in sample_table.iterrows():
        print(f"{row['Configuration']} & {row['Neutral_Samples']} & {row['HP_Samples']} & {row['Total_Samples']} & {row['Overall_SHAP_Abs_Mean']:.4f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

def save_tables_to_csv(detailed_table, importance_table, sample_table):
    """
    Save all tables to CSV files
    """
    detailed_table.to_csv('shap_values_detailed.csv', index=False)
    importance_table.to_csv('shap_values_importance.csv', index=False)
    sample_table.to_csv('sample_counts.csv', index=False)
    
    print("\nTables saved to:")
    print("- shap_values_detailed.csv")
    print("- shap_values_importance.csv") 
    print("- sample_counts.csv")

def perform_configuration_comparison(results, trait_list):
    """
    Perform detailed comparison between configurations
    """
    print("\n" + "=" * 100)
    print("CONFIGURATION COMPARISON ANALYSIS")
    print("=" * 100)
    
    if len(results) < 2:
        print("Need at least 2 configurations for comparison")
        return
    
    # Compare each pair of configurations
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            config1 = results[i]['Model']
            config2 = results[j]['Model']
            
            print(f"\nComparison: {config1} vs {config2}")
            print("-" * 50)
            
            differences = []
            for trait in trait_list:
                diff = results[j][f'{trait}_Abs_avg'] - results[i][f'{trait}_Abs_avg']
                rel_diff = (diff / results[i][f'{trait}_Abs_avg']) * 100 if results[i][f'{trait}_Abs_avg'] != 0 else float('inf')
                differences.append((trait, diff, rel_diff))
            
            # Sort by absolute difference
            differences.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Largest absolute differences:")
            for trait, diff, rel_diff in differences[:5]:
                print(f"  {trait:20} {diff:+.4f} ({rel_diff:+.1f}%)")
            
            # Find which configuration gives higher importance to each trait
            print("\nTraits with higher importance in each configuration:")
            higher_in_config1 = []
            higher_in_config2 = []
            
            for trait, diff, _ in differences:
                if diff > 0:
                    higher_in_config2.append(trait)
                elif diff < 0:
                    higher_in_config1.append(trait)
            
            print(f"  Higher in {config1}: {', '.join(higher_in_config1[:5])}")
            print(f"  Higher in {config2}: {', '.join(higher_in_config2[:5])}")

def create_importance_plot(importance_table, trait_list, sorted_traits):
    """
    Create a bar plot of feature importance across configurations
    """
    # Get top traits for plotting
    top_traits = [trait for trait, _ in sorted_traits[:8]]  # Top 8 traits
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    x = np.arange(len(top_traits))
    width = 0.25
    configurations = importance_table['Configuration'].tolist()
    
    for i, config in enumerate(configurations):
        values = [importance_table.loc[importance_table['Configuration'] == config, trait].values[0] for trait in top_traits]
        ax.bar(x + i * width, values, width, label=config, alpha=0.8)
    
    ax.set_xlabel('Rhetorical Traits', fontsize=12)
    ax.set_ylabel('Average Absolute SHAP Value', fontsize=12)
    ax.set_title('Feature Importance Across Configurations', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([trait.replace('_', '\n') for trait in top_traits], fontsize=10)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved to 'feature_importance_comparison.png'")

def main():
    """
    Main function to run the SHAP table analysis
    """
    print("SHAP Table Analysis")
    print("=" * 50)
    
    pretrained_path = '/home/michele.maggini/16_ottobre_GP3_SHAP_PRE_results/pretrained.csv'
    ft_path = '/home/michele.maggini/XAI_HIPP/SHAP/results/FT.csv'  # Update with full path if not in current directory
    ft_emb_path = '/home/michele.maggini/XAI_HIPP/SHAP/results/FT_EMB.csv'

    # Load the CSVs
    dfs = [
        (pd.read_csv(pretrained_path), 'Linear-probing'),
        (pd.read_csv(ft_path), 'FT'),
        (pd.read_csv(ft_emb_path), 'FT+EMB')
    ]
    if not dfs:
        print("Error: No dataframes provided. Please load your data into the 'dfs' variable.")
        return
    
    # Step 1: Extract unique traits
    print("Step 1: Extracting unique traits...")
    trait_list = extract_unique_traits(dfs)
    
    # Step 2: Compute SHAP statistics
    print("\nStep 2: Computing SHAP statistics...")
    results = compute_shap_statistics(dfs, trait_list)
    
    # Step 3: Create comparison tables
    print("\nStep 3: Creating comparison tables...")
    detailed_table, importance_table, sample_table = create_comparison_tables(results, trait_list)
    
    # Step 4: Print formatted tables
    print("\nStep 4: Generating formatted output...")
    sorted_traits = print_formatted_tables(detailed_table, importance_table, sample_table, trait_list)
    
    # Step 5: Create LaTeX tables
    print("\nStep 5: Generating LaTeX code...")
    create_latex_tables(detailed_table, importance_table, sample_table, trait_list)
    
    # Step 6: Save to CSV
    print("\nStep 6: Saving tables to CSV...")
    save_tables_to_csv(detailed_table, importance_table, sample_table)
    
    # Step 7: Perform configuration comparison
    print("\nStep 7: Performing configuration comparison...")
    perform_configuration_comparison(results, trait_list)
    
    # Step 8: Create visualization
    print("\nStep 8: Creating visualization...")
    create_importance_plot(importance_table, trait_list, sorted_traits)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()