import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model_dir = '/bert_models' 

base_output_dir = ''

try:
    model_names = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
    if not model_names:
        print(f"Warning: No model directories found in {base_model_dir}")
    else:
        print(f"Found {len(model_names)} models: {model_names}")
except FileNotFoundError:
    print(f"Error: Base model directory not found: {base_model_dir}")
    model_names = [] 

# Load the dataset once
try:
    df = pd.read_json('')
except FileNotFoundError:
    print("Error: Dataset JSON file not found. Please check the path.")
    df = pd.DataFrame()
except Exception as e:
    print(f"Error loading dataset JSON: {e}")
    df = pd.DataFrame()


# --- Helper functions for token attribution ---
def get_tokens_in_span(attributions, full_text, span_start, span_end):
    reconstructed_text = ""
    token_indices = []
    # Iterate through tokens skipping [CLS] and [SEP]
    for i, (token, _) in enumerate(attributions[1:-1], 1):
        clean_token = token.replace(' ', '').strip()
        if token.startswith("##"):
             clean_token = token[2:] # remove ## prefix for matching
        else:
             clean_token = token # keep original token

        if not clean_token:
            continue

        # Find the token in the remaining text
        search_start = len(reconstructed_text) - (len(clean_token) // 2 if len(reconstructed_text)>0 else 0)
        if search_start < 0: search_start = 0

        token_position = -1
        try:
            # More robust search: iterate search start slightly if not found immediately
            found = False
            for offset in range(min(len(full_text) - search_start, 5)): # Check a small window
                 current_search_start = search_start + offset
                 if full_text[current_search_start:].startswith(clean_token):
                     token_position = current_search_start
                     found = True
                     break
            if not found:
                # Fallback: use simpler find if the above fails
                token_position = full_text.find(clean_token, search_start)

        except Exception as find_error:
            token_position = -1 # Ensure it's -1 if an error occurs

        if token_position != -1:
            token_end = token_position + len(clean_token)
            # Check for overlap: token must be at least partially within the span
            if not (token_position >= span_end or token_end <= span_start):
                token_indices.append(i)
            # Update reconstructed_text based on where the token was found + its length
            reconstructed_text = full_text[:max(len(reconstructed_text), token_end)]

    return token_indices


def get_span_attribution_score(attributions, token_indices):
    if not token_indices:
        return 0.0 # Return float zero

    # Ensure indices are within the valid range of attributions
    valid_indices = [idx for idx in token_indices if 0 <= idx < len(attributions)]
    if not valid_indices:
        return 0.0

    attribution_values = [attributions[idx][1] for idx in valid_indices]
    return sum(attribution_values) / len(attribution_values) if attribution_values else 0.0


# --- Create output directories ---
# Create base output directories for all analyses
threshold_dir = os.path.join(base_output_dir, 'threshold_analysis')
trait_ranking_dir = os.path.join(base_output_dir, 'trait_ranking')
trait_discrimination_dir = os.path.join(base_output_dir, 'trait_discrimination')
context_analysis_dir = os.path.join(base_output_dir, 'context_analysis')
trait_correspondence_dir = os.path.join(base_output_dir, 'trait_correspondence')

# Create all directories
for directory in [threshold_dir, trait_ranking_dir, trait_discrimination_dir, context_analysis_dir, trait_correspondence_dir]:
    os.makedirs(directory, exist_ok=True)

# --- Model processing loop ---
# Storage for all models' results
all_models_results = {}
all_trait_stats = {}

# Process each model only if model_names and df are not empty
if model_names and not df.empty:
    for name_model in model_names:
        print(f"\nProcessing model: {name_model}")
        model_path = os.path.join(base_model_dir, name_model)

        try:
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Move model to the determined device
            model = model.to(device)
            print(f"  Model '{name_model}' loaded to {device}")

            # Initialize explainer with the model on the correct device
            cls_explainer = SequenceClassificationExplainer(model, tokenizer)

            # Prepare results storage
            results = defaultdict(list)
            trait_attributions = defaultdict(list)

            # Process each sample
            total_samples = len(df)
            print(f"  Processing {total_samples} samples...")
            for idx, row in df.iterrows():
                text = row['text']
                linguistic_traits = row['linguistic_traits']

                if not isinstance(text, str) or not text.strip():
                    continue

                try:
                    # Get attributions using the explainer
                    attributions = cls_explainer(text)

                    # Filter out None or invalid attributions
                    if attributions is None or not all(isinstance(attr_tuple, tuple) and len(attr_tuple) == 2 for attr_tuple in attributions):
                        continue

                    valid_attributions = [(token, score) for token, score in attributions[1:-1] if isinstance(score, (int, float))]
                    all_attribution_scores = [score for _, score in valid_attributions]

                    avg_text_attribution = sum(all_attribution_scores) / len(all_attribution_scores) if all_attribution_scores else 0.0
                    # Avoid division by zero or near-zero
                    if abs(avg_text_attribution) < 1e-9: avg_text_attribution = 1e-9

                    # Process linguistic traits
                    if not isinstance(linguistic_traits, dict):
                        linguistic_traits = {}

                    for trait_name, spans in linguistic_traits.items():
                        if not spans:
                            continue

                        # Handle cases where spans might be a single dict instead of list
                        if isinstance(spans, dict):
                            spans = [spans]

                        if isinstance(spans, list):
                            for span in spans:
                                if isinstance(span, dict):
                                    span_text = span.get('text', '')
                                    span_start = span.get('start')
                                    span_end = span.get('end')

                                    # Basic validation for span data
                                    if not span_text or span_start is None or span_end is None or not isinstance(span_start, int) or not isinstance(span_end, int) or span_start >= span_end or span_start < 0 or span_end > len(text):
                                        continue

                                    token_indices = get_tokens_in_span(attributions, text, span_start, span_end)
                                    span_attribution = get_span_attribution_score(attributions, token_indices)

                                    # Calculate word count in span for analysis
                                    word_count = len(span_text.split())
                                    
                                    results['sample_id'].append(idx)
                                    results['trait'].append(trait_name)
                                    results['span_text'].append(span_text)
                                    results['span_attribution'].append(span_attribution)
                                    results['avg_text_attribution'].append(avg_text_attribution)
                                    results['attribution_ratio'].append(span_attribution / avg_text_attribution)
                                    results['span_length'].append(len(span_text))
                                    results['word_count'].append(word_count)
                                    results['span_position'].append(span_start / len(text))  # Normalized position

                                    # Store only valid attributions for stats
                                    if isinstance(span_attribution, (int, float)):
                                        trait_attributions[trait_name].append(span_attribution)

                except Exception as explainer_error:
                    print(f"  Error during explanation for sample {idx}, text: '{text[:50]}...'. Error: {explainer_error}")

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{total_samples} samples")

            # Clean up GPU memory after processing all samples for this model
            model.cpu()
            del model
            del cls_explainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Model '{name_model}' moved to CPU and memory cleared.")

            # Create and save results DataFrame if results were generated
            if results:
                results_df = pd.DataFrame(results)
                results_df['model'] = name_model  # Add model name to results

                # Calculate statistics for each trait
                trait_stats = {}
                for trait, values in trait_attributions.items():
                    if values:
                        trait_stats[trait] = {
                            'count': len(values),
                            'mean_attribution': float(np.mean(values)),
                            'median_attribution': float(np.median(values)),
                            'std_attribution': float(np.std(values))
                        }

                # Save model-specific results
                output_csv = os.path.join(base_output_dir, f'span_attribution_analysis_{name_model}.csv')
                results_df.to_csv(output_csv, index=False)
                print(f"  Results saved to '{output_csv}'")

                # Store results for comparison
                all_models_results[name_model] = results_df
                all_trait_stats[name_model] = trait_stats

        except Exception as e:
            print(f"  Error processing model {name_model}: {str(e)}")
            # Attempt to clean up GPU memory even if an error occurred mid-process
            if 'model' in locals() and hasattr(model, 'cpu'):
                model.cpu()
                del model
            if 'cls_explainer' in locals():
                del cls_explainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
else:
    print("Skipping model processing because no models were found or the dataset failed to load.")

# --- Combine results only if there are results ---
if all_models_results:
    combined_results = pd.concat(all_models_results.values(), ignore_index=True)
    combined_csv_path = os.path.join(base_output_dir, 'combined_attribution_results.csv')
    combined_results.to_csv(combined_csv_path, index=False)
    print(f"\nCombined results saved to '{combined_csv_path}'")

    # Get unique list of all traits encountered across all models
    all_traits = sorted(combined_results['trait'].unique())
    model_names_processed = list(all_models_results.keys())

    print("\nAnalyzing linguistic traits across different attribution ratio thresholds...")
    
    # --- NEW: Threshold-based Analysis Section ---
    # Define multiple thresholds to analyze
    ratio_thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    threshold_results = {}
    
    for threshold in ratio_thresholds:
        print(f"\nAnalyzing threshold: {threshold}")
        threshold_data = {}
        
        # For each model, calculate metrics at this threshold
        for model_name in model_names_processed:
            model_data = combined_results[combined_results['model'] == model_name]
            
            # Calculate high attribution spans per trait
            trait_counts = {}
            for trait in all_traits:
                trait_data = model_data[model_data['trait'] == trait]
                high_attr_spans = trait_data[trait_data['attribution_ratio'] >= threshold]
                
                total_spans = len(trait_data)
                high_spans = len(high_attr_spans)
                proportion = high_spans / total_spans if total_spans > 0 else 0
                
                trait_counts[trait] = {
                    'total_spans': total_spans,
                    'high_attribution_spans': high_spans,
                    'proportion': proportion,
                    'mean_ratio': trait_data['attribution_ratio'].mean(),
                    'median_ratio': trait_data['attribution_ratio'].median()
                }
            
            threshold_data[model_name] = trait_counts
        
        threshold_results[threshold] = threshold_data
    
    # 1. Plot trend of high attribution span proportion across thresholds
    for model_name in model_names_processed:
        plt.figure(figsize=(14, 10))
        
        for trait in all_traits:
            proportions = [threshold_results[t][model_name][trait]['proportion'] for t in ratio_thresholds]
            plt.plot(ratio_thresholds, proportions, marker='o', label=trait, linewidth=2)
        
        plt.title(f'Proportion of High Attribution Spans vs Threshold - {model_name}', fontsize=16)
        plt.xlabel('Attribution Ratio Threshold', fontsize=14)
        plt.ylabel('Proportion of Spans Above Threshold', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_dir, f'threshold_trend_{model_name}.png'), dpi=300)
        plt.close()
    
    # 2. Create a heatmap of proportions for each trait at different thresholds
    for model_name in model_names_processed:
        # Prepare data matrix for heatmap: traits x thresholds
        heatmap_data = np.zeros((len(all_traits), len(ratio_thresholds)))
        
        for i, trait in enumerate(all_traits):
            for j, threshold in enumerate(ratio_thresholds):
                heatmap_data[i, j] = threshold_results[threshold][model_name][trait]['proportion']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=[f'{t}' for t in ratio_thresholds],
                    yticklabels=all_traits)
        plt.title(f'Proportion of High Attribution Spans by Threshold - {model_name}', fontsize=16)
        plt.xlabel('Attribution Ratio Threshold', fontsize=14)
        plt.ylabel('Linguistic Trait', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_dir, f'threshold_heatmap_{model_name}.png'), dpi=300)
        plt.close()
    
    # 3. Bar chart comparing traits at each threshold
    for threshold in ratio_thresholds:
        plt.figure(figsize=(16, 10))
        
        trait_positions = np.arange(len(all_traits))
        bar_width = 0.8 / len(model_names_processed)
        
        for i, model_name in enumerate(model_names_processed):
            proportions = [threshold_results[threshold][model_name][trait]['proportion'] for trait in all_traits]
            plt.bar(trait_positions + i * bar_width - 0.4 + bar_width/2, proportions, bar_width, label=model_name)
        
        plt.title(f'Proportion of High Attribution Spans (Threshold = {threshold})', fontsize=16)
        plt.xlabel('Linguistic Trait', fontsize=14)
        plt.ylabel('Proportion of Spans Above Threshold', fontsize=14)
        plt.xticks(trait_positions, all_traits, rotation=45, ha='right', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_dir, f'bar_chart_threshold_{threshold}.png'), dpi=300)
        plt.close()
    
    # 4. Trait Ranking Analysis - how traits rank in importance across thresholds
    for model_name in model_names_processed:
        # Create dataframe to store trait ranks at each threshold
        trait_rankings = pd.DataFrame(index=all_traits, columns=ratio_thresholds)
        
        for threshold in ratio_thresholds:
            # Get proportions for this threshold and rank them
            proportions = {trait: threshold_results[threshold][model_name][trait]['proportion'] for trait in all_traits}
            # Sort traits by proportion (descending) and assign ranks
            sorted_traits = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
            for rank, (trait, _) in enumerate(sorted_traits, 1):
                trait_rankings.at[trait, threshold] = rank
        
        # Plot the rank changes across thresholds
        plt.figure(figsize=(12, 10))
        for trait in all_traits:
            plt.plot(ratio_thresholds, trait_rankings.loc[trait], marker='o', linewidth=2, label=trait)
        
        plt.title(f'Trait Importance Ranking Across Thresholds - {model_name}', fontsize=16)
        plt.xlabel('Attribution Ratio Threshold', fontsize=14)
        plt.ylabel('Rank (1 = Highest Proportion)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Make rank 1 at the top
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(trait_ranking_dir, f'rank_changes_{model_name}.png'), dpi=300)
        plt.close()
    
    # --- NEW: Additional Insights Section ---
    print("\nGenerating additional insights for linguistic trait analysis...")
    
    # 1. Trait Discrimination Analysis: How well does each trait discriminate between important and non-important spans?
    # For each model, analyze trait discrimination ability
    for model_name in model_names_processed:
        model_data = combined_results[combined_results['model'] == model_name]
        
        # Calculate discrimination metrics
        discrimination_stats = []
        
        for trait in all_traits:
            trait_data = model_data[model_data['trait'] == trait]['attribution_ratio']
            other_data = model_data[model_data['trait'] != trait]['attribution_ratio']
            
            if len(trait_data) < 5 or len(other_data) < 5:
                continue  # Skip traits with too few samples
            
            # Calculate Mann-Whitney U test (non-parametric test for distribution differences)
            try:
                u_stat, p_value = mannwhitneyu(trait_data, other_data)
                effect_size = u_stat / (len(trait_data) * len(other_data))  # Normalized U statistic
            except:
                p_value = 1.0
                effect_size = 0.0
            
            # Add to results
            discrimination_stats.append({
                'trait': trait,
                'mean_ratio': float(trait_data.mean()),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'sample_size': len(trait_data)
            })
        
        # Convert to DataFrame and sort by statistical significance
        discrim_df = pd.DataFrame(discrimination_stats).sort_values('p_value')
        discrim_df.to_csv(os.path.join(trait_discrimination_dir, f'discrimination_stats_{model_name}.csv'), index=False)
        
        # Plot discrimination ability
        plt.figure(figsize=(14, 8))
        x_pos = np.arange(len(discrim_df))
        plt.bar(x_pos, -np.log10(discrim_df['p_value']), color='steelblue')
        plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        plt.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01')
        plt.title(f'Trait Discrimination Significance - {model_name}', fontsize=16)
        plt.xlabel('Linguistic Trait', fontsize=14)
        plt.ylabel('-log10(p-value)', fontsize=14)
        plt.xticks(x_pos, discrim_df['trait'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(trait_discrimination_dir, f'discrimination_plot_{model_name}.png'), dpi=300)
        plt.close()
    
    # 2. Context Dependency Analysis: Do attribution patterns change based on span position or length?
    for model_name in model_names_processed:
        model_data = combined_results[combined_results['model'] == model_name].copy()
        
        # Position vs. attribution analysis (bin by position)
        model_data['position_bin'] = pd.cut(model_data['span_position'], bins=10, labels=False)
        
        position_analysis = model_data.groupby(['trait', 'position_bin'])['attribution_ratio'].mean().reset_index()
        
        # For each trait, plot position vs attribution
        for trait in all_traits:
            trait_pos_data = position_analysis[position_analysis['trait'] == trait]
            if len(trait_pos_data) < 3:  # Skip if too few data points
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(trait_pos_data['position_bin'], trait_pos_data['attribution_ratio'], marker='o', linewidth=2)
            plt.title(f'Position Effect on Attribution - {trait} ({model_name})', fontsize=14)
            plt.xlabel('Position in Text (Binned 0-9)', fontsize=12)
            plt.ylabel('Mean Attribution Ratio', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # Clean trait name for file path to avoid special characters
            trait_filename = trait.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            plt.savefig(os.path.join(context_analysis_dir, f'position_effect_{model_name}_{trait_filename}.png'), dpi=300)
            plt.close()
        
        # Length vs. attribution analysis
        model_data['length_bin'] = pd.qcut(model_data['span_length'], q=5, labels=False, duplicates='drop')
        length_analysis = model_data.groupby(['trait', 'length_bin'])['attribution_ratio'].mean().reset_index()
        
        plt.figure(figsize=(14, 10))
        for trait in all_traits:
            trait_len_data = length_analysis[length_analysis['trait'] == trait]
            if len(trait_len_data) < 3:
                continue
            plt.plot(trait_len_data['length_bin'], trait_len_data['attribution_ratio'], marker='o', linewidth=2, label=trait)
        
        plt.title(f'Span Length Effect on Attribution Ratio - {model_name}', fontsize=16)
        plt.xlabel('Span Length (Quintiles)', fontsize=14)
        plt.ylabel('Mean Attribution Ratio', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(context_analysis_dir, f'length_effect_{model_name}.png'), dpi=300)
        plt.close()
    
    # 3. Trait Correspondence Analysis
    # Create summary report for thresholds analysis
    summary_report = os.path.join(base_output_dir, 'threshold_analysis_summary.txt')
    with open(summary_report, 'w') as f:
        f.write("Linguistic Trait Attribution Analysis Summary\n")
        f.write("=========================================\n\n")
        
        f.write("1. Most Important Traits by Model\n")
        f.write("------------------------------\n")
        for model_name in model_names_processed:
            f.write(f"\nModel: {model_name}\n")
            
            # Report top traits at each threshold
            for threshold in ratio_thresholds:
                trait_proportions = []
                for trait in all_traits:
                    prop = threshold_results[threshold][model_name][trait]['proportion']
                    count = threshold_results[threshold][model_name][trait]['high_attribution_spans']
                    total = threshold_results[threshold][model_name][trait]['total_spans']
                    trait_proportions.append((trait, prop, count, total))
                
                # Sort by proportion (descending)
                trait_proportions.sort(key=lambda x: x[1], reverse=True)
                
                f.write(f"\n  At threshold {threshold}:\n")
                for i, (trait, prop, count, total) in enumerate(trait_proportions[:5], 1):
                    f.write(f"    {i}. {trait}: {prop:.2%} ({count}/{total} spans)\n")
        
        f.write("\n\n2. Trait Stability Across Thresholds\n")
        f.write("--------------------------------\n")
        
        for model_name in model_names_processed:
            f.write(f"\nModel: {model_name}\n")
            
            # Calculate rank stability (variance in rank across thresholds)
            rank_stability = {}
            for trait in all_traits:
                ranks = []
                for threshold in ratio_thresholds:
                    proportions = {t: threshold_results[threshold][model_name][t]['proportion'] for t in all_traits}
                    sorted_traits = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
                    rank = [i for i, (t, _) in enumerate(sorted_traits, 1) if t == trait][0]
                    ranks.append(rank)
                
                # Calculate variance in rank (lower is more stable)
                rank_stability[trait] = np.var(ranks)
            
            # Sort by stability (ascending variance)
            stable_traits = sorted(rank_stability.items(), key=lambda x: x[1])
            
            f.write("\n  Most stable traits (consistent importance across thresholds):\n")
            for i, (trait, variance) in enumerate(stable_traits[:5], 1):
                f.write(f"    {i}. {trait}: rank variance = {variance:.2f}\n")

    print(f"\nThreshold analysis summary saved to {summary_report}")
    print("\nAll analyses and visualizations completed!")

else:
    print("\nNo combined results were generated. Skipping analyses.")