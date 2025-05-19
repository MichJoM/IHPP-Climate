import numpy as np
import pandas as pd
import argparse
import os
import logging
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


class HyperpartisanConfig(PretrainedConfig):
    model_type = "ft_all_traits"
    
    def __init__(
        self,
        base_model_name= "dbmdz/bert-base-italian-uncased",
        n_linguistic_traits=24, #For each of these 8 trait types, you're generating 3 features (presence, count, average length).
                                #Therefore: 8 trait types × 3 features per trait type = 24 total linguistic trait features.
        n_refined_techniques=18,
        **kwargs
    ):
        if "model_type" not in kwargs:
            kwargs["model_type"] = "ft_all_traits"
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.n_linguistic_traits = n_linguistic_traits
        self.n_refined_techniques = n_refined_techniques
        
class HyperpartisanDataset(torch.utils.data.Dataset):
    def __init__(self, texts, linguistic_traits, refined_techniques, labels, tokenizer, max_len):
        self.texts = texts
        # Convert to numpy arrays upfront
        self.linguistic_traits = np.array(linguistic_traits, dtype=np.float32)
        self.refined_techniques = np.array(refined_techniques, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'linguistic_traits': torch.from_numpy(self.linguistic_traits[idx]),
            'refined_techniques': torch.from_numpy(self.refined_techniques[idx]),
            'labels': torch.tensor(self.labels[idx])
        }
    
class HyperpartisanModel(PreTrainedModel):
    config_class = HyperpartisanConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model_prefix = "ft_all_traits"
        
        # Base model
        self.model = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.model.config.hidden_size
        
        # Initialize feature layers
        self.linguistic_trait_layer = self._init_feature_layer(config.n_linguistic_traits)
        self.refined_technique_layer = self._init_feature_layer(config.n_refined_techniques)
        
        # Calculate classifier input size
        classifier_input_size = hidden_size
        if config.n_linguistic_traits > 0:
            classifier_input_size += 128
        if config.n_refined_techniques > 0:
            classifier_input_size += 128
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def _init_feature_layer(self, input_dim):
        if input_dim == 0:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask, linguistic_traits, refined_techniques, token_type_ids=None):
        # Process text
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=validate_token_type_ids(token_type_ids)
        )
        
        text_features = outputs.last_hidden_state[:, 0]  # CLS token
        combined_features = [text_features]
        
        # Process linguistic traits if present
        if self.linguistic_trait_layer is not None:
            combined_features.append(self.linguistic_trait_layer(linguistic_traits))
            
        # Process refined techniques if present
        if self.refined_technique_layer is not None:
            combined_features.append(self.refined_technique_layer(refined_techniques))
        
        return self.classifier(torch.cat(combined_features, dim=1))

def encode_linguistic_traits(traits_dict, exclude=None, max_text_length=512):
    """Encode linguistic traits, optionally excluding one trait."""
    # If traits_dict is a string, parse it into a dictionary
    if isinstance(traits_dict, str):
        try:
            traits_dict = json.loads(traits_dict)
        except json.JSONDecodeError:
            # If parsing fails, return a zero vector
            return np.zeros(24)  # Return zeros for all 24 features (8 traits * 3 features)
    
    # Define the traits we're looking for in the original order
    all_trait_types = [
        "Loaded_language", "Figurative_Speech", "Epithet", "Neologism", 
        "Irony/Sarcasm", "Agents", "Terms", "Hyperbolic_Language"
    ]
    
    # Determine which traits to keep
    if exclude is not None and exclude in all_trait_types:
        trait_types = [t for t in all_trait_types if t != exclude]
    else:
        trait_types = all_trait_types
    
    # Initialize feature vector (presence, count, avg_length for each trait)
    features = np.zeros(len(trait_types) * 3)
    
    # For each trait type (in the filtered list)
    for trait_idx, trait_type in enumerate(trait_types):
        if trait_type in traits_dict and len(traits_dict[trait_type]) > 0:
            # Mark presence (1 if present)
            features[trait_idx] = 1
            
            # Count spans
            features[len(trait_types) + trait_idx] = len(traits_dict[trait_type])
            
            # Calculate average span length
            total_length = 0
            for span in traits_dict[trait_type]:
                if isinstance(span, dict):  # Ensure span is a dictionary
                    total_length += (span["end"] - span["start"])
            features[2 * len(trait_types) + trait_idx] = total_length / len(traits_dict[trait_type])
    
    # If we excluded a trait, we need to maintain the same feature vector size
    # by padding with zeros in the appropriate positions
    if exclude is not None and exclude in all_trait_types:
        # Find the index of the excluded trait in the original list
        excluded_idx = all_trait_types.index(exclude)
        
        # Create a new feature vector with zeros for the excluded trait
        full_features = np.zeros(24)  # 8 traits * 3 features
        
        # Copy the non-excluded features to their original positions
        current_pos = 0
        for i, trait in enumerate(all_trait_types):
            if trait != exclude:
                # Copy all three features for this trait
                full_features[i] = features[current_pos]
                full_features[8 + i] = features[len(trait_types) + current_pos]
                full_features[16 + i] = features[2 * len(trait_types) + current_pos]
                current_pos += 1
        
        return full_features
    
    return features

def encode_refined_techniques(techniques_str, exclude=None):
    """Encode refined techniques, optionally excluding one technique."""
    # Define all possible techniques
    all_techniques = [
        "Slogan/Conversation_Killer",
        "Appeal_to_Time",
        "Appeal_to_Values/Flag_Waving",
        "Appeal_to_Authority",
        "Appeal_to_Popularity",
        "Appeal_to_Fear",
        "Straw_Man/Red_Herring",
        "Tu_Quoque/Whataboutism",
        "Loaded_Language",
        "Repetition",
        "Intentional_Confusion_Vagueness",
        "Exaggeration_Minimisation",
        "Name_Calling",
        "Reductio_ad_Hitlerum",
        "Smear/Doubt",
        "Causal_Oversimplification/Consequential_Oversimplification",
        "False_Dilemma_No_Choice",
        "no_technique_detected"
    ]
    
    # If exclude is specified, remove it from the list
    if exclude is not None and exclude in all_techniques:
        all_techniques.remove(exclude)
    
    # Clean input and split
    techniques = techniques_str.replace("\\", "/").split(",")
    
    # Create one-hot encoding
    encoding = np.zeros(len(all_techniques))
    for technique in techniques:
        technique = technique.strip()
        if technique in all_techniques:
            encoding[all_techniques.index(technique)] = 1
    
    return encoding

def validate_token_type_ids(token_type_ids):
    """
    Ensure token_type_ids only contain valid values (0 or 1).
    """
    if token_type_ids is None:
        return None
    
    # Check if there are any values > 1
    if torch.any(token_type_ids > 1):
        # Clone to avoid modifying the original tensor
        fixed_ids = token_type_ids.clone()
        # Set any value > 1 to 1
        fixed_ids[fixed_ids > 1] = 1
        return fixed_ids
    
    return token_type_ids


######ERROR ANALYSIS######
def detailed_error_analysis(model, dataloader, device, texts, labels, label_names=None):
    """
    Perform detailed error analysis on model predictions.
    
    Args:
        model: The trained model
        dataloader: DataLoader containing evaluation data
        device: Device to run inference on
        texts: List of original text inputs corresponding to dataloader items
        labels: List of true labels corresponding to dataloader items
        label_names: Optional dictionary mapping from label index to human-readable name
    
    Returns:
        Dictionary containing error analysis results
    """
    model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []
    all_indices = []
    idx = 0
    
    # Default label names if not provided
    if label_names is None:
        label_names = {0: "Class 0", 1: "Class 1"}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_traits = batch['linguistic_traits'].to(device)
            refined_techniques = batch['refined_techniques'].to(device)
            batch_labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_traits=linguistic_traits,
                refined_techniques=refined_techniques
            )
            
            # Get predictions and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            # Store indices for matching with original texts
            batch_size = input_ids.size(0)
            all_indices.extend(range(idx, idx + batch_size))
            idx += batch_size
    
    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate error analysis metrics
    error_analysis = {
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(all_labels, all_predictions, target_names=[label_names[i] for i in sorted(label_names.keys())], output_dict=True),
        'errors': []
    }
    
    # Find misclassified examples
    for i, (pred, label, probs, idx) in enumerate(zip(all_predictions, all_labels, all_probs, all_indices)):
        if pred != label:
            confidence = probs[pred]
            error_analysis['errors'].append({
                'index': idx,
                'text': texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx],  # Truncate long texts
                'true_label': label_names[label],
                'predicted_label': label_names[pred],
                'confidence': float(confidence),
                'probabilities': {label_names[i]: float(p) for i, p in enumerate(probs)}
            })
    
    # Sort errors by confidence (most confident errors first)
    error_analysis['errors'] = sorted(error_analysis['errors'], key=lambda x: x['confidence'], reverse=True)
    
    # Analyze patterns in errors
    if error_analysis['errors']:
        # Group errors by true/predicted label combinations
        error_patterns = {}
        for error in error_analysis['errors']:
            key = f"{error['true_label']} -> {error['predicted_label']}"
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(error)
        
        error_analysis['error_patterns'] = {
            pattern: {
                'count': len(errors),
                'percentage': len(errors) / len(error_analysis['errors']) * 100,
                'avg_confidence': sum(e['confidence'] for e in errors) / len(errors),
                'examples': errors[:3]  # Include only first 3 examples
            } for pattern, errors in error_patterns.items()
        }
        
        # Calculate length statistics for misclassified examples
        text_lengths = [len(error['text']) for error in error_analysis['errors']]
        error_analysis['length_stats'] = {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths)
        }
        
        # Feature importance for errors (if available)
        if 'linguistic_traits' in batch and 'refined_techniques' in batch:
            error_analysis['feature_analysis'] = "Feature importance analysis would require additional processing"
    
    return error_analysis

def visualize_error_analysis(error_analysis, output_dir=None):
    """
    Visualize the results of error analysis
    
    Args:
        error_analysis: The error analysis results from detailed_error_analysis
        output_dir: Optional directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Create a directory for visualizations if provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    conf_matrix = error_analysis['confusion_matrix']
    
    # Convert to percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    ax = sns.heatmap(
        conf_matrix_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues', 
        cbar=True,
        xticklabels=list(error_analysis['classification_report'].keys())[:-3],
        yticklabels=list(error_analysis['classification_report'].keys())[:-3]
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (% of True Label)')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.show()
    
    # 2. Error patterns visualization
    if 'error_patterns' in error_analysis:
        patterns = list(error_analysis['error_patterns'].keys())
        counts = [info['count'] for info in error_analysis['error_patterns'].values()]
        confidences = [info['avg_confidence'] for info in error_analysis['error_patterns'].values()]
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        patterns = [patterns[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        
        # Plot error patterns
        plt.figure(figsize=(12, 6))
        ax = plt.barh(patterns, counts, color='skyblue')
        plt.xlabel('Number of Errors')
        plt.ylabel('Error Pattern (True -> Predicted)')
        plt.title('Common Error Patterns')
        
        # Add count annotations
        for i, v in enumerate(counts):
            plt.text(v + 0.1, i, str(v), color='black', va='center')
            
        if output_dir:
            plt.savefig(os.path.join(output_dir, "error_patterns.png"), bbox_inches='tight')
        plt.show()
        
        # Plot confidence by error pattern
        plt.figure(figsize=(12, 6))
        ax = plt.barh(patterns, confidences, color='coral')
        plt.xlabel('Average Confidence')
        plt.ylabel('Error Pattern (True -> Predicted)')
        plt.title('Model Confidence in Error Patterns')
        
        # Add confidence annotations
        for i, v in enumerate(confidences):
            plt.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')
            
        if output_dir:
            plt.savefig(os.path.join(output_dir, "error_confidence.png"), bbox_inches='tight')
        plt.show()
    
    # 3. Generate summary report
    print("=== ERROR ANALYSIS SUMMARY ===")
    print(f"Total errors: {len(error_analysis['errors'])}")
    
    if 'error_patterns' in error_analysis:
        print("\nError Patterns:")
        for pattern, info in sorted(error_analysis['error_patterns'].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {pattern}: {info['count']} errors ({info['percentage']:.1f}%), avg confidence: {info['avg_confidence']:.3f}")
    
    if 'length_stats' in error_analysis:
        print("\nText Length Statistics for Errors:")
        for stat, value in error_analysis['length_stats'].items():
            print(f"  {stat}: {value}")
    
    # Save the top errors to a file
    if output_dir:
        error_df = pd.DataFrame(error_analysis['errors'])
        if not error_df.empty:
            error_df.to_csv(os.path.join(output_dir, "misclassified_examples.csv"), index=False)
            print(f"\nDetailed error examples saved to {os.path.join(output_dir, 'misclassified_examples.csv')}")

def feature_importance_analysis(model, dataloader, device, feature_names=None):
    """
    Analyze feature importance using permutation importance
    
    Args:
        model: The trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        feature_names: Optional list of feature names
    
    Returns:
        Dictionary of feature importance results
    """
    from sklearn.metrics import f1_score
    import numpy as np
    
    model.eval()
    
    # Get baseline performance
    all_features = {
        'linguistic_traits': [],
        'refined_techniques': []
    }
    all_labels = []
    all_preds = []
    
    # First pass to collect features and get baseline performance
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_traits = batch['linguistic_traits'].to(device)
            refined_techniques = batch['refined_techniques'].to(device)
            labels = batch['labels'].to(device)
            
            # Store features
            all_features['linguistic_traits'].append(linguistic_traits.cpu().numpy())
            all_features['refined_techniques'].append(refined_techniques.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Get baseline predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_traits=linguistic_traits,
                refined_techniques=refined_techniques
            )
            _, preds = torch.max(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
    
    # Concatenate arrays
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)
    
    # Calculate baseline f1
    baseline_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Define feature groups for permutation
    if not feature_names:
        # Default feature names based on dimensions
        n_linguistic = all_features['linguistic_traits'].shape[1]
        n_refined = all_features['refined_techniques'].shape[1]
        
        linguistic_names = [f"Linguistic_{i}" for i in range(n_linguistic)]
        refined_names = [f"Technique_{i}" for i in range(n_refined)]
        
        feature_names = {
            'linguistic_traits': linguistic_names,
            'refined_techniques': refined_names
        }
    
    # Permutation importance calculation
    importance_results = {}
    
    for feature_type, features in all_features.items():
        feature_importances = []
        
        for i in range(features.shape[1]):
            # Create a copy of the features
            permuted_features = {
                'linguistic_traits': all_features['linguistic_traits'].copy(),
                'refined_techniques': all_features['refined_techniques'].copy()
            }
            
            # Permute this feature
            np.random.shuffle(permuted_features[feature_type][:, i])
            
            # Evaluate with permuted feature
            permuted_preds = []
            
            for j in range(0, len(all_labels), dataloader.batch_size):
                end_idx = min(j + dataloader.batch_size, len(all_labels))
                batch_size = end_idx - j
                
                # Get batch slices
                batch_input_ids = dataloader.dataset[j:end_idx]['input_ids'].to(device)
                batch_attention_mask = dataloader.dataset[j:end_idx]['attention_mask'].to(device)
                
                # Create tensors from permuted features
                batch_linguistic = torch.tensor(
                    permuted_features['linguistic_traits'][j:end_idx], 
                    dtype=torch.float
                ).to(device)
                
                batch_refined = torch.tensor(
                    permuted_features['refined_techniques'][j:end_idx], 
                    dtype=torch.float
                ).to(device)
                
                # Get predictions with permuted feature
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        linguistic_traits=batch_linguistic,
                        refined_techniques=batch_refined
                    )
                    _, preds = torch.max(outputs, dim=1)
                    permuted_preds.append(preds.cpu().numpy())
            
            # Combine predictions
            permuted_preds = np.concatenate(permuted_preds)
            
            # Calculate permuted f1
            permuted_f1 = f1_score(all_labels, permuted_preds, average='weighted')
            
            # Feature importance = decrease in performance
            importance = baseline_f1 - permuted_f1
            feature_importances.append(importance)
        
        # Store results for this feature type
        feature_type_names = feature_names.get(feature_type, [f"Feature_{i}" for i in range(len(feature_importances))])
        importance_results[feature_type] = {
            'names': feature_type_names,
            'importance': feature_importances
        }
    
    return {
        'baseline_f1': baseline_f1,
        'feature_importance': importance_results
    }

def visualize_feature_importance(importance_results, output_dir=None):
    """
    Visualize feature importance analysis results
    
    Args:
        importance_results: Results from feature_importance_analysis
        output_dir: Optional directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Create a directory for visualizations if provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine all features for visualization
    all_features = []
    all_importances = []
    all_categories = []
    
    for feature_type, results in importance_results['feature_importance'].items():
        all_features.extend(results['names'])
        all_importances.extend(results['importance'])
        all_categories.extend([feature_type] * len(results['names']))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': all_features,
        'Importance': all_importances,
        'Category': all_categories
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Plot overall feature importance
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Importance', y='Feature', hue='Category', data=df.head(20), palette='viridis')
    plt.title('Top 20 Feature Importance (F1 Score Decrease When Permuted)')
    plt.xlabel('Importance (higher means more important)')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
    plt.show()
    
    # Plot importance by category
    for feature_type, results in importance_results['feature_importance'].items():
        # Get top features for this category
        df_category = df[df['Category'] == feature_type].sort_values('Importance', ascending=False).head(10)
        
        if len(df_category) > 0:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=df_category, palette='Blues_d')
            plt.title(f'Top 10 {feature_type} Feature Importance')
            plt.xlabel('Importance (higher means more important)')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{feature_type}_importance.png"), bbox_inches='tight')
            plt.show()
    
    # Save feature importance to CSV
    if output_dir:
        df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        print(f"Feature importance results saved to {os.path.join(output_dir, 'feature_importance.csv')}")

# Function to analyze the most common linguistic traits and techniques in errors
def analyze_error_features(error_indices, df, label_names=None):
    """
    Analyze which linguistic traits and propaganda techniques are most common in errors
    
    Args:
        error_indices: Indices of misclassified examples
        df: Original dataframe with feature information
        label_names: Optional dictionary mapping from label index to human-readable name
    
    Returns:
        Dictionary with analysis results
    """
    import pandas as pd
    import json
    
    # Default label names if not provided
    if label_names is None:
        label_names = {0: "Class 0", 1: "Class 1"}
    
    # Filter dataframe to error examples
    error_df = df.iloc[error_indices].copy()
    
    # Analyze linguistic traits
    trait_counts = {
        'Loaded_language': 0, 'Metafore': 0, 'Epiteti': 0, 'Neologismi': 0, 
        'Ironia': 0, 'Agents': 0, 'Entities': 0, 'Iperboli': 0
    }
    
    # Analyze propaganda techniques
    technique_counts = {}
    
    # Process each error
    for _, row in error_df.iterrows():
        # Parse linguistic traits
        traits = row['linguistic_traits']
        if isinstance(traits, str):
            try:
                traits = json.loads(traits)
            except:
                continue
                
        # Count traits
        for trait_type in trait_counts.keys():
            if trait_type in traits and len(traits[trait_type]) > 0:
                trait_counts[trait_type] += 1
        
        # Count techniques
        techniques = row['refined_technique'].split(',')
        for technique in techniques:
            technique = technique.strip()
            if technique:
                if technique in technique_counts:
                    technique_counts[technique] += 1
                else:
                    technique_counts[technique] = 1
    
    # Calculate percentages
    n_errors = len(error_df)
    trait_percentages = {k: (v/n_errors*100) for k, v in trait_counts.items()}
    technique_percentages = {k: (v/n_errors*100) for k, v in technique_counts.items()}
    
    # Compare with overall dataset
    overall_trait_counts = {trait: 0 for trait in trait_counts}
    overall_technique_counts = {}
    
    for _, row in df.iterrows():
        # Parse linguistic traits
        traits = row['linguistic_traits']
        if isinstance(traits, str):
            try:
                traits = json.loads(traits)
            except:
                continue
                
        # Count traits
        for trait_type in overall_trait_counts.keys():
            if trait_type in traits and len(traits[trait_type]) > 0:
                overall_trait_counts[trait_type] += 1
        
        # Count techniques
        techniques = row['refined_technique'].split(',')
        for technique in techniques:
            technique = technique.strip()
            if technique:
                if technique in overall_technique_counts:
                    overall_technique_counts[technique] += 1
                else:
                    overall_technique_counts[technique] = 1
    
    # Calculate overall percentages
    n_total = len(df)
    overall_trait_percentages = {k: (v/n_total*100) for k, v in overall_trait_counts.items()}
    overall_technique_percentages = {k: (v/n_total*100) for k, v in overall_technique_counts.items()}
    
    # Calculate relative enrichment (how much more common in errors vs overall)
    trait_enrichment = {
        k: trait_percentages[k] / overall_trait_percentages[k] if overall_trait_percentages[k] > 0 else float('inf')
        for k in trait_counts
    }
    
    technique_enrichment = {
        k: technique_percentages.get(k, 0) / overall_technique_percentages.get(k, 1) 
        for k in set(list(technique_counts.keys()) + list(overall_technique_counts.keys()))
        if k in technique_counts and k in overall_technique_counts and overall_technique_counts[k] > 0
    }
    
    # Prepare results
    results = {
        'linguistic_traits': {
            'counts': trait_counts,
            'percentages': trait_percentages,
            'overall_percentages': overall_trait_percentages,
            'enrichment': trait_enrichment
        },
        'propaganda_techniques': {
            'counts': technique_counts,
            'percentages': technique_percentages,
            'overall_percentages': overall_technique_percentages,
            'enrichment': technique_enrichment
        }
    }
    
    return results

def visualize_error_features(feature_analysis, output_dir=None):
    """
    Visualize feature analysis for errors
    
    Args:
        feature_analysis: Results from analyze_error_features
        output_dir: Optional directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create a directory for visualizations if provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize linguistic traits
    trait_data = feature_analysis['linguistic_traits']
    
    # Create DataFrame for traits
    trait_df = pd.DataFrame({
        'Trait': list(trait_data['counts'].keys()),
        'Error %': list(trait_data['percentages'].values()),
        'Overall %': list(trait_data['overall_percentages'].values()),
        'Enrichment': list(trait_data['enrichment'].values())
    })
    
    # Sort by enrichment
    trait_df = trait_df.sort_values('Enrichment', ascending=False)
    
    # Plot trait percentages
    plt.figure(figsize=(12, 6))
    trait_df.plot(x='Trait', y=['Error %', 'Overall %'], kind='bar', figsize=(12, 6))
    plt.title('Linguistic Traits: Presence in Errors vs Overall Dataset')
    plt.xlabel('Linguistic Trait')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "trait_percentages.png"), bbox_inches='tight')
    plt.show()
    
    # Plot trait enrichment
    plt.figure(figsize=(12, 6))
    plt.bar(trait_df['Trait'], trait_df['Enrichment'])
    plt.title('Linguistic Traits: Enrichment in Errors vs Overall Dataset')
    plt.xlabel('Linguistic Trait')
    plt.ylabel('Enrichment (Error % / Overall %)')
    plt.xticks(rotation=45)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "trait_enrichment.png"), bbox_inches='tight')
    plt.show()
    
    # Visualize propaganda techniques
    technique_data = feature_analysis['propaganda_techniques']
    
    # Create DataFrame for techniques (top 10 by count)
    technique_df = pd.DataFrame({
        'Technique': list(technique_data['counts'].keys()),
        'Count': list(technique_data['counts'].values()),
        'Error %': list(technique_data['percentages'].values())
    })
    
    # Sort by count and get top 10
    technique_df = technique_df.sort_values('Count', ascending=False).head(10)
    
    # Plot technique counts
    plt.figure(figsize=(14, 6))
    plt.bar(technique_df['Technique'], technique_df['Count'])
    plt.title('Top 10 Propaganda Techniques in Misclassified Examples')
    plt.xlabel('Propaganda Technique')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "technique_counts.png"), bbox_inches='tight')
    plt.show()
    
    # Create DataFrame for technique enrichment
    if technique_data['enrichment']:
        enrichment_df = pd.DataFrame({
            'Technique': list(technique_data['enrichment'].keys()),
            'Enrichment': list(technique_data['enrichment'].values())
        })
        
        # Sort by enrichment and get top 10
        enrichment_df = enrichment_df.sort_values('Enrichment', ascending=False).head(10)
        
        # Plot technique enrichment
        plt.figure(figsize=(14, 6))
        plt.bar(enrichment_df['Technique'], enrichment_df['Enrichment'])
        plt.title('Top 10 Propaganda Techniques: Enrichment in Errors vs Overall Dataset')
        plt.xlabel('Propaganda Technique')
        plt.ylabel('Enrichment (Error % / Overall %)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "technique_enrichment.png"), bbox_inches='tight')
        plt.show()
    
    # Save results to CSV
    if output_dir:
        trait_df.to_csv(os.path.join(output_dir, "trait_analysis.csv"), index=False)
        technique_df.to_csv(os.path.join(output_dir, "technique_analysis.csv"), index=False)
        
        if technique_data['enrichment']:
            enrichment_df.to_csv(os.path.join(output_dir, "technique_enrichment.csv"), index=False)
            
# Add this to the eval_model function to collect more detailed information
def eval_model_with_analysis(model, dataloader, device, df=None, text_column='text', output_dir=None, label_names=None):
    """
    Evaluate model with detailed error analysis
    
    Args:
        model: The trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        df: Original dataframe with feature information
        text_column: Column name containing the text in df
        output_dir: Directory to save analysis results
        label_names: Dictionary mapping from label index to human-readable name
    
    Returns:
        Dictionary with evaluation metrics and analysis results
    """
    # Default label names if not provided
    if label_names is None:
        label_names = {0: "Non-Hyperpartisan", 1: "Hyperpartisan"}
    
    # Basic evaluation
    accuracy, avg_loss, f1, recall = eval_model(model, dataloader, device)
    
    results = {
        'metrics': {
            'accuracy': accuracy,
            'loss': avg_loss,
            'f1': f1,
            'recall': recall
        }
    }
    
    # If dataframe is provided, perform detailed analysis
    if df is not None:
        # Extract texts for error analysis
        texts = df[text_column].tolist()
        labels = df['labels'].tolist()
        
        # Perform error analysis
        error_analysis = detailed_error_analysis(model, dataloader, device, texts, labels, label_names)
        results['error_analysis'] = error_analysis
        
        # Visualize error analysis
        if output_dir:
            analysis_dir = os.path.join(output_dir, "error_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            visualize_error_analysis(error_analysis, analysis_dir)
            
            # Get indices of misclassified examples
            error_indices = [error['index'] for error in error_analysis['errors']]
            
            # Analyze features in errors
            feature_analysis = analyze_error_features(error_indices, df, label_names)
            results['feature_analysis'] = feature_analysis
            
            # Visualize feature analysis
            visualize_error_features(feature_analysis, analysis_dir)
            
            # Perform feature importance analysis if possible
            try:
                importance_results = feature_importance_analysis(model, dataloader, device)
                results['feature_importance'] = importance_results
                
                # Visualize feature importance
                visualize_feature_importance(importance_results, analysis_dir)
            except Exception as e:
                print(f"Warning: Feature importance analysis failed: {str(e)}")
    
    return results

######ERROR ANALALYSIS END######



class HyperpartisanConfig(PretrainedConfig):
    model_type = "ft_all_traits"
    
    def __init__(
        self,
        base_model_name= "dbmdz/bert-base-italian-uncased",
        n_linguistic_traits=24, #For each of these 8 trait types, you're generating 3 features (presence, count, average length).
                                #Therefore: 8 trait types × 3 features per trait type = 24 total linguistic trait features.
        n_refined_techniques=18,
        **kwargs
    ):
        if "model_type" not in kwargs:
            kwargs["model_type"] = "ft_all_traits"

        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.n_linguistic_traits = n_linguistic_traits
        self.n_refined_techniques = n_refined_techniques
        self.base_tokenizer_name = base_model_name


        
class HyperpartisanModel(PreTrainedModel):
    config_class = HyperpartisanConfig
    #model_prefix=""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_prefix="ft_all_traits"
        
        # Use base AutoModel
        self.model = AutoModel.from_pretrained(config.base_model_name)
        
        # Get the hidden size from the model config
        hidden_size = self.model.config.hidden_size
        
        # Feature processing layers
        self.linguistic_trait_layer = nn.Sequential(
            nn.Linear(config.n_linguistic_traits, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.refined_technique_layer = nn.Sequential(
            nn.Linear(config.n_refined_techniques, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined features size
        combined_size = hidden_size + 128 + 128
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Binary classification
        )

    def forward(self, input_ids, attention_mask, linguistic_traits, refined_techniques, token_type_ids=None):
        # Process text with the model model, explicitly setting token_type_ids to None
        token_type_ids = validate_token_type_ids(token_type_ids)
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids = token_type_ids
            # Do not pass token_type_ids at all
        )
        
        # Get the pooled output or CLS token representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # Models like BERT
            text_features = outputs.pooler_output
        else:
            # Models that don't have a pooler (get CLS token or average)
            text_features = outputs.last_hidden_state[:, 0]
        
        # Process additional features
        linguistic_features = self.linguistic_trait_layer(linguistic_traits)
        technique_features = self.refined_technique_layer(refined_techniques)
        
        # Combine all features
        combined_features = torch.cat((text_features, linguistic_features, technique_features), dim=1)
        
        # Get logits from the classifier
        logits = self.classifier(combined_features)
        
        return logits


    def get_input_embeddings(self):
        # Delegate to the model model
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # Delegate to the model model
        self.model.set_input_embeddings(value)


class HyperpartisanDataset(torch.utils.data.Dataset):
    def __init__(self, texts, linguistic_traits, refined_techniques, labels, tokenizer, max_len):
        self.texts = texts
        self.linguistic_traits = linguistic_traits
        self.refined_techniques = refined_techniques
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        linguistic_trait = self.linguistic_traits[idx]
        refined_technique = self.refined_techniques[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Squeeze the extra dimension from the encoding
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        eval_model_with_analysis
        # Convert other features to tensors
        linguistic_trait_tensor = torch.tensor(linguistic_trait, dtype=torch.float)
        refined_technique_tensor = torch.tensor(refined_technique, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'linguistic_traits': linguistic_trait_tensor,
            'refined_techniques': refined_technique_tensor,
            'labels': label_tensor
        }

def get_predictions(model, dataloader, device):
    """Helper function to get predictions from a model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_traits = batch['linguistic_traits'].to(device)
            refined_techniques = batch['refined_techniques'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_traits=linguistic_traits,
                refined_techniques=refined_techniques
            )
            
            # Store predictions and labels
            _, preds = torch.max(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        linguistic_traits = batch['linguistic_traits'].to(device)
        refined_techniques = batch['refined_techniques'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            linguistic_traits=linguistic_traits,
            refined_techniques=refined_techniques
        )
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Store predictions and labels for metrics calculation
        _, preds = torch.max(outputs, dim=1)
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss, f1, recall


from scipy.stats import ttest_rel
import numpy as np

def run_ablation_study(model, df, tokenizer, params, output_dir, feature_type, labels_to_remove):
    results = {}
    
    # First, get full model predictions for significance testing
    print("Getting full model predictions for significance testing...")
    full_features = {
        'linguistic': df['linguistic_traits'].apply(lambda x: encode_linguistic_traits(x)).tolist(),
        'refined': df['refined_technique'].apply(lambda x: encode_refined_techniques(x)).tolist()
    }
    
    full_dataset = HyperpartisanDataset(
        texts=df['text'].tolist(),
        linguistic_traits=full_features['linguistic'],
        refined_techniques=full_features['refined'],
        labels=df['labels'].tolist(),
        tokenizer=tokenizer,
        max_len=params['max_len']
    )
    full_loader = DataLoader(full_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Get full model predictions
    full_preds, full_labels = get_predictions(model, full_loader, model.device)
    full_f1 = f1_score(full_labels, full_preds, average='weighted')
    
    # Store predictions for significance testing
    all_preds = {'full': full_preds}
    
    for label_to_remove in labels_to_remove:
        print(f"\nRemoving {label_to_remove}...")
        
        # Encode features with current label removed
        if feature_type == "linguistic":
            features = df['linguistic_traits'].apply(
                lambda x: encode_linguistic_traits(x, exclude=label_to_remove)
            ).tolist()
            n_features = len(features[0])
            other_features = full_features['refined']
        else:
            features = df['refined_technique'].apply(
                lambda x: encode_refined_techniques(x, exclude=label_to_remove)
            ).tolist()
            n_features = len(features[0])
            other_features = full_features['linguistic']
        
        # Create a temporary model with adjusted feature dimensions
        temp_config = HyperpartisanConfig(
            base_model_name=model.config.base_model_name,
            n_linguistic_traits=len(other_features[0]) if feature_type == "refined" else n_features,
            n_refined_techniques=n_features if feature_type == "refined" else len(other_features[0])
        )
        
        temp_model = HyperpartisanModel(temp_config).to(model.device)
        
        # Copy weights from original model (except for the modified feature layers)
        state_dict = model.state_dict()
        
        # Remove keys for feature layers we're replacing
        keys_to_remove = []
        if feature_type == "linguistic":
            keys_to_remove.extend([k for k in state_dict.keys() if 'linguistic_trait_layer' in k])
        else:
            keys_to_remove.extend([k for k in state_dict.keys() if 'refined_technique_layer' in k])
            
        for k in keys_to_remove:
            state_dict.pop(k)
            
        temp_model.load_state_dict(state_dict, strict=False)
        
        # Create dataset
        dataset = HyperpartisanDataset(
            texts=df['text'].tolist(),
            linguistic_traits=other_features if feature_type == "refined" else features,
            refined_techniques=features if feature_type == "refined" else other_features,
            labels=df['labels'].tolist(),
            tokenizer=tokenizer,
            max_len=params['max_len']
        )
        
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Evaluate and get predictions
        metrics = eval_model(temp_model, dataloader, model.device)
        preds, _ = get_predictions(temp_model, dataloader, model.device)
        
        # Store predictions for significance testing
        all_preds[label_to_remove] = preds
        
        # Calculate p-value against full model
        _, p_value = ttest_rel(
            [1 if p == t else 0 for p, t in zip(full_preds, full_labels)],  # Full model correct (1) or not (0)
            [1 if p == t else 0 for p, t in zip(preds, full_labels)]        # Ablated model correct (1) or not (0)
        )
        
        # Calculate effect size (Cohen's d)
        effect_size = (metrics[2] - full_f1) / np.sqrt(
            (np.std([1 if p == t else 0 for p, t in zip(full_preds, full_labels)])**2 +
            np.std([1 if p == t else 0 for p, t in zip(preds, full_labels)])**2) / 2
        )
        
        results[label_to_remove] = {
            "accuracy": float(metrics[0]),  # Ensure float
            "loss": float(metrics[1]),
            "f1": float(metrics[2]),
            "recall": float(metrics[3]),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "significant": bool(p_value < 0.05)  # Convert to native bool
        }
        
        # Clean up
        del temp_model
        torch.cuda.empty_cache()
        plt.close('all')
    
    return results


def create_ablation_plot(results, title, output_path):
    """Modified to include significance markers"""
    labels = list(results.keys())
    f1_scores = [x['f1'] for x in results.values()]
    significant = [x['significant'] for x in results.values()]
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=labels, y=f1_scores)
    
    # Add significance markers
    for i, (score, sig) in enumerate(zip(f1_scores, significant)):
        if sig:
            # Add star above significant bars
            ax.text(i, score + 0.01, '*', ha='center', va='bottom', color='red', fontsize=14)
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_traits = batch['linguistic_traits'].to(device)
            refined_techniques = batch['refined_techniques'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_traits=linguistic_traits,
                refined_techniques=refined_techniques
            )
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            
            # Store predictions and labels
            _, preds = torch.max(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss, f1, recall

def objective(trial, args, tokenizer, device, df):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    max_len = trial.suggest_categorical("max_len", [32, 64, 128])


    print(f"\n----- Trial {trial.number} -----", flush=True)
    print(f"Parameters: lr={learning_rate}, batch_size={batch_size}, max_len={max_len}", flush=True)

    # Load and preprocess the dataset
    texts = df['text'].tolist()
    linguistic_traits = df['linguistic_traits'].apply(lambda x: encode_linguistic_traits(x)).tolist()
    refined_techniques = df['refined_technique'].apply(lambda x: encode_refined_techniques(x)).tolist()
    labels = df['labels'].tolist()

    # Get dimensions of encoded features
    n_linguistic_traits = len(linguistic_traits[0])
    n_refined_techniques = len(refined_techniques[0])
    
    print(f"Feature dimensions: linguistic={n_linguistic_traits}, techniques={n_refined_techniques}", flush=True)

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(texts))
    train_texts, val_texts = texts[:train_size], texts[train_size:]
    train_linguistic_traits, val_linguistic_traits = linguistic_traits[:train_size], linguistic_traits[train_size:]
    train_refined_techniques, val_refined_techniques = refined_techniques[:train_size], refined_techniques[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    print(f"Dataset split: train={len(train_texts)}, val={len(val_texts)}", flush=True)

    # Create datasets
    try:
        print("Creating datasets...", flush=True)
        train_dataset = HyperpartisanDataset(train_texts, train_linguistic_traits, train_refined_techniques, train_labels, tokenizer, max_len)
        val_dataset = HyperpartisanDataset(val_texts, val_linguistic_traits, val_refined_techniques, val_labels, tokenizer, max_len)
        
        # Sample an item to verify
        sample_item = train_dataset[0]
        print(f"Sample item keys: {list(sample_item.keys())}", flush=True)
        print(f"input_ids shape: {sample_item['input_ids'].shape}", flush=True)
        print(f"attention_mask shape: {sample_item['attention_mask'].shape}", flush=True)
        print(f"linguistic_traits shape: {sample_item['linguistic_traits'].shape}", flush=True)
        print(f"refined_techniques shape: {sample_item['refined_techniques'].shape}", flush=True)
        
        # Create dataloaders
        print("Creating dataloaders...", flush=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False,
            num_workers=0  # No parallel loading to avoid potential issues
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        # Initialize the model with custom config
        print(f"Initializing model with: {args.model_name}", flush=True)

        config = HyperpartisanConfig(
            base_model_name=args.model_name,
            n_linguistic_traits=n_linguistic_traits,
            n_refined_techniques=n_refined_techniques
        )
        
        model = HyperpartisanModel(config)
        model = model.to(device)
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters", flush=True)

        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )

        # Training loop with error handling
        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = float('inf')
        best_epoch = -1
        best_val_recall = 0  
        
        # Create a directory for saving models
        model_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Dictionary to store metrics
        metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_recall': []
        }
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
            try:
                # Train
                train_acc, train_loss, train_f1, train_recall = train_model(model, train_loader, optimizer, scheduler, device)
                val_acc, val_loss, val_f1, val_recall = eval_model(model, val_loader, device)
                
                print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}", flush=True)
                
                # Validate
                val_acc, val_loss, val_f1, val_recall = eval_model(model, val_loader, device)
                print(f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}", flush=True)
                
                # Save metrics history
                metrics_history['train_loss'].append(train_loss)
                metrics_history['train_acc'].append(train_acc)
                metrics_history['train_f1'].append(train_f1)
                metrics_history['val_loss'].append(val_loss)
                metrics_history['val_acc'].append(val_acc)
                metrics_history['val_f1'].append(val_f1)
                metrics_history['train_recall'].append(train_recall)
                metrics_history['val_recall'].append(val_recall)
                # Track best performance for early stopping
                improved = False
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    improved = True
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    improved = True
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True

                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    improved = True
                
                if improved:
                    best_epoch = epoch
                    # Save the model using HuggingFace's save_pretrained
                    model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}")
                    os.makedirs(model_path, exist_ok=True)
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
                    print(f"Saved best model to {model_path}", flush=True)
                
                # Report to Optuna (using F1 score as the primary metric)
                trial.report(val_f1, epoch)
                
                # Check for pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            except Exception as e:
                print(f"Error during epoch {epoch + 1}: {str(e)}", flush=True)
                # Log the error but continue to the next epoch
                import traceback
                traceback.print_exc()
        
        # Save the metrics history
        metrics_path = os.path.join(model_dir, "metrics_history.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f)
        
        print(f"\nTrial {trial.number} summary:", flush=True)
        print(f"Best epoch: {best_epoch + 1}", flush=True)
        print(f"Best validation accuracy: {best_val_acc:.4f}", flush=True)
        print(f"Best validation F1: {best_val_f1:.4f}", flush=True)
        print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
        print(f"Best recall: {best_val_recall:.4f}", flush=True)
        # Return F1 score as the objective value
        return best_val_f1#, {"accuracy": best_val_acc, "f1": best_val_f1, "loss": best_val_loss, "recall": best_val_recall}
        
    except Exception as e:
        print(f"Error in trial setup: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        # Return a very low score to indicate failure
        return 0.0
def plot_performance_across_runs(all_metrics, output_dir):
    """Plot model performance across all runs"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    runs = [m['run'] for m in all_metrics]
    f1_scores = [m['f1'] for m in all_metrics]
    macro_f1_scores = [m['macro_f1'] for m in all_metrics]
    
    plt.figure(figsize=(12, 6))
    
    # Plot weighted F1 and macro F1
    plt.plot(runs, f1_scores, marker='o', label='Weighted F1')
    plt.plot(runs, macro_f1_scores, marker='o', label='Macro F1')
    
    # Add average lines
    avg_f1 = np.mean(f1_scores)
    avg_macro_f1 = np.mean(macro_f1_scores)
    plt.axhline(y=avg_f1, color='blue', linestyle='--', alpha=0.3)
    plt.axhline(y=avg_macro_f1, color='orange', linestyle='--', alpha=0.3)
    
    plt.title('Model Performance Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('F1 Score')
    plt.xticks(runs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'performance_across_runs.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Hyperpartisan classifier model with BERT and additional features')
    
    # Required parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='BERT model to use (default: bert-base-uncased)')
    parser.add_argument('--train_path', type=str, required=True,
                      help='Path to the training JSON dataset file')
    parser.add_argument('--test_path', type=str, required=True,
                      help='Path to the test JSON dataset file')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory to save the model and results')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--n_trials', type=int, default=10,
                      help='Number of Optuna trials (default: 10)')
    parser.add_argument('--n_runs', type=int, default=1,
                      help='Number of times to run the full experiment (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--study_name', type=str, default="MDEBERTA")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets once (they'll be used for all runs)
    print("\n===== Loading Datasets =====")
    logging.info("===== Loading Datasets =====")
    train_df = pd.read_json(args.train_path)
    test_df = pd.read_json(args.test_path)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    logging.info(f"Training set size: {len(train_df)}")
    logging.info(f"Test set size: {len(test_df)}")
    
    # Run the experiment n_runs times
    for run in range(args.n_runs):
        run_dir = os.path.join(args.output_dir, f"run_{run+1}")
        os.makedirs(run_dir, exist_ok=True)
        
        logging.info(f"Starting run {run+1}/{args.n_runs}")
        print(f"\n===== Starting Run {run+1}/{args.n_runs} =====")
        
        # Set seed for reproducibility (different seed for each run)
        current_seed = args.seed + run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # Register custom model
        model_type = "ft_all_traits"
        AutoConfig.register(model_type, HyperpartisanConfig)
        AutoModel.register(HyperpartisanConfig, HyperpartisanModel)
        AutoModelForSequenceClassification.register(HyperpartisanConfig, HyperpartisanModel)
            
        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        logging.info(f"Using device: {device}")

        # Initialize the tokenizer
        print(f"Loading tokenizer for model: {args.model_name}")
        logging.info(f"Loading tokenizer for model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=False,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Run Optuna study to find the best model (using train_df)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{args.study_name}_run_{run+1}",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )

        # Objective function wrapper (uses train_df)
        def objective_wrapper(trial):
            try:
                return objective(trial, args, tokenizer, device, train_df)
            except Exception as e:
                print(f"Trial failed: {str(e)}")
                logging.error(f"Trial failed: {str(e)}")
                return 0.0  # Return lowest possible score
        
        # Run optimization
        study.optimize(objective_wrapper, n_trials=args.n_trials)
        
        print("\n===== Study Results =====")
        logging.info("\n===== Study Results =====")
        print(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best trial: {study.best_trial.number}")
        print(f"Best F1 score: {study.best_trial.value:.4f}")
        logging.info(f"Best F1 score: {study.best_trial.value:.4f}")
        print(f"Best parameters: {study.best_trial.params}")
        logging.info(f"Best parameters: {study.best_trial.params}")

        # Find the best model checkpoint for the best trial
        # First check if the best trial directory is in the run directory (expected location)
        best_trial_dir = os.path.join(run_dir, f"trial_{study.best_trial.number}")
        
        # If not found in the run directory, check if it's in the parent directory
        if not os.path.exists(best_trial_dir):
            parent_best_trial_dir = os.path.join(args.output_dir, f"trial_{study.best_trial.number}")
            if os.path.exists(parent_best_trial_dir):
                best_trial_dir = parent_best_trial_dir
                print(f"Best trial directory found outside run directory: {best_trial_dir}")
                logging.info(f"Best trial directory found outside run directory: {best_trial_dir}")
            else:
                # If still not found, search for it in the entire output directory
                print(f"Searching for trial_{study.best_trial.number} in output directory...")
                logging.info(f"Searching for trial_{study.best_trial.number} in output directory...")
                for root, dirs, _ in os.walk(args.output_dir):
                    if f"trial_{study.best_trial.number}" in dirs:
                        best_trial_dir = os.path.join(root, f"trial_{study.best_trial.number}")
                        print(f"Found best trial directory at: {best_trial_dir}")
                        logging.info(f"Found best trial directory at: {best_trial_dir}")
                        break
        
        if not os.path.exists(best_trial_dir):
            print(f"Error: Cannot find best trial directory for trial_{study.best_trial.number}")
            logging.error(f"Cannot find best trial directory for trial_{study.best_trial.number}")
            print(f"Skipping run {run+1} evaluation")
            logging.error(f"Skipping run {run+1} evaluation")
            continue
            
        # After finding the trial directory, proceed with model loading and evaluation
        best_model_path = None
        best_epoch = -1

        # Try to find best model checkpoint in the trial directory
        try:
            # Iterate through the folders in the trial directory
            for folder in os.listdir(best_trial_dir):
                if folder.startswith("best_model_epoch_"):
                    epoch_str = folder[len("best_model_epoch_"):]
                    try:
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_model_path = os.path.join(best_trial_dir, folder)
                    except ValueError:
                        print(f"Skipping folder {folder} due to invalid epoch format.")
                        logging.warning(f"Skipping folder {folder} due to invalid epoch format.")
        except Exception as e:
            print(f"Error finding best model in {best_trial_dir}: {str(e)}")
            logging.error(f"Error finding best model in {best_trial_dir}: {str(e)}")
            continue

        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            logging.info(f"Loading best model from {best_model_path}")
            
            try:
                # Initialize the final model with the best parameters
                config = HyperpartisanConfig(
                    base_model_name=args.model_name, 
                    n_linguistic_traits=24,
                    n_refined_techniques=18
                )
                
                # Load the model from the best checkpoint
                best_model = HyperpartisanModel.from_pretrained(best_model_path, local_files_only=True)
                best_model.config.model_type = "ft_all_traits"
                best_model = best_model.to(device)

                print(f"Loading tokenizer from {best_model_path}")
                logging.info(f"Loading tokenizer from {best_model_path}")
                tokenizer = AutoTokenizer.from_pretrained(best_model_path, local_files_only=True)
                
                # Save the final model and tokenizer
                best_model.config.save_pretrained(run_dir)
                best_model.save_pretrained(run_dir)
                tokenizer.save_pretrained(run_dir)

                print(f"Best model and tokenizer saved to: {run_dir}")
                logging.info(f"Best model and tokenizer saved to: {run_dir}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                logging.error(f"Error loading model: {str(e)}")
                continue
        else:
            print("No valid model found to load.")
            logging.error("No valid model found to load.")
            continue

        ##############################################################
        ### Evaluate Best Model on Test Dataset ###
        ##############################################################

        print("\n===== Evaluating Best Model on Test Dataset =====")
        logging.info("\n===== Evaluating Best Model on Test Dataset =====")

        # Prepare the test dataset for evaluation
        test_dataset = HyperpartisanDataset(
            texts=test_df['text'].tolist(),
            linguistic_traits=test_df['linguistic_traits'].apply(lambda x: encode_linguistic_traits(x)).tolist(),
            refined_techniques=test_df['refined_technique'].apply(lambda x: encode_refined_techniques(x)).tolist(),
            labels=test_df['labels'].tolist(),
            tokenizer=tokenizer,
            max_len=study.best_trial.params['max_len']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=study.best_trial.params['batch_size'],
            shuffle=False
        )

        # Evaluate the best model on test set
        accuracy, avg_loss, f1, recall = eval_model(best_model, test_loader, device)

        # Calculate additional metrics (macro F1 and precision)
        from sklearn.metrics import precision_score, f1_score
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                linguistic_traits = batch['linguistic_traits'].to(device)
                refined_techniques = batch['refined_techniques'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = best_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    linguistic_traits=linguistic_traits,
                    refined_techniques=refined_techniques
                )
                _, preds = torch.max(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        precision = precision_score(all_labels, all_preds, average='weighted')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        # Save best model metrics
        best_model_metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "macro_f1": macro_f1,
            "recall": recall,
            "precision": precision,
            "best_params": study.best_trial.params,
            "seed": current_seed
        }

        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(best_model_metrics, f, indent=2)

        print("\nTest Set Metrics:")
        logging.info("\nTest Set Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        logging.info(f"F1 Score (weighted): {f1:.4f}")
        print(f"F1 Score (macro): {macro_f1:.4f}")
        logging.info(f"F1 Score (macro): {macro_f1:.4f}")
        print(f"Recall: {recall:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        print("Metrics saved to:", os.path.join(run_dir, "test_metrics.json"))
        logging.info(f"Metrics saved to: {os.path.join(run_dir, 'test_metrics.json')}")

        # Run ablation studies with statistical testing on test set
        print("\n===== Running Ablation Studies with Statistical Testing =====")
        logging.info("\n===== Running Ablation Studies with Statistical Testing =====")

        # Get best parameters
        best_params = {
            'max_len': study.best_trial.params['max_len'],
            'batch_size': study.best_trial.params['batch_size']
        }

        # Define features to remove
        linguistic_traits = ["Loaded_language", "Metafore", "Epiteti", "Neologismi", 
                            "Ironia", "Agents", "Entities", "Iperboli"]

        refined_techniques = ["Slogan/Conversation_Killer", "Appeal_to_Time", 
                            "Appeal_to_Values/Flag_Waving", "Appeal_to_Authority",
                            "Appeal_to_Popularity", "Appeal_to_Fear", 
                            "Straw_Man/Red_Herring", "Tu_Quoque/Whataboutism",
                            "Loaded_Language", "Repetition", 
                            "Intentional_Confusion_Vagueness", "Exaggeration_Minimisation",
                            "Name_Calling", "Reductio_ad_Hitlerum", "Smear/Doubt",
                            "Causal_Oversimplification/Consequential_Oversimplification",
                            "False_Dilemma_No_Choice", "no_technique_detected"]

        # Run ablation studies with statistical testing on test set
        print("\n===== Experiment 1: Refined Techniques Ablation =====")
        logging.info("\n===== Experiment 1: Refined Techniques Ablation =====")
        exp1_results = run_ablation_study(
            best_model, test_df, tokenizer, best_params, run_dir,
            "refined", refined_techniques)

        print("\n===== Experiment 2: Linguistic Traits Ablation =====")
        logging.info("\n===== Experiment 2: Linguistic Traits Ablation =====")
        exp2_results = run_ablation_study(
            best_model, test_df, tokenizer, best_params, run_dir,
            "linguistic", linguistic_traits)

        # Create and save plots with significance markers
        print("\n===== Generating Results Visualization =====")
        logging.info("\n===== Generating Results Visualization =====")
        create_ablation_plot(exp1_results, 
                            "F1 Scores When Removing Refined Techniques (∗ p < 0.05)",
                            os.path.join(run_dir, "refined_techniques_ablation_stats.png"))

        create_ablation_plot(exp2_results,
                            "F1 Scores When Removing Linguistic Traits (∗ p < 0.05)",
                            os.path.join(run_dir, "linguistic_traits_ablation_stats.png"))

        # Save all results with statistical metrics
        ablation_results = {
            "refined_techniques": exp1_results,
            "linguistic_traits": exp2_results
        }

        with open(os.path.join(run_dir, "ablation_results_with_stats.json"), "w") as f:
            json.dump(ablation_results, f, indent=2)

        print("\n===== Statistical Ablation Analysis Completed =====")
        logging.info("\n===== Statistical Ablation Analysis Completed =====")
        print("Results saved to:")
        logging.info("Results saved to:")
        print(f"- {os.path.join(run_dir, 'refined_techniques_ablation_stats.png')}")
        logging.info(f"- {os.path.join(run_dir, 'refined_techniques_ablation_stats.png')}")
        print(f"- {os.path.join(run_dir, 'linguistic_traits_ablation_stats.png')}")
        logging.info(f"- {os.path.join(run_dir, 'linguistic_traits_ablation_stats.png')}")
        print(f"- {os.path.join(run_dir, 'ablation_results_with_stats.json')}")
        logging.info(f"- {os.path.join(run_dir, 'ablation_results_with_stats.json')}")
                             
        print(f"\n===== Run {run+1}/{args.n_runs} Completed =====")
        logging.info(f"\n===== Run {run+1}/{args.n_runs} Completed =====")
        print(f"Results saved to: {run_dir}")
        logging.info(f"Results saved to: {run_dir}")

    # After all runs are complete, aggregate results
    if args.n_runs > 1:
        print("\n===== Aggregating Results Across All Runs =====")
        logging.info("\n===== Aggregating Results Across All Runs =====")
        
        # Collect metrics from all runs
        all_metrics = []
        for run in range(args.n_runs):
            run_dir = os.path.join(args.output_dir, f"run_{run+1}")
            metrics_file = os.path.join(run_dir, "test_metrics.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics['run'] = run+1
                    all_metrics.append(metrics)
        
        # Calculate averages and standard deviations
        if all_metrics:
            avg_metrics = {
                'accuracy': {
                    'mean': np.mean([m['accuracy'] for m in all_metrics]),
                    'std': np.std([m['accuracy'] for m in all_metrics])
                },
                'f1': {
                    'mean': np.mean([m['f1'] for m in all_metrics]),
                    'std': np.std([m['f1'] for m in all_metrics])
                },
                'macro_f1': {
                    'mean': np.mean([m['macro_f1'] for m in all_metrics]),
                    'std': np.std([m['macro_f1'] for m in all_metrics])
                },
                'recall': {
                    'mean': np.mean([m['recall'] for m in all_metrics]),
                    'std': np.std([m['recall'] for m in all_metrics])
                },
                'precision': {
                    'mean': np.mean([m['precision'] for m in all_metrics]),
                    'std': np.std([m['precision'] for m in all_metrics])
                },
                'num_runs': len(all_metrics)
            }
            
            # Create performance plot
            plot_path = plot_performance_across_runs(all_metrics, args.output_dir)
            
            with open(os.path.join(args.output_dir, "aggregated_metrics.json"), "w") as f:
                json.dump({
                    'average_metrics': avg_metrics,
                    'all_runs': all_metrics
                }, f, indent=2)
            
            print("\nAverage Metrics Across All Runs (± standard deviation):")
            logging.info("\nAverage Metrics Across All Runs (± standard deviation):")
            print(f"Accuracy: {avg_metrics['accuracy']['mean']:.4f} ± {avg_metrics['accuracy']['std']:.4f}")
            logging.info(f"Accuracy: {avg_metrics['accuracy']['mean']:.4f} ± {avg_metrics['accuracy']['std']:.4f}")
            print(f"F1 Score (weighted): {avg_metrics['f1']['mean']:.4f} ± {avg_metrics['f1']['std']:.4f}")
            logging.info(f"F1 Score (weighted): {avg_metrics['f1']['mean']:.4f} ± {avg_metrics['f1']['std']:.4f}")
            print(f"F1 Score (macro): {avg_metrics['macro_f1']['mean']:.4f} ± {avg_metrics['macro_f1']['std']:.4f}")
            logging.info(f"F1 Score (macro): {avg_metrics['macro_f1']['mean']:.4f} ± {avg_metrics['macro_f1']['std']:.4f}")
            print(f"Recall: {avg_metrics['recall']['mean']:.4f} ± {avg_metrics['recall']['std']:.4f}")
            logging.info(f"Recall: {avg_metrics['recall']['mean']:.4f} ± {avg_metrics['recall']['std']:.4f}")
            print(f"Precision: {avg_metrics['precision']['mean']:.4f} ± {avg_metrics['precision']['std']:.4f}")
            logging.info(f"Precision: {avg_metrics['precision']['mean']:.4f} ± {avg_metrics['precision']['std']:.4f}")
            print(f"Based on {avg_metrics['num_runs']} runs")
            logging.info(f"Based on {avg_metrics['num_runs']} runs")
            print(f"Aggregated results saved to: {os.path.join(args.output_dir, 'aggregated_metrics.json')}")
            logging.info(f"Aggregated results saved to: {os.path.join(args.output_dir, 'aggregated_metrics.json')}")
            print(f"Performance plot saved to: {plot_path}")
            logging.info(f"Performance plot saved to: {plot_path}")

    print("\n===== All Runs Completed =====")
    logging.info("\n===== All Runs Completed =====")
    print(f"Final results saved to: {args.output_dir}")
    logging.info(f"Final results saved to: {args.output_dir}")