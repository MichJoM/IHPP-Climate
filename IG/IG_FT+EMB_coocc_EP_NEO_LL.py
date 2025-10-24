import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from collections import defaultdict
from scipy.stats import mannwhitneyu
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""
Focused Analysis: EP and NEO Attribution vs. Loaded Language
For custom FT+EMB architecture models
Uses manual Integrated Gradients implementation
"""

# --- Custom Model Architecture ---

class HyperpartisanConfig(PretrainedConfig):
    model_type = "ft_emb_traits"

    def __init__(self, base_model_name="dbmdz/bert-base-italian-uncased", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name


class HyperpartisanModel(PreTrainedModel):
    config_class = HyperpartisanConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.model.config.hidden_size
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

        self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state

        # Average pool over text segment (segment 0, excluding special tokens)
        text_token_mask = (
            (token_type_ids == 0)
            & (input_ids != self.cls_token_id)
            & (input_ids != self.sep_token_id)
            & (input_ids != self.pad_token_id)
        ).float()
        text_features = (hidden * text_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / text_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

        # Average pool over traits segment (segment 1, excluding special tokens)
        traits_token_mask = (
            (token_type_ids == 1)
            & (input_ids != self.sep_token_id)
            & (input_ids != self.pad_token_id)
        ).float()
        traits_features = (hidden * traits_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / traits_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

        # Gating mechanism for fusion
        combined = torch.cat((text_features, traits_features), dim=1)
        gate = torch.sigmoid(self.fusion_linear(combined))
        fused = text_features * gate + traits_features * (1 - gate)

        return self.classifier(fused)


# --- Configuration ---

BASE_MODEL_DIR = '/home/michele.maggini/XAI_HIPP/models/FT_EMB_top/arch_new'
DATASET_PATH = '/home/michele.maggini/PORRO_2/datasets/HIPP_test.json'
OUTPUT_DIR = '/home/michele.maggini/XAI_HIPP/IG/results_EP_NEO_LL_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories
COOCCURRENCE_DIR = os.path.join(OUTPUT_DIR, "cooccurrence")
CONDITIONAL_DIR = os.path.join(OUTPUT_DIR, "conditional_attribution")
INTERACTION_DIR = os.path.join(OUTPUT_DIR, "interaction_effects")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

for dir_path in [COOCCURRENCE_DIR, CONDITIONAL_DIR, INTERACTION_DIR, SUMMARY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_TEXT_LENGTH = 256  # Match your training setup
TRAITS_OF_INTEREST = ["Epithet", "Neologism", "Loaded_language"]

# --- Helper Functions ---

def smart_truncate(text, max_length=MAX_TEXT_LENGTH):
    """Truncate text at sentence boundary"""
    if len(text) <= max_length:
        return text
    sentence_ends = [".", "!", "?"]
    for i in range(max_length - 1, max_length // 2, -1):
        if text[i] in sentence_ends:
            return text[:i + 1]
    return text[:max_length]


def load_dataset(filepath):
    """Load and prepare dataset"""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
        if "text" in df.columns:
            df["text"] = df["text"].apply(
                lambda x: smart_truncate(x) if isinstance(x, str) else x
            )
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()


def load_model_metadata(model_dir):
    """Load ablation results and test metrics for a model"""
    ablation_path = os.path.join(model_dir, "ablation_results.json")
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    
    metadata = {
        "model_dir": model_dir,
        "model_name": os.path.basename(model_dir)
    }
    
    if os.path.exists(ablation_path):
        with open(ablation_path, 'r') as f:
            metadata["ablation"] = json.load(f)
    else:
        metadata["ablation"] = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metadata["baseline_metrics"] = json.load(f)
    else:
        metadata["baseline_metrics"] = {}
    
    return metadata


# --- Manual Integrated Gradients Implementation ---
from captum.attr import IntegratedGradients

from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, input_ids, attention_mask, token_type_ids, 
                                target_class=1, n_steps=50):
    """
    Compute Integrated Gradients for token attributions using captum
    
    Args:
        model: The model to explain
        input_ids: Input token IDs [1, seq_len]
        attention_mask: Attention mask [1, seq_len]
        token_type_ids: Token type IDs [1, seq_len]
        target_class: Class to compute gradients for
        n_steps: Number of integration steps
        
    Returns:
        attributions: Token-level attributions [seq_len]
    """
    model.eval()
    
    # Get embeddings
    try:
        embeddings = model.model.embeddings.word_embeddings(input_ids)
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("Embeddings output is not a tensor")
    except Exception as e:
        print(f"Error in embedding computation: {e}")
        return np.zeros(input_ids.shape[1])
    
    # Define custom forward function
    def custom_forward(inputs_embeds):
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden = outputs.last_hidden_state
        
        # Replicate the model's forward logic for classification
        text_token_mask = (
            (token_type_ids == 0)
            & (attention_mask == 1)
        ).float()
        text_features = (hidden * text_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / text_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)
        
        traits_token_mask = (
            (token_type_ids == 1)
            & (attention_mask == 1)
        ).float()
        traits_features = (hidden * traits_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / traits_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)
        
        combined = torch.cat((text_features, traits_features), dim=1)
        gate = torch.sigmoid(model.fusion_linear(combined))
        fused = text_features * gate + traits_features * (1 - gate)
        logits = model.classifier(fused)
        
        return logits[:, target_class]
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(custom_forward)
    
    # Compute baseline
    baseline = torch.zeros_like(embeddings)
    
    try:
        # Compute attributions
        attributions = ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            target=None,  # Use predicted class from forward pass
            n_steps=n_steps,
            method="gausslegendre",
            return_convergence_delta=False
        )
        # Sum over embedding dimension to get per-token attributions
        attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    except Exception as e:
        print(f"Error in Integrated Gradients computation: {e}")
        return np.zeros(input_ids.shape[1])
    
    return attributions


def get_token_attributions(model, tokenizer, text, traits_text):
    """
    Get token attributions for a text sample
    
    Returns:
        tokens: List of tokens
        attributions: List of attribution scores (same length as tokens)
    """
    # Tokenize
    encoding = tokenizer(
        text,
        traits_text,
        return_tensors="pt",
        padding='max_length',
        max_length=MAX_TEXT_LENGTH,
        truncation=True,
        return_token_type_ids=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Predict the class first
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
    predicted_class = logits.argmax(dim=-1).item()
    
    # Compute attributions without no_grad, as gradients are needed
    attributions = compute_integrated_gradients(
        model, input_ids, attention_mask, token_type_ids, predicted_class
    )
    
    return tokens, attributions


def normalize_attributions(attributions):
    """Normalize attribution scores using robust scaling"""
    if len(attributions) == 0:
        return attributions
    
    q1 = np.percentile(attributions, 10)
    q3 = np.percentile(attributions, 90)
    
    if abs(q3 - q1) < 1e-6:
        return np.full_like(attributions, 0.5)
    
    normalized = np.clip((attributions - q1) / (q3 - q1), 0, 1)
    return normalized


def get_tokens_in_span(tokens, full_text, span_start, span_end):
    """
    Identify token indices within a character span
    
    Returns:
        token_indices: List of token indices within the span
    """
    token_indices = []
    reconstructed_text = ""
    
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        clean_token = token.replace("##", "").replace(" ", "").strip()
        if not clean_token:
            continue
        
        # Find token position in text
        search_start = max(0, len(reconstructed_text) - len(clean_token) // 2)
        token_position = -1
        
        for offset in range(min(len(full_text) - search_start, 5)):
            current_search_start = search_start + offset
            if 0 <= current_search_start < len(full_text):
                if full_text[current_search_start:].startswith(clean_token):
                    token_position = current_search_start
                    break
        
        if token_position == -1:
            token_position = full_text.find(clean_token, search_start)
        
        if token_position != -1:
            token_end = token_position + len(clean_token)
            # Check if token overlaps with span
            if not (token_position >= span_end or token_end <= span_start):
                token_indices.append(i)
            reconstructed_text = full_text[:max(len(reconstructed_text), token_end)]
    
    return token_indices


def get_span_attribution(attributions, token_indices):
    """Calculate median attribution for a span"""
    if not token_indices:
        return 0.0
    
    valid_indices = [idx for idx in token_indices if 0 <= idx < len(attributions)]
    if not valid_indices:
        return 0.0
    
    attribution_values = [attributions[idx] for idx in valid_indices]
    return np.median(attribution_values) if attribution_values else 0.0


# --- Co-occurrence Analysis ---

def analyze_cooccurrence_focused(df):
    """
    Analyze co-occurrence specifically for EP, NEO, and LL
    """
    print("\n=== Analyzing EP, NEO, and LL Co-occurrence ===")
    
    cooccurrence_stats = {
        "EP_alone": 0,
        "EP_with_LL": 0,
        "EP_with_NEO": 0,
        "EP_with_both": 0,
        "NEO_alone": 0,
        "NEO_with_LL": 0,
        "NEO_with_EP": 0,
        "NEO_with_both": 0,
        "LL_alone": 0,
        "LL_with_EP": 0,
        "LL_with_NEO": 0,
        "LL_with_both": 0,
        "all_three": 0,
        "total_paragraphs": 0
    }
    
    for idx, row in df.iterrows():
        linguistic_traits = row.get("linguistic_traits", {})
        if not isinstance(linguistic_traits, dict):
            continue
        
        cooccurrence_stats["total_paragraphs"] += 1
        
        has_EP = "Epithet" in linguistic_traits and linguistic_traits["Epithet"]
        has_NEO = "Neologism" in linguistic_traits and linguistic_traits["Neologism"]
        has_LL = "Loaded_language" in linguistic_traits and linguistic_traits["Loaded_language"]
        
        # Count combinations
        if has_EP and has_NEO and has_LL:
            cooccurrence_stats["all_three"] += 1
            cooccurrence_stats["EP_with_both"] += 1
            cooccurrence_stats["NEO_with_both"] += 1
            cooccurrence_stats["LL_with_both"] += 1
        elif has_EP and has_LL:
            cooccurrence_stats["EP_with_LL"] += 1
            cooccurrence_stats["LL_with_EP"] += 1
        elif has_EP and has_NEO:
            cooccurrence_stats["EP_with_NEO"] += 1
            cooccurrence_stats["NEO_with_EP"] += 1
        elif has_NEO and has_LL:
            cooccurrence_stats["NEO_with_LL"] += 1
            cooccurrence_stats["LL_with_NEO"] += 1
        elif has_EP:
            cooccurrence_stats["EP_alone"] += 1
        elif has_NEO:
            cooccurrence_stats["NEO_alone"] += 1
        elif has_LL:
            cooccurrence_stats["LL_alone"] += 1
    
    # Calculate conditional probabilities
    total_EP = cooccurrence_stats["EP_alone"] + cooccurrence_stats["EP_with_LL"] + \
               cooccurrence_stats["EP_with_NEO"] + cooccurrence_stats["EP_with_both"]
    total_NEO = cooccurrence_stats["NEO_alone"] + cooccurrence_stats["NEO_with_LL"] + \
                cooccurrence_stats["NEO_with_EP"] + cooccurrence_stats["NEO_with_both"]
    total_LL = cooccurrence_stats["LL_alone"] + cooccurrence_stats["LL_with_EP"] + \
               cooccurrence_stats["LL_with_NEO"] + cooccurrence_stats["LL_with_both"]
    
    conditional_probs = {
        "P(LL|EP)": (cooccurrence_stats["EP_with_LL"] + cooccurrence_stats["EP_with_both"]) / total_EP if total_EP > 0 else 0,
        "P(NEO|EP)": (cooccurrence_stats["EP_with_NEO"] + cooccurrence_stats["EP_with_both"]) / total_EP if total_EP > 0 else 0,
        "P(LL|NEO)": (cooccurrence_stats["NEO_with_LL"] + cooccurrence_stats["NEO_with_both"]) / total_NEO if total_NEO > 0 else 0,
        "P(EP|NEO)": (cooccurrence_stats["NEO_with_EP"] + cooccurrence_stats["NEO_with_both"]) / total_NEO if total_NEO > 0 else 0,
        "P(EP|LL)": (cooccurrence_stats["LL_with_EP"] + cooccurrence_stats["LL_with_both"]) / total_LL if total_LL > 0 else 0,
        "P(NEO|LL)": (cooccurrence_stats["LL_with_NEO"] + cooccurrence_stats["LL_with_both"]) / total_LL if total_LL > 0 else 0,
    }
    
    # Save results
    with open(os.path.join(COOCCURRENCE_DIR, 'EP_NEO_LL_cooccurrence.json'), 'w') as f:
        json.dump({
            "counts": cooccurrence_stats,
            "totals": {"EP": total_EP, "NEO": total_NEO, "LL": total_LL},
            "conditional_probabilities": conditional_probs
        }, f, indent=2)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of co-occurrence patterns
    patterns = ['EP alone', 'EP+LL', 'EP+NEO', 'EP+LL+NEO', 
                'NEO alone', 'NEO+LL', 'LL alone']
    counts = [
        cooccurrence_stats["EP_alone"],
        cooccurrence_stats["EP_with_LL"],
        cooccurrence_stats["EP_with_NEO"],
        cooccurrence_stats["all_three"],
        cooccurrence_stats["NEO_alone"],
        cooccurrence_stats["NEO_with_LL"],
        cooccurrence_stats["LL_alone"]
    ]
    
    ax1.barh(patterns, counts, color=['#ff7f0e', '#ff7f0e', '#ff7f0e', '#d62728',
                                       '#2ca02c', '#2ca02c', '#8c564b'])
    ax1.set_xlabel('Count', fontsize=12)
    ax1.set_title('Co-occurrence Patterns', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Conditional probability heatmap
    cond_prob_matrix = np.array([
        [1.0, conditional_probs["P(LL|EP)"], conditional_probs["P(NEO|EP)"]],
        [conditional_probs["P(EP|LL)"], 1.0, conditional_probs["P(NEO|LL)"]],
        [conditional_probs["P(EP|NEO)"], conditional_probs["P(LL|NEO)"], 1.0]
    ])
    
    sns.heatmap(cond_prob_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['EP', 'LL', 'NEO'],
                yticklabels=['EP', 'LL', 'NEO'],
                ax=ax2, vmin=0, vmax=1, square=True)
    ax2.set_title('Conditional Probabilities: P(Column | Row)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(COOCCURRENCE_DIR, 'EP_NEO_LL_cooccurrence.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Co-occurrence analysis saved to {COOCCURRENCE_DIR}")
    print(f"\n  Key Statistics:")
    print(f"    P(LL|EP) = {conditional_probs['P(LL|EP)']:.3f}")
    print(f"    P(LL|NEO) = {conditional_probs['P(LL|NEO)']:.3f}")
    print(f"    P(NEO|EP) = {conditional_probs['P(NEO|EP)']:.3f}")
    
    return cooccurrence_stats, conditional_probs


# --- Conditional Attribution Analysis ---

def analyze_conditional_attribution_focused(model, tokenizer, df, model_name):
    """
    Analyze EP and NEO attribution conditioned on LL presence
    """
    print(f"\n=== Analyzing Conditional Attribution for {model_name} ===")
    
    # Storage for different conditions
    attribution_data = {
        "EP_alone": [],
        "EP_with_LL": [],
        "EP_without_LL": [],
        "NEO_alone": [],
        "NEO_with_LL": [],
        "NEO_without_LL": [],
        "LL_alone": [],
        "LL_with_EP": [],
        "LL_with_NEO": [],
        "LL_with_both": []
    }
    
    # Sample for faster processing (optional - remove if you want full dataset)
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing samples"):
        text = row.get("text", "")
        linguistic_traits = row.get("linguistic_traits", {})
        
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(linguistic_traits, dict):
            continue
        
        try:
            # Prepare traits text
            traits_spans = []
            for spans in linguistic_traits.values():
                if isinstance(spans, list):
                    traits_spans.extend([span.get('text', '') for span in spans if isinstance(span, dict)])
                elif isinstance(spans, dict):
                    traits_spans.append(spans.get('text', ''))
            traits_text = " [SEP] ".join(traits_spans) if traits_spans else "[PAD]"
            
            # Get attributions
            tokens, attributions = get_token_attributions(model, tokenizer, text, traits_text)
            
            if len(attributions) == 0:
                continue
            
            # Normalize
            normalized_attributions = normalize_attributions(attributions)
            
            # Check which traits are present
            has_EP = "Epithet" in linguistic_traits and linguistic_traits["Epithet"]
            has_NEO = "Neologism" in linguistic_traits and linguistic_traits["Neologism"]
            has_LL = "Loaded_language" in linguistic_traits and linguistic_traits["Loaded_language"]
            
            # Process Epithet
            if has_EP:
                spans = linguistic_traits["Epithet"]
                if isinstance(spans, dict):
                    spans = [spans]
                
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    
                    span_start = span.get("start")
                    span_end = span.get("end")
                    
                    if span_start is None or span_end is None:
                        continue
                    
                    token_indices = get_tokens_in_span(tokens, text, span_start, span_end)
                    
                    if not token_indices:
                        continue
                    
                    span_attribution = get_span_attribution(normalized_attributions, token_indices)
                    
                    # Categorize
                    if not has_LL and not has_NEO:
                        attribution_data["EP_alone"].append(span_attribution)
                    
                    if has_LL:
                        attribution_data["EP_with_LL"].append(span_attribution)
                    else:
                        attribution_data["EP_without_LL"].append(span_attribution)
            
            # Process Neologism
            if has_NEO:
                spans = linguistic_traits["Neologism"]
                if isinstance(spans, dict):
                    spans = [spans]
                
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    
                    span_start = span.get("start")
                    span_end = span.get("end")
                    
                    if span_start is None or span_end is None:
                        continue
                    
                    token_indices = get_tokens_in_span(tokens, text, span_start, span_end)
                    
                    if not token_indices:
                        continue
                    
                    span_attribution = get_span_attribution(normalized_attributions, token_indices)
                    
                    # Categorize
                    if not has_LL and not has_EP:
                        attribution_data["NEO_alone"].append(span_attribution)
                    
                    if has_LL:
                        attribution_data["NEO_with_LL"].append(span_attribution)
                    else:
                        attribution_data["NEO_without_LL"].append(span_attribution)
            
            # Process Loaded_language
            if has_LL:
                spans = linguistic_traits["Loaded_language"]
                if isinstance(spans, dict):
                    spans = [spans]
                
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    
                    span_start = span.get("start")
                    span_end = span.get("end")
                    
                    if span_start is None or span_end is None:
                        continue
                    
                    token_indices = get_tokens_in_span(tokens, text, span_start, span_end)
                    
                    if not token_indices:
                        continue
                    
                    span_attribution = get_span_attribution(normalized_attributions, token_indices)
                    
                    # Categorize
                    if not has_EP and not has_NEO:
                        attribution_data["LL_alone"].append(span_attribution)
                    elif has_EP and not has_NEO:
                        attribution_data["LL_with_EP"].append(span_attribution)
                    elif has_NEO and not has_EP:
                        attribution_data["LL_with_NEO"].append(span_attribution)
                    elif has_EP and has_NEO:
                        attribution_data["LL_with_both"].append(span_attribution)
        
        except Exception as e:
            print(f"    Error on sample {idx}: {e}")
            continue
    
    # Compute statistics
    results = []
    
    for condition, values in attribution_data.items():
        if values:
            results.append({
                'model': model_name,
                'condition': condition,
                'n': len(values),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            })
    
    results_df = pd.DataFrame(results)
    
    # Statistical tests
    comparisons = []
    
    # EP: with_LL vs. without_LL
    if attribution_data["EP_with_LL"] and attribution_data["EP_without_LL"]:
        u_stat, p_value = mannwhitneyu(
            attribution_data["EP_with_LL"],
            attribution_data["EP_without_LL"],
            alternative='two-sided'
        )
        effect_size = np.median(attribution_data["EP_with_LL"]) - \
                      np.median(attribution_data["EP_without_LL"])
        
        comparisons.append({
            'model': model_name,
            'trait': 'Epithet',
            'comparison': 'with_LL vs without_LL',
            'median_with_LL': float(np.median(attribution_data["EP_with_LL"])),
            'median_without_LL': float(np.median(attribution_data["EP_without_LL"])),
            'effect_size': float(effect_size),
            'mannwhitney_u': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })
    
    # NEO: with_LL vs. without_LL
    if attribution_data["NEO_with_LL"] and attribution_data["NEO_without_LL"]:
        u_stat, p_value = mannwhitneyu(
            attribution_data["NEO_with_LL"],
            attribution_data["NEO_without_LL"],
            alternative='two-sided'
        )
        effect_size = np.median(attribution_data["NEO_with_LL"]) - \
                      np.median(attribution_data["NEO_without_LL"])
        
        comparisons.append({
            'model': model_name,
            'trait': 'Neologism',
            'comparison': 'with_LL vs without_LL',
            'median_with_LL': float(np.median(attribution_data["NEO_with_LL"])),
            'median_without_LL': float(np.median(attribution_data["NEO_without_LL"])),
            'effect_size': float(effect_size),
            'mannwhitney_u': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })
    
    comparisons_df = pd.DataFrame(comparisons)
    
    # Save results
    model_safe_name = model_name.replace('/', '_').replace('-', '_')
    results_df.to_csv(
        os.path.join(CONDITIONAL_DIR, f'conditional_stats_{model_safe_name}.csv'),
        index=False
    )
    comparisons_df.to_csv(
        os.path.join(CONDITIONAL_DIR, f'statistical_tests_{model_safe_name}.csv'),
        index=False
    )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # EP comparison
    ep_data = []
    ep_labels = []
    
    if attribution_data["EP_alone"]:
        ep_data.append(attribution_data["EP_alone"])
        ep_labels.append('EP alone')
    if attribution_data["EP_without_LL"]:
        ep_data.append(attribution_data["EP_without_LL"])
        ep_labels.append('EP w/o LL')
    if attribution_data["EP_with_LL"]:
        ep_data.append(attribution_data["EP_with_LL"])
        ep_labels.append('EP w/ LL')
    
    if ep_data:
        bp1 = axes[0].boxplot(ep_data, labels=ep_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('#ff7f0e')
        axes[0].set_title(f'Epithet Attribution by Condition\n{model_name}', fontsize=12)
        axes[0].set_ylabel('Attribution Score', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
    else:
        axes[0].text(0.5, 0.5, 'No EP data', ha='center', va='center', transform=axes[0].transAxes)
    
    # NEO comparison
    neo_data = []
    neo_labels = []
    
    if attribution_data["NEO_alone"]:
        neo_data.append(attribution_data["NEO_alone"])
        neo_labels.append('NEO alone')
    if attribution_data["NEO_without_LL"]:
        neo_data.append(attribution_data["NEO_without_LL"])
        neo_labels.append('NEO w/o LL')
    if attribution_data["NEO_with_LL"]:
        neo_data.append(attribution_data["NEO_with_LL"])
        neo_labels.append('NEO w/ LL')
    
    if neo_data:
        bp2 = axes[1].boxplot(neo_data, labels=neo_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#2ca02c')
        axes[1].set_title(f'Neologism Attribution by Condition\n{model_name}', fontsize=12)
        axes[1].set_ylabel('Attribution Score', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    else:
        axes[1].text(0.5, 0.5, 'No NEO data', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONDITIONAL_DIR, f'conditional_comparison_{model_safe_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    print(f"  Conditional attribution saved to {CONDITIONAL_DIR}")
    
    return results_df, comparisons_df, attribution_data


# --- Interaction Effect Analysis ---

def analyze_interaction_effects(attribution_data, model_name, ablation_results):
    """
    Quantify boosting effects and compare with ablation results
    """
    print(f"\n=== Analyzing Interaction Effects for {model_name} ===")
    
    interaction_results = []
    
    # Get baseline F1 from ablation results
    baseline_f1 = ablation_results.get("_baseline_f1", np.nan)
    
    # EP interaction with LL
    if attribution_data["EP_alone"] and attribution_data["EP_with_LL"]:
        median_alone = np.median(attribution_data["EP_alone"])
        median_with_LL = np.median(attribution_data["EP_with_LL"])
        
        if median_alone != 0:
            boosting_factor = (median_with_LL - median_alone) / abs(median_alone)
        else:
            boosting_factor = median_with_LL  # If alone is 0, boosting is just the with_LL value
        
        # Get ablation impact
        ep_ablation = ablation_results.get("Epithet", {})
        ablation_f1 = ep_ablation.get("f1", baseline_f1)
        ablation_f1_drop = (ablation_f1 - baseline_f1) if not np.isnan(baseline_f1) else np.nan
        
        interaction_results.append({
            'model': model_name,
            'trait': 'Epithet',
            'n_alone': len(attribution_data["EP_alone"]),
            'n_with_LL': len(attribution_data["EP_with_LL"]),
            'median_alone': float(median_alone),
            'median_with_LL': float(median_with_LL),
            'absolute_difference': float(median_with_LL - median_alone),
            'boosting_factor': float(boosting_factor),
            'baseline_f1': float(baseline_f1) if not np.isnan(baseline_f1) else None,
            'ablation_f1': float(ablation_f1) if not np.isnan(ablation_f1) else None,
            'ablation_f1_drop': float(ablation_f1_drop) if not np.isnan(ablation_f1_drop) else None,
            'ablation_significant': ep_ablation.get("significant", False)
        })
    
    # NEO interaction with LL
    if attribution_data["NEO_alone"] and attribution_data["NEO_with_LL"]:
        median_alone = np.median(attribution_data["NEO_alone"])
        median_with_LL = np.median(attribution_data["NEO_with_LL"])
        
        if median_alone != 0:
            boosting_factor = (median_with_LL - median_alone) / abs(median_alone)
        else:
            boosting_factor = median_with_LL
        
        # Get ablation impact
        neo_ablation = ablation_results.get("Neologism", {})
        ablation_f1 = neo_ablation.get("f1", baseline_f1)
        ablation_f1_drop = (ablation_f1 - baseline_f1) if not np.isnan(baseline_f1) else np.nan
        
        interaction_results.append({
            'model': model_name,
            'trait': 'Neologism',
            'n_alone': len(attribution_data["NEO_alone"]),
            'n_with_LL': len(attribution_data["NEO_with_LL"]),
            'median_alone': float(median_alone),
            'median_with_LL': float(median_with_LL),
            'absolute_difference': float(median_with_LL - median_alone),
            'boosting_factor': float(boosting_factor),
            'baseline_f1': float(baseline_f1) if not np.isnan(baseline_f1) else None,
            'ablation_f1': float(ablation_f1) if not np.isnan(ablation_f1) else None,
            'ablation_f1_drop': float(ablation_f1_drop) if not np.isnan(ablation_f1_drop) else None,
            'ablation_significant': neo_ablation.get("significant", False)
        })
    
    interaction_df = pd.DataFrame(interaction_results)
    
    # Save results
    model_safe_name = model_name.replace('/', '_').replace('-', '_')
    interaction_df.to_csv(
        os.path.join(INTERACTION_DIR, f'interaction_effects_{model_safe_name}.csv'),
        index=False
    )
    
    print(f"  Interaction effects saved to {INTERACTION_DIR}")
    
    return interaction_df


# --- Cross-Model Summary ---

def create_cross_model_summary(all_results):
    """
    Create summary comparing all models
    """
    print("\n=== Creating Cross-Model Summary ===")
    
    # Combine all results
    all_conditional = pd.concat([r['conditional'] for r in all_results.values() if not r['conditional'].empty], ignore_index=True)
    all_comparisons = pd.concat([r['comparisons'] for r in all_results.values() if not r['comparisons'].empty], ignore_index=True)
    all_interactions = pd.concat([r['interactions'] for r in all_results.values() if not r['interactions'].empty], ignore_index=True)
    
    # Save combined results
    all_conditional.to_csv(
        os.path.join(SUMMARY_DIR, 'all_models_conditional_stats.csv'),
        index=False
    )
    all_comparisons.to_csv(
        os.path.join(SUMMARY_DIR, 'all_models_statistical_tests.csv'),
        index=False
    )
    all_interactions.to_csv(
        os.path.join(SUMMARY_DIR, 'all_models_interactions.csv'),
        index=False
    )
    
    # Create summary visualizations
    
    # 1. Boosting factors across models
    if not all_interactions.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # EP boosting
        ep_interactions = all_interactions[all_interactions['trait'] == 'Epithet']
        if not ep_interactions.empty:
            models = ep_interactions['model'].values
            boosting = ep_interactions['boosting_factor'].values
            ablation_sig = ep_interactions['ablation_significant'].values
            colors = ['red' if sig else 'gray' for sig in ablation_sig]
            
            axes[0].barh(range(len(models)), boosting, color=colors)
            axes[0].set_yticks(range(len(models)))
            axes[0].set_yticklabels(models, fontsize=9)
            axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
            axes[0].set_xlabel('Boosting Factor (%)', fontsize=12)
            axes[0].set_title('Epithet: LL Boosting Effect Across Models\n(Red = significant ablation impact)', 
                             fontsize=13)
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # NEO boosting
        neo_interactions = all_interactions[all_interactions['trait'] == 'Neologism']
        if not neo_interactions.empty:
            models = neo_interactions['model'].values
            boosting = neo_interactions['boosting_factor'].values
            ablation_sig = neo_interactions['ablation_significant'].values
            colors = ['red' if sig else 'gray' for sig in ablation_sig]
            
            axes[1].barh(range(len(models)), boosting, color=colors)
            axes[1].set_yticks(range(len(models)))
            axes[1].set_yticklabels(models, fontsize=9)
            axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Boosting Factor (%)', fontsize=12)
            axes[1].set_title('Neologism: LL Boosting Effect Across Models\n(Red = significant ablation impact)', 
                             fontsize=13)
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(SUMMARY_DIR, 'cross_model_boosting_comparison.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    # 2. Attribution comparison heatmap
    if not all_conditional.empty:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        model_names = all_conditional['model'].unique()
        
        # EP heatmap
        ep_conditions = ['EP_alone', 'EP_without_LL', 'EP_with_LL']
        ep_data = []
        ep_models = []
        
        for model_name in model_names:
            model_conditional = all_conditional[all_conditional['model'] == model_name]
            row_data = []
            has_data = False
            for condition in ep_conditions:
                cond_data = model_conditional[model_conditional['condition'] == condition]
                if not cond_data.empty:
                    row_data.append(cond_data['median'].values[0])
                    has_data = True
                else:
                    row_data.append(np.nan)
            if has_data:
                ep_data.append(row_data)
                ep_models.append(model_name)
        
        if ep_data:
            ep_array = np.array(ep_data)
            sns.heatmap(ep_array, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=['EP alone', 'EP w/o LL', 'EP w/ LL'],
                       yticklabels=ep_models, ax=axes[0], cbar_kws={'label': 'Median Attribution'},
                       vmin=0, vmax=1)
            axes[0].set_title('Epithet: Median Attribution Across Conditions and Models', fontsize=13)
        else:
            axes[0].text(0.5, 0.5, 'No EP data across models', ha='center', va='center', 
                        transform=axes[0].transAxes, fontsize=14)
        
        # NEO heatmap
        neo_conditions = ['NEO_alone', 'NEO_without_LL', 'NEO_with_LL']
        neo_data = []
        neo_models = []
        
        for model_name in model_names:
            model_conditional = all_conditional[all_conditional['model'] == model_name]
            row_data = []
            has_data = False
            for condition in neo_conditions:
                cond_data = model_conditional[model_conditional['condition'] == condition]
                if not cond_data.empty:
                    row_data.append(cond_data['median'].values[0])
                    has_data = True
                else:
                    row_data.append(np.nan)
            if has_data:
                neo_data.append(row_data)
                neo_models.append(model_name)
        
        if neo_data:
            neo_array = np.array(neo_data)
            sns.heatmap(neo_array, annot=True, fmt='.3f', cmap='YlGnBu',
                       xticklabels=['NEO alone', 'NEO w/o LL', 'NEO w/ LL'],
                       yticklabels=neo_models, ax=axes[1], cbar_kws={'label': 'Median Attribution'},
                       vmin=0, vmax=1)
            axes[1].set_title('Neologism: Median Attribution Across Conditions and Models', fontsize=13)
        else:
            axes[1].text(0.5, 0.5, 'No NEO data across models', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(SUMMARY_DIR, 'cross_model_attribution_heatmap.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    # 3. Create interpretation report
    interpretation_report = []
    
    for model_name in all_results.keys():
        if all_interactions.empty:
            continue
            
        model_interactions = all_interactions[all_interactions['model'] == model_name]
        model_comparisons = all_comparisons[all_comparisons['model'] == model_name] if not all_comparisons.empty else pd.DataFrame()
        
        # EP interpretation
        ep_int = model_interactions[model_interactions['trait'] == 'Epithet']
        ep_comp = model_comparisons[model_comparisons['trait'] == 'Epithet'] if not model_comparisons.empty else pd.DataFrame()
        
        if not ep_int.empty:
            ep_boosting = ep_int['boosting_factor'].values[0]
            ep_ablation_sig = ep_int['ablation_significant'].values[0]
            ep_ablation_drop = ep_int['ablation_f1_drop'].values[0]
            ep_p_value = ep_comp['p_value'].values[0] if not ep_comp.empty else np.nan
            
            if abs(ep_boosting) > 0.5:
                boosting_desc = f"strong {'positive' if ep_boosting > 0 else 'negative'} boosting ({ep_boosting:+.1%})"
            elif abs(ep_boosting) > 0.2:
                boosting_desc = f"moderate {'positive' if ep_boosting > 0 else 'negative'} boosting ({ep_boosting:+.1%})"
            else:
                boosting_desc = f"minimal boosting effect ({ep_boosting:+.1%})"
            
            if ep_ablation_sig:
                ablation_desc = f"significant independent contribution (ΔF1: {ep_ablation_drop:+.3f})"
            else:
                ablation_desc = f"no significant ablation impact (ΔF1: {ep_ablation_drop:+.3f})"
            
            if not np.isnan(ep_p_value) and ep_p_value < 0.05:
                stat_sig = "statistically significant difference"
            else:
                stat_sig = "no significant difference"
            
            ep_interpretation = f"EP shows {boosting_desc} from LL co-occurrence ({stat_sig}), with {ablation_desc}."
        else:
            ep_interpretation = "Insufficient EP data for interpretation."
        
        # NEO interpretation
        neo_int = model_interactions[model_interactions['trait'] == 'Neologism']
        neo_comp = model_comparisons[model_comparisons['trait'] == 'Neologism'] if not model_comparisons.empty else pd.DataFrame()
        
        if not neo_int.empty:
            neo_boosting = neo_int['boosting_factor'].values[0]
            neo_ablation_sig = neo_int['ablation_significant'].values[0]
            neo_ablation_drop = neo_int['ablation_f1_drop'].values[0]
            neo_p_value = neo_comp['p_value'].values[0] if not neo_comp.empty else np.nan
            
            if abs(neo_boosting) > 0.5:
                boosting_desc = f"strong {'positive' if neo_boosting > 0 else 'negative'} boosting ({neo_boosting:+.1%})"
            elif abs(neo_boosting) > 0.2:
                boosting_desc = f"moderate {'positive' if neo_boosting > 0 else 'negative'} boosting ({neo_boosting:+.1%})"
            else:
                boosting_desc = f"minimal boosting effect ({neo_boosting:+.1%})"
            
            if neo_ablation_sig:
                ablation_desc = f"significant independent contribution (ΔF1: {neo_ablation_drop:+.3f})"
            else:
                ablation_desc = f"no significant ablation impact (ΔF1: {neo_ablation_drop:+.3f})"
            
            if not np.isnan(neo_p_value) and neo_p_value < 0.05:
                stat_sig = "statistically significant difference"
            else:
                stat_sig = "no significant difference"
            
            neo_interpretation = f"NEO shows {boosting_desc} from LL co-occurrence ({stat_sig}), with {ablation_desc}."
        else:
            neo_interpretation = "Insufficient NEO data for interpretation."
        
        interpretation_report.append({
            'model': model_name,
            'EP_interpretation': ep_interpretation,
            'NEO_interpretation': neo_interpretation
        })
    
    interpretation_df = pd.DataFrame(interpretation_report)
    interpretation_df.to_csv(
        os.path.join(SUMMARY_DIR, 'interpretation_report.csv'),
        index=False
    )
    
    # Print summary to console
    print("\n" + "="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    for _, row in interpretation_df.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"  Epithet: {row['EP_interpretation']}")
        print(f"  Neologism: {row['NEO_interpretation']}")
    print("="*80)
    
    print(f"\n  Cross-model summary saved to {SUMMARY_DIR}")


# --- Main Execution ---

def main():
    print("="*80)
    print("EP and NEO Attribution Analysis Relative to Loaded Language")
    print("Custom FT+EMB Architecture Models - Manual IG Implementation")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    
    if df.empty:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # Step 1: Co-occurrence Analysis (dataset-level, only once)
    cooccurrence_stats, conditional_probs = analyze_cooccurrence_focused(df)
    
    # Step 2: Find all model directories
    model_dirs = [
        os.path.join(BASE_MODEL_DIR, d) 
        for d in os.listdir(BASE_MODEL_DIR) 
        if d.startswith('arch-') and os.path.isdir(os.path.join(BASE_MODEL_DIR, d))
    ]
    
    if not model_dirs:
        print(f"No model directories found in {BASE_MODEL_DIR}")
        return
    
    print(f"\nFound {len(model_dirs)} models to analyze")
    
    # Storage for all model results
    all_results = {}
    
    # Step 3: Process each model
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model metadata (ablation results, baseline metrics)
            metadata = load_model_metadata(model_dir)
            
            # Extract baseline F1 from test_metrics
            baseline_f1 = metadata['baseline_metrics'].get('f1', np.nan)
            
            # Add baseline to ablation results for easy access
            ablation_with_baseline = metadata['ablation'].copy()
            ablation_with_baseline['_baseline_f1'] = baseline_f1
            
            print(f"  Baseline F1: {baseline_f1:.4f}" if not np.isnan(baseline_f1) else "  Baseline F1: Not available")
            
            # Load the model
            print(f"  Loading model from {model_dir}")
            config = HyperpartisanConfig.from_pretrained(
                model_dir, 
                trust_remote_code=True
            )
            
            print(f"  Base model: {config.base_model_name}")
            
            # Load tokenizer from base model
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name,
                trust_remote_code=True
            )
            
            # Load custom model
            model = HyperpartisanModel.from_pretrained(
                model_dir,
                config=config,
                trust_remote_code=True
            )
            model = model.to(device)
            model.eval()
            
            print(f"  Model loaded successfully")
            
            # Analyze conditional attribution
            conditional_results, comparisons, attribution_data = analyze_conditional_attribution_focused(
                model, tokenizer, df, model_name
            )
            
            # Analyze interaction effects
            interaction_results = analyze_interaction_effects(
                attribution_data, model_name, ablation_with_baseline
            )
            
            # Store results
            all_results[model_name] = {
                'conditional': conditional_results,
                'comparisons': comparisons,
                'interactions': interaction_results,
                'attribution_data': attribution_data,
                'metadata': metadata
            }
            
            # Cleanup
            model.cpu()
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  {model_name} processing complete")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            if 'model' in locals():
                try:
                    model.cpu()
                    del model
                except:
                    pass
            if 'tokenizer' in locals():
                try:
                    del tokenizer
                except:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Step 4: Create cross-model summary
    if all_results:
        create_cross_model_summary(all_results)
    else:
        print("\nNo results to summarize.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*80)
    
    # Print file structure
    print("\nGenerated files:")
    print(f"  {COOCCURRENCE_DIR}/")
    print(f"    - EP_NEO_LL_cooccurrence.json")
    print(f"    - EP_NEO_LL_cooccurrence.png")
    print(f"  {CONDITIONAL_DIR}/")
    print(f"    - conditional_stats_[model].csv (per model)")
    print(f"    - statistical_tests_[model].csv (per model)")
    print(f"    - conditional_comparison_[model].png (per model)")
    print(f"  {INTERACTION_DIR}/")
    print(f"    - interaction_effects_[model].csv (per model)")
    print(f"  {SUMMARY_DIR}/")
    print(f"    - all_models_conditional_stats.csv")
    print(f"    - all_models_statistical_tests.csv")
    print(f"    - all_models_interactions.csv")
    print(f"    - cross_model_boosting_comparison.png")
    print(f"    - cross_model_attribution_heatmap.png")
    print(f"    - interpretation_report.csv")


if __name__ == "__main__":
    main()
