import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu, iqr
from sklearn.metrics import roc_curve, auc
import json
from sklearn.utils import resample
from captum.attr import IntegratedGradients

import gc  # For explicit garbage collection
from tqdm import tqdm  # For better progress tracking

# --- Setup code: device check, directories, model list, df load ---
# Check if CUDA is available and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the base directory for fine-tuned models
base_model_dir = "/home/michele.maggini/XAI_HIPP/models/FT_EMB_top/arch_new"

# Define base output directory for all results
base_output_dir = "/home/michele.maggini/XAI_HIPP/IG/results_FT_EMB_ARCH_NEW_ALL"

# Set maximum text length to avoid token length issues (512 is BERT's max)
MAX_TEXT_LENGTH = 498  # Setting lower than 512 to leave room for special tokens

# Percentile thresholds for analysis (more meaningful than absolute values)
PERCENTILE_THRESHOLDS = [0, 75, 85, 90, 95]  # Include 0 for overall; Analyze top 100%, 25%, 15%, 10%, and 5% most important tokens


# Function to truncate text at a sentence boundary if possible
def smart_truncate(text, max_length=MAX_TEXT_LENGTH):
    """Truncate text at a sentence boundary if possible, otherwise hard truncate."""
    if len(text) <= max_length:
        return text

    # Try to find the last sentence boundary before max_length
    sentence_ends = [".", "!", "?"]
    for i in range(
        max_length - 1, max_length // 2, -1
    ):  # Start from max_length and go backwards
        if text[i] in sentence_ends:
            return text[: i + 1]

    # If no sentence boundary found, simply truncate
    return text[:max_length]


# Load the dataset in chunks to reduce memory usage
def load_dataset_in_chunks(filepath, chunk_size=1000):
    """Load JSON dataset in chunks to reduce memory pressure."""
    try:
        # Open the JSON file and load it
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

        # Process in chunks
        chunk_dfs = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            # Apply text truncation to each chunk
            if "text" in chunk.columns:
                chunk["text"] = chunk["text"].apply(
                    lambda x: smart_truncate(x) if isinstance(x, str) else x
                )
            chunk_dfs.append(chunk)

        return chunk_dfs
    except FileNotFoundError:
        print(f"Error: Dataset JSON file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error loading dataset JSON: {e}")
        return []


# Load the dataset
dataset_chunks = load_dataset_in_chunks(
    "/home/michele.maggini/PORRO_2/datasets/HIPP_test.json"
)


# --- Enhanced helper functions for token attribution ---
def get_tokens_in_span(attributions, full_text, span_start, span_end):
    """
    Improved function to identify tokens within a span with more robust matching.
    Returns:
    - token_indices: indices of tokens within the span
    - non_span_indices: indices of tokens not in the span
    - tokens_info: detailed information about matched tokens
    """
    reconstructed_text = ""
    token_indices = []
    tokens_info = []  # Store detailed token information for debugging
    all_token_indices = set(
        range(1, len(attributions) - 1)
    )  # All tokens except [CLS] and [SEP]

    # Iterate through tokens skipping [CLS] and [SEP]
    for i, (token, attribution) in enumerate(attributions[1:-1], 1):
        clean_token = token.replace(" ", "").strip()
        if token.startswith("##"):
            clean_token = token[2:]  # remove ## prefix for matching
        else:
            clean_token = token  # keep original token

        if not clean_token:
            continue

        # Find the token in the remaining text
        search_start = len(reconstructed_text) - (
            len(clean_token) // 2 if len(reconstructed_text) > 0 else 0
        )
        if search_start < 0:
            search_start = 0

        token_position = -1
        try:
            # More robust search: iterate search start slightly if not found immediately
            found = False
            for offset in range(
                min(len(full_text) - search_start, 5)
            ):  # Check a small window
                current_search_start = search_start + offset
                if current_search_start < len(full_text) and current_search_start >= 0:
                    if full_text[current_search_start:].startswith(clean_token):
                        token_position = current_search_start
                        found = True
                        break
            if not found:
                # Fallback: use simpler find if the above fails
                token_position = full_text.find(clean_token, search_start)

        except Exception as find_error:
            token_position = -1  # Ensure it's -1 if an error occurs

        # Store token info for debugging
        token_info = {
            "token_idx": i,
            "token": token,
            "clean_token": clean_token,
            "position": token_position,
            "attribution": attribution,
            "in_span": False,
        }

        if token_position != -1:
            token_end = token_position + len(clean_token)
            # Check for overlap: token must be at least partially within the span
            if not (token_position >= span_end or token_end <= span_start):
                token_indices.append(i)
                token_info["in_span"] = True
            # Update reconstructed_text based on where the token was found + its length
            reconstructed_text = full_text[: max(len(reconstructed_text), token_end)]

        tokens_info.append(token_info)

    # Return both the token indices in the span and those not in the span
    non_span_indices = list(all_token_indices - set(token_indices))
    return token_indices, non_span_indices, tokens_info


def get_span_attribution_score(attributions, token_indices, method="median"):
    """
    Calculate span attribution using different aggregation methods.

    Parameters:
    - attributions: List of (token, attribution) tuples
    - token_indices: List of token indices to include
    - method: Aggregation method ('mean', 'median', 'sum', 'max')

    Returns:
    - Aggregated attribution score
    """
    if not token_indices:
        return 0.0  # Return float zero

    # Ensure indices are within the valid range of attributions
    valid_indices = [idx for idx in token_indices if 0 <= idx < len(attributions)]
    if not valid_indices:
        return 0.0

    attribution_values = [attributions[idx][1] for idx in valid_indices]

    if method == "mean":
        return (
            sum(attribution_values) / len(attribution_values)
            if attribution_values
            else 0.0
        )
    elif method == "median":
        return np.median(attribution_values) if attribution_values else 0.0
    elif method == "sum":
        return sum(attribution_values) if attribution_values else 0.0
    elif method == "max":
        return max(attribution_values) if attribution_values else 0.0
    else:
        # Default to mean if invalid method
        return (
            sum(attribution_values) / len(attribution_values)
            if attribution_values
            else 0.0
        )


def normalize_attributions(attributions):
    """
    Normalize attribution scores using robust min-max scaling.
    This preserves the relative ranking while making scores comparable.
    """
    values = [attr[1] for attr in attributions]

    # Add safety check for empty values
    if not values:
        return attributions

    # Use robust min-max scaling (ignore extreme outliers)
    q1 = np.percentile(values, 10)  # 10th percentile as min
    q3 = np.percentile(values, 90)  # 90th percentile as max

    # Avoid division by zero
    if abs(q3 - q1) < 1e-6:
        normalized = [(token, 0.5) for token, _ in attributions]  # All equal
    else:
        normalized = [
            (token, max(0, min(1, (score - q1) / (q3 - q1))))
            for token, score in attributions
        ]

    return normalized


def is_special_token(token):
    """Check if a token is a special token ([CLS], [SEP], [PAD], etc.)"""
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
    return token in special_tokens


def get_percentile_threshold(data, percentile):
    """
    Calculate percentile threshold for attribution ratios.

    Parameters:
    - data: Series or list of attribution ratios
    - percentile: Percentile value (0-100)

    Returns:
    - Threshold value at specified percentile
    """
    if len(data) == 0:
        return 0.0
    return np.percentile(data, percentile)


def get_high_attribution_samples_by_percentile(results_df, percentile=90, top_n=10):
    """
    Identify samples with attribution ratios above a certain percentile.

    Parameters:
    - results_df: DataFrame with attribution results
    - percentile: Percentile threshold (e.g., 90 for top 10%)
    - top_n: Number of top samples to return

    Returns:
    - DataFrame with top high-attribution samples
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    threshold = get_percentile_threshold(results_df["attribution_ratio"], percentile)
    high_attr = results_df[results_df["attribution_ratio"] >= threshold].copy()
    return high_attr.sort_values("attribution_ratio", ascending=False).head(top_n)


# Function to calculate Cohen's d
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0


# Function to calculate bootstrap confidence interval
def bootstrap_ci(data, n_bootstraps=1000, ci=95):
    if len(data) == 0:
        return 0.0, 0.0
    bootstraps = [np.mean(resample(data)) for _ in range(n_bootstraps)]
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return lower, upper


# --- Create expanded output directories ---
# Create base output directories for all analyses
percentile_dir = os.path.join(base_output_dir, "percentile_analysis")
token_dir = os.path.join(base_output_dir, "token_analysis")
diagnostic_dir = os.path.join(base_output_dir, "diagnostics")

# Create all directories
os.makedirs(percentile_dir, exist_ok=True)
os.makedirs(token_dir, exist_ok=True)
os.makedirs(diagnostic_dir, exist_ok=True)

# --- Model processing loop ---
# Storage for all models' results
all_models_results = {}
all_trait_stats = {}
non_trait_attributions = defaultdict(list)  # Store attributions for non-trait tokens

# Add storage for token-level data
token_level_data = defaultdict(list)

# Batch size for processing
BATCH_SIZE = 32  # Process samples in smaller batches

# Dynamically load fine-tuned models from the specified directory
model_names = [
    os.path.join(base_model_dir, d)
    for d in os.listdir(base_model_dir)
    if os.path.isdir(os.path.join(base_model_dir, d)) and d.startswith("arch-")
]

# Process each model only if model_names and dataset_chunks are not empty
if model_names and dataset_chunks:
    for model_path in model_names:
        model_name = os.path.basename(
            model_path
        )  # Use directory name as model identifier
        print(f"\nProcessing model: {model_name}")

        try:
            # Load model and tokenizer from the fine-tuned model directory
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Move model to the determined device
            model = model.to(device)
            print(f"  Model '{model_name}' loaded to {device}")

            # Make sure tokenizer knows the max token limit
            tokenizer.model_max_length = 512

            # Prepare results storage
            results = defaultdict(list)
            trait_attributions = defaultdict(list)

            # Add storage for all trait and non-trait attribution values for effect size
            all_trait_attrs = []
            all_non_trait_attrs = []

            # Process each chunk
            total_chunks = len(dataset_chunks)
            print(f"  Processing {total_chunks} data chunks...")

            for chunk_idx, df_chunk in enumerate(dataset_chunks):
                print(f"  Processing chunk {chunk_idx + 1}/{total_chunks}")

                # Process samples in smaller batches within each chunk
                for batch_start in range(0, len(df_chunk), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(df_chunk))
                    batch = df_chunk.iloc[batch_start:batch_end]

                    # Process each sample in the batch
                    for idx, row in tqdm(
                        batch.iterrows(),
                        total=len(batch),
                        desc=f"Batch {batch_start // BATCH_SIZE + 1}",
                    ):
                        text = row["text"]
                        linguistic_traits = row["linguistic_traits"]

                        if not isinstance(text, str) or not text.strip():
                            continue

                        # Make sure text is not too long for the model
                        if len(text) > MAX_TEXT_LENGTH:
                            text = smart_truncate(text)

                        try:
                            # Construct traits_text by concatenating all span texts
                            traits_spans = []
                            for spans in linguistic_traits.values():
                                if isinstance(spans, list):
                                    traits_spans.extend([span.get('text', '') for span in spans if isinstance(span, dict)])
                                elif isinstance(spans, dict):
                                    traits_spans.append(spans.get('text', ''))
                            traits_text = " [SEP] ".join(traits_spans)  # Adjust separator if needed during training

                            # Tokenize combined
                            inputs = tokenizer(text, traits_text, return_tensors="pt", truncation=True, max_length=512)

                            # Move to device
                            inputs = {k: v.to(device) for k, v in inputs.items()}

                            # First, get prediction to know which class to attribute to (assuming binary classification)
                            with torch.no_grad():
                                logits = model(**inputs)
                            predicted_class = logits.argmax(-1).item()

                            # Get embedding layer
                            embedding_layer = model.model.get_input_embeddings()

                            # Find the position of the first [SEP]
                            sep_pos = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()

                            # Text embeds range: 0 to sep_pos + 1
                            text_embeds_range = slice(0, sep_pos + 1)

                            # Traits embeds range: sep_pos + 1 to end
                            traits_embeds_range = slice(sep_pos + 1, None)

                            # Compute full embeds
                            full_embeds = embedding_layer(inputs['input_ids'])

                            # Fixed traits embeds
                            with torch.no_grad():
                                fixed_traits_embeds = full_embeds[:, traits_embeds_range, :]

                            # Define custom forward
                            def custom_forward(text_embeddings_var):
                                batch_size = text_embeddings_var.shape[0]

                                # Reconstruct full embeds
                                full_embeds_var = torch.cat([text_embeddings_var, fixed_traits_embeds.expand(batch_size, -1, -1)], dim=1)

                                # Expand inputs
                                attention_mask_expanded = inputs['attention_mask'].expand(batch_size, -1)
                                token_type_ids_expanded = inputs['token_type_ids'].expand(batch_size, -1)
                                input_ids_expanded = inputs['input_ids'].expand(batch_size, -1)

                                outputs = model.model(
                                    inputs_embeds=full_embeds_var,
                                    attention_mask=attention_mask_expanded,
                                    token_type_ids=token_type_ids_expanded
                                )
                                hidden = outputs.last_hidden_state

                                # Average pool over text segment
                                text_token_mask = (
                                    (token_type_ids_expanded == 0)
                                    & (input_ids_expanded != tokenizer.cls_token_id)
                                    & (input_ids_expanded != tokenizer.sep_token_id)
                                    & (input_ids_expanded != tokenizer.pad_token_id)
                                ).float()
                                text_features = (hidden * text_token_mask.unsqueeze(2)).sum(
                                    dim=1
                                ) / text_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

                                # Average pool over traits segment
                                traits_token_mask = (
                                    (token_type_ids_expanded == 1)
                                    & (input_ids_expanded != tokenizer.sep_token_id)
                                    & (input_ids_expanded != tokenizer.pad_token_id)
                                ).float()
                                traits_features = (hidden * traits_token_mask.unsqueeze(2)).sum(
                                    dim=1
                                ) / traits_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

                                # Gating mechanism for fusion
                                combined = torch.cat((text_features, traits_features), dim=1)
                                gate = torch.sigmoid(model.fusion_linear(combined))
                                fused = text_features * gate + traits_features * (1 - gate)

                                logits = model.classifier(fused)
                                return logits[:, predicted_class]

                            # Text embeddings
                            text_embeddings = full_embeds[:, text_embeds_range, :]

                            # Integrated Gradients
                            ig = IntegratedGradients(custom_forward)

                            # Attribute
                            attributions, delta = ig.attribute(inputs=text_embeddings, target=None, n_steps=50, return_convergence_delta=True)

                            # Sum over embedding dim and normalize
                            attributions = attributions.sum(dim=-1).squeeze(0)
                            attributions = attributions / torch.norm(attributions)

                            # Get tokens for text part
                            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, text_embeds_range])

                            # Format as list of (token, attr)
                            attributions_list = [(token, float(attr)) for token, attr in zip(tokens, attributions)]

                            # Apply robust normalization to handle outliers
                            normalized_attributions = normalize_attributions(
                                attributions_list
                            )

                            valid_attributions = [
                                (token, score)
                                for token, score in normalized_attributions[1:-1]
                                if isinstance(score, (int, float))
                            ]
                            all_attribution_scores = [
                                score for _, score in valid_attributions
                            ]

                            # Calculate attribution statistics with median (more robust)
                            avg_text_attribution = (
                                np.median(all_attribution_scores)
                                if all_attribution_scores
                                else 0.0
                            )
                            # Avoid division by zero or near-zero
                            if abs(avg_text_attribution) < 1e-9:
                                avg_text_attribution = 1e-9

                            # Track which tokens are part of any linguistic trait
                            all_trait_tokens = set()

                            # Create a map to track which trait(s) each token belongs to
                            token_trait_map = [
                                [] for _ in range(len(normalized_attributions))
                            ]

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
                                            span_text = span.get("text", "")
                                            span_start = span.get("start")
                                            span_end = span.get("end")

                                            # Basic validation for span data
                                            if (
                                                not span_text
                                                or span_start is None
                                                or span_end is None
                                                or not isinstance(span_start, int)
                                                or not isinstance(span_end, int)
                                                or span_start >= span_end
                                                or span_start < 0
                                                or span_end > len(text)
                                            ):
                                                continue

                                            (
                                                token_indices,
                                                non_span_indices,
                                                tokens_info,
                                            ) = get_tokens_in_span(
                                                normalized_attributions,
                                                text,
                                                span_start,
                                                span_end,
                                            )

                                            # Calculate span attribution using multiple methods
                                            span_attribution_mean = (
                                                get_span_attribution_score(
                                                    normalized_attributions,
                                                    token_indices,
                                                    "mean",
                                                )
                                            )
                                            span_attribution_median = (
                                                get_span_attribution_score(
                                                    normalized_attributions,
                                                    token_indices,
                                                    "median",
                                                )
                                            )
                                            span_attribution_sum = (
                                                get_span_attribution_score(
                                                    normalized_attributions,
                                                    token_indices,
                                                    "sum",
                                                )
                                            )

                                            # Use median for primary analysis (more robust to outliers)
                                            span_attribution = span_attribution_median

                                            # Add to the set of tokens that are part of linguistic traits
                                            all_trait_tokens.update(token_indices)

                                            # Update token_trait_map
                                            for token_idx in token_indices:
                                                if token_idx < len(token_trait_map):
                                                    token_trait_map[token_idx].append(
                                                        trait_name
                                                    )

                                            # Calculate word count in span for analysis
                                            word_count = len(span_text.split())

                                            results["sample_id"].append(idx)
                                            results["trait"].append(trait_name)
                                            results["span_text"].append(span_text)
                                            results["span_attribution"].append(
                                                span_attribution
                                            )
                                            results["span_attribution_mean"].append(
                                                span_attribution_mean
                                            )
                                            results["span_attribution_median"].append(
                                                span_attribution_median
                                            )
                                            results["span_attribution_sum"].append(
                                                span_attribution_sum
                                            )
                                            results["avg_text_attribution"].append(
                                                avg_text_attribution
                                            )
                                            results["attribution_ratio"].append(
                                                span_attribution / avg_text_attribution
                                            )
                                            results["span_length"].append(
                                                len(span_text)
                                            )
                                            results["word_count"].append(word_count)
                                            results["span_position"].append(
                                                span_start / len(text)
                                            )  # Normalized position
                                            results["token_count"].append(
                                                len(token_indices)
                                            )

                                            # Add debugging info for this span - limit JSON size
                                            simplified_tokens_info = [
                                                {
                                                    k: v
                                                    for k, v in t.items()
                                                    if k != "attribution"
                                                }
                                                for t in tokens_info[
                                                    :10
                                                ]  # Only store first 10 tokens
                                            ]
                                            results["token_info"].append(
                                                json.dumps(simplified_tokens_info)
                                            )

                                            # Store only valid attributions for stats
                                            if isinstance(
                                                span_attribution, (int, float)
                                            ):
                                                trait_attributions[trait_name].append(
                                                    span_attribution
                                                )
                                                all_trait_attrs.append(span_attribution)

                            # Store detailed token-level data for this sample - limit size by sampling
                            # to reduce memory usage (take every Nth token)
                            sampling_rate = max(
                                1, len(normalized_attributions) // 50
                            )  # Sample at most 50 tokens per text

                            for i, (token, attr) in enumerate(normalized_attributions):
                                if is_special_token(token):
                                    continue
                                # Only store every Nth token to save memory
                                if i % sampling_rate != 0:
                                    continue

                                token_traits = (
                                    token_trait_map[i]
                                    if i < len(token_trait_map)
                                    else []
                                )
                                is_trait_token = len(token_traits) > 0

                                token_level_data["sample_id"].append(idx)
                                token_level_data["model"].append(model_name)
                                token_level_data["token_idx"].append(i)
                                token_level_data["token"].append(token)
                                token_level_data["attribution"].append(attr)
                                token_level_data["is_trait_token"].append(
                                    is_trait_token
                                )
                                token_level_data["traits"].append(
                                    ",".join(token_traits) if token_traits else "none"
                                )
                                token_level_data["token_position"].append(
                                    i / len(normalized_attributions)
                                )  # Normalized position

                            # Calculate average attribution for tokens not part of any linguistic trait
                            # Separate special tokens from other non-trait tokens
                            normal_non_trait_tokens = set()

                            for i, (token, attr) in enumerate(normalized_attributions):
                                if i not in all_trait_tokens:
                                    if not is_special_token(token):
                                        normal_non_trait_tokens.add(i)
                                        all_non_trait_attrs.append(attr)

                            # Calculate attribution for normal non-trait tokens
                            if normal_non_trait_tokens:
                                non_trait_attr = get_span_attribution_score(
                                    normalized_attributions,
                                    list(normal_non_trait_tokens),
                                    "median",
                                )
                                non_trait_ratio = (
                                    non_trait_attr / avg_text_attribution
                                    if avg_text_attribution
                                    else 0
                                )
                                non_trait_attributions[model_name].append(
                                    non_trait_ratio
                                )

                        except Exception as explainer_error:
                            print(
                                f"  Error during explanation for sample {idx}, text: '{text[:50]}...'. Error: {explainer_error}"
                            )

                    # Save intermediate results every BATCH_SIZE samples
                    if (
                        len(results.get("sample_id", [])) % (BATCH_SIZE * 5) == 0
                        and results
                    ):
                        # Create intermediate results DataFrame
                        interim_results_df = pd.DataFrame(results)
                        interim_results_df["model"] = model_name

                        # Save interim results
                        interim_output_csv = os.path.join(
                            base_output_dir,
                            f"interim_results_{model_name.replace('/', '_')}_chunk{chunk_idx}_batch{batch_start // BATCH_SIZE}.csv",
                        )
                        interim_results_df.to_csv(interim_output_csv, index=False)
                        print(f"  Saved interim results to '{interim_output_csv}'")

                    # Force garbage collection after each batch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save token-level data for this model - limit size by sampling
            if token_level_data:
                # If we have too many tokens, sample a subset
                max_tokens = 100000  # Maximum number of token entries to save
                token_df = pd.DataFrame(token_level_data)

                if len(token_df) > max_tokens:
                    # Sample tokens to reduce size
                    token_df = token_df.sample(max_tokens, random_state=42)

                token_csv_path = os.path.join(
                    token_dir, f"token_level_data_{model_name.replace('/', '_')}.csv"
                )
                token_df.to_csv(token_csv_path, index=False)
                print(f"  Token-level data saved to '{token_csv_path}'")

                # Clear token data to free memory
                token_level_data = defaultdict(list)

            # Clean up GPU memory after processing all samples for this model
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Model '{model_name}' moved to CPU and memory cleared.")

            # Create and save results DataFrame if results were generated
            if results:
                results_df = pd.DataFrame(results)
                results_df["model"] = model_name  # Add model name to results

                # Calculate statistics for each trait
                trait_stats = {}
                for trait, values in trait_attributions.items():
                    if values:
                        mean_attr = np.mean(values)
                        ci_lower, ci_upper = bootstrap_ci(values)
                        trait_stats[trait] = {
                            "count": len(values),
                            "mean_attribution": float(mean_attr),
                            "median_attribution": float(np.median(values)),
                            "std_attribution": float(np.std(values)),
                            "ci_lower": float(ci_lower),
                            "ci_upper": float(ci_upper),
                        }

                # Calculate Cohen's d for trait vs non-trait
                if all_trait_attrs and all_non_trait_attrs:
                    cohen_d = cohens_d(all_trait_attrs, all_non_trait_attrs)
                    trait_stats["cohens_d_vs_non_trait"] = float(cohen_d)

                # Save model-specific results
                output_csv = os.path.join(
                    base_output_dir,
                    f"span_attribution_analysis_{model_name.replace('/', '_')}.csv",
                )
                results_df.to_csv(output_csv, index=False)
                print(f"  Results saved to '{output_csv}'")

                # Generate and save diagnostic information using percentiles
                high_attr_samples = get_high_attribution_samples_by_percentile(
                    results_df, percentile=90
                )
                high_attr_csv = os.path.join(
                    diagnostic_dir,
                    f"high_attribution_samples_{model_name.replace('/', '_')}.csv",
                )
                high_attr_samples.to_csv(high_attr_csv, index=False)
                print(
                    f"  High attribution samples (top 10%) saved to '{high_attr_csv}'"
                )

                # Store results for comparison
                all_models_results[model_name] = results_df
                all_trait_stats[model_name] = trait_stats

                # Clear individual results to free memory
                results = defaultdict(list)
                trait_attributions = defaultdict(list)
                gc.collect()

        except Exception as e:
            print(f"  Error processing model {model_name}: {str(e)}")
            # Attempt to clean up GPU memory even if an error occurred mid-process
            if "model" in locals() and hasattr(model, "cpu"):
                model.cpu()
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
else:
    print(
        "Skipping model processing because no models were found or the dataset failed to load."
    )

# --- Combine results only if there are results ---
if all_models_results:
    # Combine results more memory-efficiently
    print("\nCombining results from all models...")
    combined_results = pd.concat(
        [
            df.sample(min(10000, len(df)), random_state=42)
            for df in all_models_results.values()
        ],
        ignore_index=True,
    )
    combined_csv_path = os.path.join(
        base_output_dir, "combined_attribution_results.csv"
    )
    combined_results.to_csv(combined_csv_path, index=False)
    print(f"\nCombined results saved to '{combined_csv_path}'")

    # Get unique list of all traits encountered across all models
    all_traits = sorted(combined_results["trait"].unique())
    model_names_processed = list(all_models_results.keys())

    print("\nAnalyzing linguistic traits across different percentile thresholds...")

    # --- Enhanced Percentile-based Analysis Section ---
    percentile_results = {}

    for percentile in PERCENTILE_THRESHOLDS:
        print(f"\nAnalyzing top {100 - percentile}% most important tokens...")
        percentile_data = {}

        # For each model, calculate metrics at this percentile
        for model_name in model_names_processed:
            model_data = combined_results[combined_results["model"] == model_name]

            # Calculate high attribution spans per trait
            trait_counts = {}
            for trait in all_traits:
                trait_data = model_data[model_data["trait"] == trait]

                if len(trait_data) > 0:
                    # Calculate trait-specific percentile threshold
                    threshold = get_percentile_threshold(
                        trait_data["attribution_ratio"], percentile
                    )
                    high_attr_spans = trait_data[
                        trait_data["attribution_ratio"] >= threshold
                    ]

                    total_spans = len(trait_data)
                    high_spans = len(high_attr_spans)
                    proportion = high_spans / total_spans if total_spans > 0 else 0
                    mean_high_ratio = float(high_attr_spans["attribution_ratio"].mean()) if len(high_attr_spans) > 0 else 0.0

                    trait_counts[trait] = {
                        "total_spans": total_spans,
                        "high_attribution_spans": high_spans,
                        "proportion": proportion,
                        "percentile_threshold": float(threshold),
                        "mean_ratio": float(trait_data["attribution_ratio"].mean()),
                        "median_ratio": float(trait_data["attribution_ratio"].median()),
                        "mean_high_ratio": mean_high_ratio,
                    }
                else:
                    trait_counts[trait] = {
                        "total_spans": 0,
                        "high_attribution_spans": 0,
                        "proportion": 0,
                        "percentile_threshold": 0,
                        "mean_ratio": 0,
                        "median_ratio": 0,
                        "mean_high_ratio": 0,
                    }

            # For non-trait, use the same percentile-based proportion
            non_trait_ratios = non_trait_attributions[model_name]
            if non_trait_ratios:
                threshold_non_trait = get_percentile_threshold(
                    non_trait_ratios, percentile
                )
                high_non_trait = len(
                    [r for r in non_trait_ratios if r >= threshold_non_trait]
                )
                total_non_trait = len(non_trait_ratios)
                proportion_non_trait = (
                    high_non_trait / total_non_trait if total_non_trait > 0 else 0
                )
                mean_high_non_trait = float(np.mean([r for r in non_trait_ratios if r >= threshold_non_trait])) if high_non_trait > 0 else 0.0

                trait_counts["non_trait"] = {
                    "total_spans": total_non_trait,
                    "high_attribution_spans": high_non_trait,
                    "proportion": proportion_non_trait,
                    "percentile_threshold": float(threshold_non_trait),
                    "mean_ratio": float(np.mean(non_trait_ratios)),
                    "median_ratio": float(np.median(non_trait_ratios)),
                    "mean_high_ratio": mean_high_non_trait,
                }
            else:
                trait_counts["non_trait"] = {
                    "total_spans": 0,
                    "high_attribution_spans": 0,
                    "proportion": 0,
                    "percentile_threshold": 0,
                    "mean_ratio": 0,
                    "median_ratio": 0,
                    "mean_high_ratio": 0,
                }

            percentile_data[model_name] = trait_counts

        percentile_results[percentile] = percentile_data

    # Save percentile_results and non_trait_attributions as JSON
    percentile_json_path = os.path.join(base_output_dir, "percentile_results.json")
    with open(percentile_json_path, "w") as f:
        json.dump(percentile_results, f, default=float, indent=2)
    print(f"Percentile results saved to '{percentile_json_path}'")

    non_trait_json_path = os.path.join(base_output_dir, "non_trait_attributions.json")
    with open(non_trait_json_path, "w") as f:
        json.dump(dict(non_trait_attributions), f, default=float, indent=2)
    print(f"Non-trait attributions saved to '{non_trait_json_path}'")

    # Save trait stats including effect sizes and CIs
    trait_stats_json_path = os.path.join(base_output_dir, "trait_stats.json")
    with open(trait_stats_json_path, "w") as f:
        json.dump(all_trait_stats, f, default=float, indent=2)
    print(f"Trait stats saved to '{trait_stats_json_path}'")

    # --- Generate summary reports for each percentile ---
    print("\nGenerating summary reports...")
    for percentile in PERCENTILE_THRESHOLDS:
        summary_data = []

        for model_name in model_names_processed:
            model_percentile_data = percentile_results[percentile][model_name]

            for trait, stats in model_percentile_data.items():
                summary_data.append(
                    {
                        "model": model_name,
                        "trait": trait,
                        "percentile": percentile,
                        "total_spans": stats["total_spans"],
                        "high_attribution_spans": stats["high_attribution_spans"],
                        "proportion": stats["proportion"],
                        "threshold_value": stats["percentile_threshold"],
                        "mean_ratio": stats["mean_ratio"],
                        "median_ratio": stats["median_ratio"],
                        "mean_high_ratio": stats["mean_high_ratio"],
                    }
                )

        # Create summary DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(
            percentile_dir, f"percentile_{percentile}_summary.csv"
        )
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  Percentile {percentile} summary saved to '{summary_csv_path}'")

    print(f"\nAnalysis complete! All results saved to: {base_output_dir}")

else:
    print("\nNo combined results were generated. Skipping analyses.")