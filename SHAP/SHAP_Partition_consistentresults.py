import json
import numpy as np
import shap
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import pickle
import sys
import pandas as pd
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoModel
import torch.nn as nn
from torch.amp import autocast

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set environment variables
os.environ['HF_HUB_DISABLE_INTERACTIVE_PROMPT'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom classes
class HyperpartisanConfig(PretrainedConfig):
    model_type = "ft_emb_traits"

    def __init__(self, base_model_name="nickprock/sentence-bert-base-italian-uncased", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name

class HyperpartisanModel(PreTrainedModel):
    config_class = HyperpartisanConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.base_model_name, trust_remote_code=True)
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

    def forward(self, input_ids, attention_mask, token_type_ids=None):
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

def get_linguistic_trait_spans(sample):
    """Extract spans for each linguistic trait separately with better error handling"""
    trait_spans = defaultdict(set)

    if "linguistic_traits" not in sample:
        print(f"Warning: 'linguistic_traits' not found in sample: {sample.get('article_id', 'unknown')}")
        return trait_spans

    linguistic_traits = sample["linguistic_traits"]

    if isinstance(linguistic_traits, dict):
        for trait_name, trait_spans_list in linguistic_traits.items():
            if trait_spans_list and isinstance(trait_spans_list, (list, dict)):
                if isinstance(trait_spans_list, dict):
                    trait_spans_list = [trait_spans_list]
                for span_info in trait_spans_list:
                    if isinstance(span_info, dict) and "start" in span_info and "end" in span_info:
                        start = span_info["start"]
                        end = span_info["end"]
                        trait_spans[trait_name].update(range(start, end))
                    else:
                        print(f"Warning: Invalid span format in trait {trait_name}: {span_info}")
    else:
        print(f"Warning: linguistic_traits is not a dictionary: {type(linguistic_traits)}")

    return trait_spans

def analyze_chunk(model_path, chunk_texts, chunk_samples, chunk_start):
    try:
        # CRITICAL FIX: Load config first to get the correct base_model_name
        config = HyperpartisanConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loaded config with base_model_name: {config.base_model_name}")
        
        # Load tokenizer from the BASE MODEL specified in config, not from model_path
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
        
        # Load the model with explicit config
        model = HyperpartisanModel.from_pretrained(
            model_path, 
            config=config,
            use_safetensors=True,
            trust_remote_code=True
        )
        model.eval()
        model.cuda()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Store reference to samples
    all_samples = chunk_samples.copy()

    def predict_proba(texts):
        """Predict probabilities for text inputs - returns shape (n_samples, 2)"""
        if isinstance(texts, str):
            texts = [texts]
        try:
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []

            samples_to_use = all_samples[:len(texts)] if len(all_samples) >= len(texts) else all_samples * (len(texts) // len(all_samples) + 1)
            samples_to_use = samples_to_use[:len(texts)]

            max_length = 256
            for text, sample in zip(texts, samples_to_use):
                traits_spans = []
                linguistic_traits = sample.get("linguistic_traits", {})
                for spans in linguistic_traits.values():
                    if isinstance(spans, list):
                        traits_spans.extend([span.get('text', '') for span in spans if isinstance(span, dict)])
                    elif isinstance(spans, dict):
                        traits_spans.append(spans.get('text', ''))
                traits_text = " [SEP] ".join(traits_spans) if traits_spans else "[PAD]"

                inputs = tokenizer(
                    text,
                    traits_text,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    return_token_type_ids=True
                )
                input_ids_list.append(inputs["input_ids"].squeeze(0))
                attention_mask_list.append(inputs["attention_mask"].squeeze(0))
                token_type_ids_list.append(inputs["token_type_ids"].squeeze(0))

            input_ids = torch.stack(input_ids_list).to('cuda')
            attention_mask = torch.stack(attention_mask_list).to('cuda')
            token_type_ids = torch.stack(token_type_ids_list).to('cuda')

            with torch.no_grad(), autocast(device_type='cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs
        except Exception as e:
            print(f"Error in predict_proba batch: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(texts), 2))

    # Prepare background data as text for Text masker
    background_size = min(3, len(chunk_texts))  # Increased to 3 for more variability
    background_texts = chunk_texts[:background_size] if chunk_texts else [""]
    background_samples = chunk_samples[:background_size] if chunk_samples else [{}]
    background_data = []

    max_length = 256
    for text, sample in zip(background_texts, background_samples):
        traits_spans = []
        linguistic_traits = sample.get("linguistic_traits", {})
        for spans in linguistic_traits.values():
            if isinstance(spans, list):
                traits_spans.extend([span.get('text', '') for span in spans if isinstance(spans, dict)])
            elif isinstance(spans, dict):
                traits_spans.append(spans.get('text', ''))
        traits_text = " [SEP] ".join(traits_spans) if traits_spans else "[PAD]"
        combined_text = text + " [SEP] " + traits_text
        if not text.strip():
            print(f"Warning: Empty text in background sample: {sample.get('article_id', 'unknown')}")
            combined_text = "[PAD]"
        background_data.append(combined_text)

    # Test background data with predict_proba
    test_probs = predict_proba(background_data)
    print(f"Background predict_proba output: {test_probs}")

    # Initialize PartitionExplainer with Text masker
    try:
        masker = shap.maskers.Text(tokenizer=tokenizer, mask_token="[PAD]")
        explainer = shap.PartitionExplainer(
            model=predict_proba,
            masker=masker,
            link=shap.links.identity
        )
        print("SHAP PartitionExplainer initialized successfully")
    except Exception as e:
        print(f"Error initializing SHAP PartitionExplainer: {e}")
        import traceback
        traceback.print_exc()
        return []

    chunk_results = []
    for j, (text, sample) in enumerate(zip(chunk_texts, chunk_samples)):
        try:
            print(f"Processing sample {j+1}/{len(chunk_texts)}")
            i = chunk_start + j
            trait_spans = get_linguistic_trait_spans(sample)

            # Prepare input text for SHAP
            traits_spans = []
            linguistic_traits = sample.get("linguistic_traits", {})
            for spans in linguistic_traits.values():
                if isinstance(spans, list):
                    traits_spans.extend([span.get('text', '') for span in spans if isinstance(spans, dict)])
                elif isinstance(spans, dict):
                    traits_spans.append(spans.get('text', ''))
            traits_text = " [SEP] ".join(traits_spans) if traits_spans else "[PAD]"
            input_text = [text + " [SEP] " + traits_text]
            if not text.strip():
                print(f"Warning: Empty text in sample {j+1}: {sample.get('article_id', 'unknown')}")
                input_text = ["[PAD]"]

            # Compute SHAP values
            shap_values = explainer(input_text)
            if isinstance(shap_values, shap._explanation.Explanation):
                shap_values_array = shap_values.values
            else:
                shap_values_array = shap_values
            print(f"SHAP values shape for sample {j+1}: {np.array(shap_values_array).shape if isinstance(shap_values_array, list) else shap_values_array.shape}")

            # Extract tokens for result
            encoded = tokenizer.encode_plus(
                text,
                traits_text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=True
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

            if tokens and tokens[0] == tokenizer.cls_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == tokenizer.sep_token:
                tokens = tokens[:-1]

            offsets = encoded["offset_mapping"]
            if len(offsets) > 0 and offsets[0] == (0, 0):
                offsets = offsets[1:]
            if len(offsets) > 0 and offsets[-1] == (0, 0):
                offsets = offsets[:-1]

            token_trait_mapping = []
            for start, end in offsets:
                token_traits = []
                if end <= start:
                    token_traits = ["non-trait"]
                else:
                    for trait_name, trait_positions in trait_spans.items():
                        if any(pos in trait_positions for pos in range(start, end)):
                            token_traits.append(trait_name)
                    if not token_traits:
                        token_traits = ["non-trait"]
                token_trait_mapping.append(token_traits)

            # Process SHAP values
            if isinstance(shap_values_array, list) and len(shap_values_array) == 2:
                shap_vals = shap_values_array[1][0]  # Class 1, first sample
            elif isinstance(shap_values_array, np.ndarray):
                if len(shap_values_array.shape) == 3:  # [n_samples, n_features, n_classes]
                    shap_vals = shap_values_array[0, :, 1]  # First sample, class 1
                elif len(shap_values_array.shape) == 2:  # [n_samples, n_features]
                    shap_vals = shap_values_array[0, :]
                else:
                    shap_vals = shap_values_array
            else:
                print(f"Warning: Unexpected SHAP values type/shape: {type(shap_values_array)}, {np.array(shap_values_array).shape if hasattr(shap_values_array, '__len__') else 'N/A'}")
                shap_vals = np.zeros(len(tokens))

            # Debug SHAP values
            print(f"SHAP values sample for sample {j+1}: {shap_vals[:5]}")

            # Adjust SHAP values to match token length
            if len(shap_vals) > len(tokens):
                shap_vals = shap_vals[:len(tokens)]
            elif len(shap_vals) < len(tokens):
                shap_vals = np.pad(shap_vals, (0, len(tokens) - len(shap_vals)), "constant")

            min_len = min(len(tokens), len(token_trait_mapping), len(shap_vals))
            
            # FIXED: Skip samples with no valid tokens after removing special tokens
            if min_len == 0:
                print(f"Warning: Sample {j+1} has no valid tokens after processing, skipping")
                continue
                
            tokens = tokens[:min_len]
            token_trait_mapping = token_trait_mapping[:min_len]
            shap_vals = shap_vals[:min_len]
            offsets = offsets[:min_len] if len(offsets) > min_len else offsets

            prob = predict_proba(input_text)[0]
            label_pred = [
                {'label': 'LABEL_0', 'score': float(prob[0])},
                {'label': 'LABEL_1', 'score': float(prob[1])}
            ]

            sample_result = {
                "sample_id": i,
                "original_index": sample.get("original_index", i),  # FIXED: Track original index
                "article_id": sample.get("article_id", i),
                "paragraph_id": sample.get("paragraph_id", 0),
                "text": text,
                "labels": sample.get("labels", None),
                "source": sample.get("source", ""),
                "refined_technique": sample.get("refined_technique", ""),
                "topic": sample.get("topic", None),
                "tokens": tokens,
                "shap_values": shap_vals.tolist(),
                "token_trait_mapping": token_trait_mapping,
                "token_positions": offsets,
                "available_traits": list(trait_spans.keys()),
                "label_pred": label_pred,
                "base_value": float(prob[1]),
                "model_name": os.path.basename(os.path.normpath(model_path))
            }
            chunk_results.append(sample_result)

        except Exception as e:
            print(f"Error processing sample {j}: {e}")
            import traceback
            traceback.print_exc()

    # Clean up
    model.cpu()
    torch.cuda.empty_cache()
    return chunk_results

def worker(args):
    import torch
    model_path, chunks, process_id, samples = args
    device_id = process_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    print(f"Process {process_id} starting on GPU {device_id}")

    process_results = []
    for chunk_idx, (chunk_texts, chunk_samples, chunk_start) in enumerate(chunks):
        print(f"Process {process_id} processing chunk {chunk_idx + 1}")
        chunk_results = analyze_chunk(model_path, chunk_texts, chunk_samples, chunk_start)
        process_results.extend(chunk_results)

    return process_results

def analyze_model(model_path, samples):
    model_name = os.path.basename(os.path.normpath(model_path))
    print(f"Analyzing model: {model_name}")

    # FIXED: Keep original indices to maintain consistency across models
    texts = []
    valid_samples = []
    original_indices = []
    
    for idx, sample in enumerate(samples):
        if "text" in sample and isinstance(sample["text"], str) and sample["text"].strip():
            texts.append(sample["text"])
            # Add original index to sample
            sample_copy = sample.copy()
            sample_copy["original_index"] = idx
            valid_samples.append(sample_copy)
            original_indices.append(idx)
    
    print(f"Processing {len(texts)} valid samples out of {len(samples)} total samples")

    if len(texts) == 0:
        print("Error: No valid texts found in samples")
        return None

    chunk_size = 4
    chunks = []
    for chunk_start in range(0, len(texts), chunk_size):
        chunk_texts = texts[chunk_start:chunk_start + chunk_size]
        chunk_samples = valid_samples[chunk_start:chunk_start + chunk_size]
        chunks.append((chunk_texts, chunk_samples, chunk_start))

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_processes = max(1, num_gpus) if num_gpus > 0 else mp.cpu_count()
    print(f"Using {num_processes} processes on {num_gpus} GPUs")

    chunks_per_process = max(1, len(chunks) // num_processes)
    split_chunks = []
    start = 0
    for i in range(num_processes):
        end = start + chunks_per_process + (1 if i < len(chunks) % num_processes else 0)
        split_chunks.append(chunks[start:end])
        start = end

    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=num_processes)
    process_args = [(model_path, split_chunks[i], i, samples) for i in range(num_processes)]
    all_process_results = pool.map(worker, process_args)
    pool.close()
    pool.join()

    results = []
    for proc_res in all_process_results:
        results.extend(proc_res)

    results.sort(key=lambda x: x["sample_id"])
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python SHAP_EMB-arch_partition_parallel.py <model_folder>")
        sys.exit(1)

    model_folder = sys.argv[1]
    model_name = os.path.basename(os.path.normpath(model_folder))
    OUTPUT_DIR = "/home/michele.maggini/XAI_HIPP/SHAP/results_FT_EMB_CONSISTENT"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    data = []
    try:
        with open("/home/michele.maggini/PORRO_2/datasets/HIPP_test.json", "r") as f:
            data = json.load(f)
        print(f"Data loaded successfully! Total samples: {len(data)}")
    except json.JSONDecodeError as e:
        print(f"Error loading JSON data: {e}")
        sys.exit(1)

    # Validate token lengths - ALSO FIXED: Load tokenizer from base model in config
    config = HyperpartisanConfig.from_pretrained(model_folder, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    token_lengths = [len(tokenizer.encode(s["text"])) for s in data if "text" in s and isinstance(s["text"], str)]
    if token_lengths:
        print(f"95th percentile token length: {np.percentile(token_lengths, 95)}")

    results = analyze_model(model_folder, data)
    if results:
        # FIXED: Report how many samples were successfully processed
        print(f"Successfully processed {len(results)} samples")
        print(f"Expected {len(data)} samples, processed {len(results)} samples")
        if len(results) != len(data):
            processed_indices = {r.get("original_index", r["sample_id"]) for r in results}
            missing_indices = [i for i in range(len(data)) if i not in processed_indices]
            print(f"WARNING: Missing {len(missing_indices)} samples. Indices: {missing_indices[:10]}...")
        
        pickle_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results for {model_name} saved to {pickle_path}")
        
        df = pd.DataFrame(results)
        for col in ['tokens', 'shap_values', 'token_trait_mapping', 'token_positions', 'available_traits', 'label_pred']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
        csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results for {model_name} saved to {csv_path}")