import json
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import pickle
import sys
import pandas as pd

def get_linguistic_trait_spans(sample):
    """Extract spans for each linguistic trait separately with better error handling"""
    trait_spans = defaultdict(set)

    if "linguistic_traits" not in sample:
        print(f"Warning: 'linguistic_traits' not found in sample: {sample.get('article_id', 'unknown')}")
        return trait_spans

    linguistic_traits = sample["linguistic_traits"]

    if isinstance(linguistic_traits, dict):
        for trait_name, trait_spans_list in linguistic_traits.items():
            if trait_spans_list and isinstance(trait_spans_list, list):
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.cuda()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return []

    def predict_proba(texts):
        if not isinstance(texts, list):
            texts = [str(text) for text in texts]
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            return torch.softmax(outputs.logits, dim=1).cpu().numpy()
        except Exception as e:
            print(f"Error in predict_proba batch: {e}")
            return np.zeros((len(texts), 2))

    try:
        explainer = shap.Explainer(predict_proba, tokenizer)
    except Exception as e:
        print(f"Error initializing SHAP explainer: {e}")
        return []

    # Compute probabilities per sample to isolate errors
    probs = []
    for j, text in enumerate(chunk_texts):
        try:
            prob = predict_proba([text])[0]
            probs.append(prob)
        except Exception as e:
            print(f"Error in predict_proba for sample {chunk_start + j}: {e}")
            probs.append(np.zeros(2))

    chunk_results = []
    shap_values_batch = None
    try:
        shap_values_batch = explainer(chunk_texts)
        if shap_values_batch is None or len(shap_values_batch.values) == 0:
            print(f"Warning: Empty SHAP values for chunk")
    except Exception as e:
        print(f"Error calculating SHAP values for chunk: {e}")

    if shap_values_batch is not None:
        for j, (text, sample, shap_values) in enumerate(zip(chunk_texts, chunk_samples, shap_values_batch)):
            i = chunk_start + j
            trait_spans = get_linguistic_trait_spans(sample)

            # Add truncation to encode_plus to avoid warnings for long texts
            encoded = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=512)
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

            try:
                shap_vals_array = shap_values.values
                print(f"SHAP values shape for sample {i}: {shap_vals_array.shape}")  # Debug shape
                if len(shap_vals_array.shape) == 3:  # [num_samples, num_tokens, num_classes]
                    shap_vals = shap_vals_array[0, :, 1]  # Take class 1 for binary classification
                elif len(shap_vals_array.shape) == 2:  # [num_tokens, num_classes]
                    shap_vals = shap_vals_array[:, 1]  # FIXED: All tokens for class 1
                else:
                    print(f"Unexpected SHAP values shape for sample {i}: {shap_vals_array.shape}")
                    continue

                if len(shap_vals) > len(tokens):
                    shap_vals = shap_vals[:len(tokens)]
                elif len(shap_vals) < len(tokens):
                    shap_vals = np.pad(shap_vals, (0, len(tokens) - len(shap_vals)), "constant")
            except Exception as e:
                print(f"Error processing SHAP values for sample {i}: {e}")
                continue

            min_len = min(len(tokens), len(token_trait_mapping), len(shap_vals))
            tokens = tokens[:min_len]
            token_trait_mapping = token_trait_mapping[:min_len]
            shap_vals = shap_vals[:min_len]
            offsets = offsets[:min_len] if len(offsets) > min_len else offsets

            # Prepare label_pred
            label_pred = [
                {'label': 'LABEL_0', 'score': float(probs[j][0])},
                {'label': 'LABEL_1', 'score': float(probs[j][1])}
            ]

            sample_result = {
                "sample_id": i,
                "article_id": sample.get("article_id", i),
                "paragraph_id": sample.get("paragraph_id", 0),
                "text": text,
                "labels": sample.get("labels", None),
                "source": sample.get("source", ""),
                "refined_technique": sample.get("refined_technique", ""),
                "topic": sample.get("topic", None),
                "tokens": tokens,
                "shap_values": shap_vals,
                "token_trait_mapping": token_trait_mapping,
                "token_positions": offsets,
                "available_traits": list(trait_spans.keys()),
                "label_pred": label_pred,
                "base_value": float(shap_values.base_values[1]),  # Store base value for class 1
                "model_name": os.path.basename(os.path.normpath(model_path))  # Handle trailing slash
            }
            chunk_results.append(sample_result)

    return chunk_results

def worker(args):
    import torch
    model_path, chunks, process_id, samples = args  # Pass samples explicitly
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

    texts = [sample["text"] for sample in samples if "text" in sample]
    valid_samples = [sample for sample in samples if "text" in sample]

    if len(texts) == 0:
        return None

    chunk_size = 8  # Reduced to minimize OOM risks
    chunks = []
    for chunk_start in range(0, len(texts), chunk_size):
        chunk_texts = texts[chunk_start:chunk_start + chunk_size]
        chunk_samples = valid_samples[chunk_start:chunk_start + chunk_size]
        chunks.append((chunk_texts, chunk_samples, chunk_start))

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_processes = max(1, num_gpus)
    if num_gpus == 0:
        num_processes = os.cpu_count() // 2 or 4
    else:
        num_processes = 2  # Set to 2 for 2 GPUs
        print(f"Using {num_processes} processes on {num_gpus} GPUs")

    chunks_per_process = len(chunks) // num_processes
    split_chunks = []
    start = 0
    for i in range(num_processes):
        end = start + chunks_per_process + (1 if i < len(chunks) % num_processes else 0)
        split_chunks.append(chunks[start:end])
        start = end

    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=num_processes)
    process_args = [(model_path, split_chunks[i], i, samples) for i in range(num_processes)]  # Pass samples
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
        print("Usage: python analyze_model.py <model_folder>")
        sys.exit(1)

    model_folder = sys.argv[1]
    model_name = os.path.basename(os.path.normpath(model_folder))
    OUTPUT_DIR = "/home/michele.maggini/XAI_HIPP/SHAP/16_ottobre_FT_results"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load dataset only in main process
    data = []
    try:
        with open("/home/michele.maggini/PORRO_2/datasets/HIPP_test.json", "r") as f:
            data = json.load(f)
        print(f"Data loaded successfully! Total samples: {len(data)}")
    except json.JSONDecodeError as e:
        print(f"Error loading JSON data: {e}")
        sys.exit(1)

    results = analyze_model(model_folder, data)
    if results:
        pickle_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results for {model_name} saved to {pickle_path}")
        
        # Save to CSV (flatten lists to strings for CSV compatibility)
        df = pd.DataFrame(results)
        # Convert lists/arrays to strings for CSV
        for col in ['tokens', 'shap_values', 'token_trait_mapping', 'token_positions', 'available_traits', 'label_pred']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
        csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results for {model_name} saved to {csv_path}")
