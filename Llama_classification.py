import json
import re
import os
import argparse
import pandas as pd
import numpy as np
import random
import time
import itertools
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig

import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Process the dataset
def process_dataset(file_path: str) -> List[Dict[str, Any]]:
    logger.info("Processing dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        processed_item = {
            'text': item['text'],
            'labels': item['labels'],  # Keep the original label for evaluation
            'linguistic_traits': {}
        }
        
        # Extract linguistic traits without spans and exclude Agents and Entities
        if 'linguistic_traits' in item:
            for trait, values in item['linguistic_traits'].items():
                # Skip Agents and Entities
                if trait in ["Agents", "Terms"]:
                    continue
                    
                # Handle different data structures
                if isinstance(values, list) and values and isinstance(values[0], dict) and 'text' in values[0]:
                    processed_item['linguistic_traits'][trait] = [v['text'] for v in values]
                elif isinstance(values, list) and (not values or isinstance(values[0], str)):
                    processed_item['linguistic_traits'][trait] = values
                elif isinstance(values, dict) and 'text' in values:
                    processed_item['linguistic_traits'][trait] = [values['text']]
                else:
                    processed_item['linguistic_traits'][trait] = []
            
        processed_data.append(processed_item)
    
    return processed_data

# Parse example string with linguistic traits
def parse_example_with_traits(example_str: str) -> Dict[str, Any]:
    """Parse an example string from dic_h_ling_traits with text, label, and linguistic traits."""
    # Split the example string by commas
    parts = example_str.split(',', 2)  # Split into 3 parts max
    
    if len(parts) < 2:
        logger.warning(f"Invalid example format: {example_str}")
        return {}
    
    text = parts[0].strip()
    label = parts[1].strip() if len(parts) > 1 else "0"
    
    # Initialize the example dictionary
    example = {
        "text": text,
        "label": label,
        "linguistic_traits": {}
    }
    
    # Parse linguistic traits if they exist
    if len(parts) > 2:
        traits_str = parts[2].strip()
        
        # Extract each trait and its values
        trait_pattern = r'([A-Za-z_/]+):\s*\[(.*?)\]'
        trait_matches = re.findall(trait_pattern, traits_str)
        
        for trait_name, trait_values in trait_matches:
            # Skip Agents and Terms
            if trait_name in ["Agents", "Terms"]:
                continue
                
            # Process trait values - extract text values
            values = []
            text_pattern = r'text:\s*([^,]+?)(?:,|\s*$)'
            text_matches = re.findall(text_pattern, trait_values)
            
            for text_value in text_matches:
                values.append(text_value.strip())
            
            example["linguistic_traits"][trait_name] = values
    
    return example

# Parse simple example string (text and label only)
def parse_simple_example(example_str: str) -> Dict[str, Any]:
    """Parse a simple example string from dic_h with text and label only."""
    # Split the example string by the last comma
    parts = example_str.rsplit(',', 1)
    
    if len(parts) < 2:
        logger.warning(f"Invalid example format: {example_str}")
        return {}
    
    text = parts[0].strip()
    label = parts[1].strip()
    
    return {
        "text": text,
        "label": label
    }

# Load few-shot examples from file
def load_examples(examples_path: str) -> Dict[str, Any]:
    logger.info("Loading few-shot examples...")
    with open(examples_path, 'r', encoding='utf-8') as f:
        raw_examples_data = json.load(f)
    
    # Create a processed structure
    examples_data = {
        "dic_h": {},
        "dic_h_ling_traits": {}
    }
    
    # Process the loaded data
    if "DPP" in raw_examples_data:
        # Process dic_h (simple examples)
        if "dic_h" in raw_examples_data["DPP"]:
            for key, example_str in raw_examples_data["DPP"]["dic_h"].items():
                examples_data["dic_h"][key] = parse_simple_example(example_str)
        
        # Process dic_h_ling_traits (examples with linguistic traits)
        if "dic_h_ling_traits" in raw_examples_data["DPP"]:
            for key, example_str in raw_examples_data["DPP"]["dic_h_ling_traits"].items():
                examples_data["dic_h_ling_traits"][key] = parse_example_with_traits(example_str)
    
    return examples_data

# Format examples for few-shot learning
def format_examples(examples_data: Dict[str, Any], shot_count: int, prompt_type: str) -> str:
    if shot_count == 0:
        return ""
    
    # Determine which dictionary to use based on prompt type
    if prompt_type == "prompt_no_LT":
        # For prompt_no_LT, use only text + classification (dic_h)
        example_dict = examples_data["dic_h"]
        example_keys = list(example_dict.keys())[:shot_count]
    else:  # prompt_type == "prompt_LT"
        # For prompt_LT, use text + classification + linguistic traits (dic_h_ling_traits)
        example_dict = examples_data["dic_h_ling_traits"]
        example_keys = list(example_dict.keys())[:shot_count]
    
    # Format examples
    examples_text = ""
    
    for key in example_keys:
        example = example_dict.get(key, {})
        
        if not example:
            continue
        
        # Extract data
        text = example.get("text", "")
        label = example.get("label", "0")  # Default to 0 if not present
        
        if prompt_type == "prompt_no_LT":
            # Simple example with just text and label
            examples_text += f"""
### Example:
Text: "{text}"

Classification: {label}

"""
        else:  # prompt_type == "prompt_LT"
            # Example with text, label, and linguistic traits
            linguistic_traits = example.get("linguistic_traits", {})
            
            # Format the linguistic traits in a human-readable way
            traits_lines = []
            for trait_name, trait_values in linguistic_traits.items():
                if trait_values:  # Only include non-empty traits
                    formatted_values = ", ".join([f'"{v}"' for v in trait_values])
                    traits_lines.append(f"{trait_name}: {formatted_values}")
            
            traits_description = "\n".join(traits_lines) if traits_lines else "No significant linguistic traits identified."
            
            examples_text += f"""
### Example:
Text: "{text}"
Linguistic traits:
{traits_description}

Classification: {label}

"""
    
    return examples_text

# Create prompts for classification
def create_classification_prompts(item: Dict[str, Any], examples_text: str, prompt_type: str) -> str:
    text = item['text']
    
    if prompt_type == "prompt_no_LT":
        # Prompt without linguistic traits information
        few_shot_instruction = "\n\nHere are some examples of how you should classify texts:" if examples_text else ""
        
        prompt = f"""<|system|>
You are an AI assistant specialized in political text analysis. Your task is to identify whether a text is hyperpartisan or not.

<|user|>
A text is considered hyperpartisan when it expresses an extreme opinion about an ideological, political, or social category, often polarizing the audience through emotionally charged, ironic/sarcastic, or excessively hyperbolic language.

### Instructions:
You will be provided with a text. You need to classify it as either hyperpartisan (1) or not hyperpartisan (0).{few_shot_instruction}

{examples_text}
### Text to classify:
"{text}"

### Required output:
Provide only the numerical label (1 for hyperpartisan, 0 for not hyperpartisan).

<|assistant|>
"""
    else:  # prompt_type == "prompt_LT"
        # Prompt with linguistic traits information
        linguistic_traits = item['linguistic_traits']
        
        # Format linguistic traits in a human-readable format
        traits_lines = []
        for trait_name, trait_values in linguistic_traits.items():
            if trait_values:  # Only include non-empty traits
                if isinstance(trait_values, list):
                    formatted_values = ", ".join([f'"{v}"' for v in trait_values])
                    traits_lines.append(f"{trait_name}: {formatted_values}")
                else:
                    traits_lines.append(f"{trait_name}: \"{trait_values}\"")
        
        traits_description = "\n".join(traits_lines) if traits_lines else "No significant linguistic traits identified."
        
        few_shot_instruction = "\n\nHere are some examples of how you should classify texts:" if examples_text else ""
        
        prompt = f"""<|system|>
You are an AI assistant specialized in political text analysis. Your task is to identify whether a text is hyperpartisan or not.

<|user|>
A text is considered hyperpartisan when it expresses an extreme opinion about an ideological, political, or social category, often polarizing the audience through emotionally charged, ironic/sarcastic, or excessively hyperbolic language.

### Instructions:
You will be provided with a text and its linguistic traits. You need to classify it as either hyperpartisan (1) or not hyperpartisan (0).{few_shot_instruction}

{examples_text}
### Text to classify:
"{text}"

Linguistic traits present:
{traits_description}

### Required output:
Provide only the numerical label (1 for hyperpartisan, 0 for not hyperpartisan).

<|assistant|>
"""

    return prompt

# Extract classification label from the model response
def extract_classification(response_text: str) -> int:
    # Clean and normalize the response
    cleaned_response = response_text.strip().lower()
    
    # Try to find just the number 0 or 1
    if cleaned_response == "0" or cleaned_response == "1":
        return int(cleaned_response)
    
    # Look for patterns like "classification: 1" or "label: 0"
    label_patterns = [
        r'classification:\s*(\d)',
        r'label:\s*(\d)',
        r'output:\s*(\d)',
        r'result:\s*(\d)',
        r'(\d)' # Last resort: just find any single digit
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, cleaned_response)
        if match:
            return int(match.group(1))
    
    # Default to 0 if nothing is found
    logger.warning(f"Could not extract classification from: {response_text}")
    return 0

# Initialize Llama model and tokenizer
def initialize_llama(model_path: str, device: str = "cuda"):
    """Initialize Llama model and tokenizer"""
    logger.info(f"Initializing hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 from {model_path} on {device}")
    quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512, # Note: Update this as per your use-case
    do_fuse=True,
)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    return model, tokenizer

# Generate classification using Llama model
def generate_classification(model, tokenizer, prompt: str, temperature: float, top_p: float, max_retries: int = 3) -> Tuple[int, str]:
    for attempt in range(max_retries):
        try:
            # Set up generation parameters
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
            )
            
            # Decode response
            response_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Extract the classification label
            classification = extract_classification(response_text)
            
            return classification, response_text
        
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retry attempts failed: {e}")
                return -1, f"Model inference failed: {str(e)}"

# Process a single item with a specific configuration
def process_item_with_config(args):
    item, model, tokenizer, prompt_type, temperature, top_p, shot_count, examples_data = args
    
    # Format examples based on configuration
    if shot_count > 0:
        examples_text = format_examples(examples_data, shot_count, prompt_type)
    else:
        examples_text = ""
    
    # Create the appropriate prompt
    prompt = create_classification_prompts(item, examples_text, prompt_type)
    
    # Generate the classification
    predicted_label, raw_response = generate_classification(
        model, tokenizer, prompt, temperature, top_p
    )
    
    # Create a unique configuration identifier
    config_id = f"{prompt_type}_{shot_count}shot"
    
    return {
        'config_id': config_id,
        'prompt_type': prompt_type,
        'shot_count': shot_count,
        'original_text': item['text'],
        'true_label': item['labels'],
        'predicted_label': predicted_label,
        'raw_response': raw_response
    }

# Create experiment directories for all configurations
def create_experiment_dirs(output_path: str) -> str:
    # Create main output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_root = os.path.join(output_path, f"classification_experiment_{timestamp}")
    os.makedirs(exp_root, exist_ok=True)
    
    return exp_root

# Calculate metrics for a specific configuration
def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    true_labels = [r['true_label'] for r in results]
    predicted_labels = [r['predicted_label'] for r in results]
    
    # Filter out any invalid predictions (marked as -1)
    valid_indices = [i for i, pred in enumerate(predicted_labels) if pred != -1]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [predicted_labels[i] for i in valid_indices]
    
    if not valid_pred:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(valid_true, valid_pred),
        'precision': precision_score(valid_true, valid_pred, zero_division=0),
        'recall': recall_score(valid_true, valid_pred, zero_division=0),
        'f1': f1_score(valid_true, valid_pred, zero_division=0)
    }
    
    return metrics

# Save results for a specific configuration
def save_config_results(results: List[Dict[str, Any]], exp_dir: str, model_name: str, 
                        temperature: float, top_p: float) -> None:
    if not results:
        return
    
    # Group results by configuration
    configs = {}
    for result in results:
        config_id = result['config_id']
        if config_id not in configs:
            configs[config_id] = []
        configs[config_id].append(result)
    
    # Create a summary DataFrame for all configurations
    all_metrics = []
    
    # Save each configuration separately
    for config_id, config_results in configs.items():
        # Extract configuration details from the first result
        first_result = config_results[0]
        shot_count = first_result['shot_count']
        prompt_type = first_result['prompt_type']
        
        # Create configuration-specific directory
        config_dir = os.path.join(exp_dir, f"{shot_count}-shot", prompt_type)
        os.makedirs(config_dir, exist_ok=True)
        
        # Format filename
        config_str = f"{model_name}_temp{temperature}_topp{top_p}"
        
        # Calculate metrics for this configuration
        metrics = calculate_metrics(config_results)
        
        # Add to summary metrics
        metrics_row = {
            'config_id': config_id,
            'prompt_type': prompt_type,
            'shot_count': shot_count,
            'model': model_name,
            'temperature': temperature,
            'top_p': top_p,
            **metrics
        }
        all_metrics.append(metrics_row)
        
        # Create DataFrame for this configuration
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'config_id'} for r in config_results])
        
        # Save as CSV and JSON
        df.to_csv(f'{config_dir}/classification_results_{config_str}.csv', index=False)
        with open(f'{config_dir}/classification_results_{config_str}.json', 'w', encoding='utf-8') as f:
            json.dump(config_results, f, indent=4, ensure_ascii=False)
        
        # Save metrics
        with open(f'{config_dir}/metrics_{config_str}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved results for configuration: {config_id} to {config_dir}")
        logger.info(f"Metrics for {config_id}: {metrics}")
    
    # Save all metrics to a summary file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f'{exp_dir}/all_metrics_summary.csv', index=False)
    
    # Create a pivot table for easier comparison
    pivot_df = metrics_df.pivot_table(
        index=['shot_count'], 
        columns=['prompt_type'],
        values=['accuracy', 'precision', 'recall', 'f1']
    )
    pivot_df.to_csv(f'{exp_dir}/metrics_comparison.csv')
    
    # Create visual comparison plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot settings
        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")
        
        # Prepare data for plotting
        plot_data = metrics_df.copy()
        
        # Plot accuracy by shot count for different prompt types
        plt.subplot(2, 2, 1)
        sns.lineplot(data=plot_data, x='shot_count', y='accuracy', hue='prompt_type', marker='o')
        plt.title('Accuracy by Shot Count')
        
        # Plot precision
        plt.subplot(2, 2, 2)
        sns.lineplot(data=plot_data, x='shot_count', y='precision', hue='prompt_type', marker='o')
        plt.title('Precision by Shot Count')
        
        # Plot recall
        plt.subplot(2, 2, 3)
        sns.lineplot(data=plot_data, x='shot_count', y='recall', hue='prompt_type', marker='o')
        plt.title('Recall by Shot Count')
        
        # Plot F1
        plt.subplot(2, 2, 4)
        sns.lineplot(data=plot_data, x='shot_count', y='f1', hue='prompt_type', marker='o')
        plt.title('F1 Score by Shot Count')
        
        plt.tight_layout()
        plt.savefig(f'{exp_dir}/metrics_comparison.png')
        logger.info(f"Saved metrics visualization to {exp_dir}/metrics_comparison.png")
    except Exception as e:
        logger.warning(f"Could not create visualization plots: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run hyperpartisan classification with Llama3.1-8b-Instruct')
    
    parser.add_argument('--file_path', type=str, required=True,
                      help='Path to the dataset file')
    parser.add_argument('--examples_path', type=str, required=True,
                      help='Path to the examples file')
    parser.add_argument('--output_path', type=str, default='./results',
                      help='Path to save results')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to Llama3.1-8b-Instruct model')
    parser.add_argument('--num_samples', type=int, default=50,
                      help='Number of samples to evaluate')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.3,
                      help='Top-p value for generation')
    parser.add_argument('--shot_counts', type=int, nargs='+', default=[0, 1, 3, 5, 10],
                      help='Number of shots to try')
    parser.add_argument('--max_workers', type=int, default=1,
                      help='Maximum number of worker threads')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'],
                      help='Device to run the model on')
    
    return parser.parse_args()

# Main function to run all configurations
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create experiment root directory
    exp_root = create_experiment_dirs(args.output_path)
    
    # Initialize Llama model and tokenizer
    model, tokenizer = initialize_llama(args.model_path, args.device)
    
    # Process the dataset
    processed_data = process_dataset(args.file_path)
    
    # Take a balanced sample of the data for evaluation
    # Ensure we have roughly equal numbers of both classes
    df = pd.DataFrame(processed_data)
    
    # Stratified sampling to maintain class balance
    class_0 = df[df['labels'] == 0].sample(min(args.num_samples//2, len(df[df['labels'] == 0])))
    class_1 = df[df['labels'] == 1].sample(min(args.num_samples//2, len(df[df['labels'] == 1])))
    
    # Combine and shuffle
    sampled_df = pd.concat([class_0, class_1]).sample(frac=1).reset_index(drop=True)
    sample_data = sampled_df.to_dict('records')
    
    logger.info(f"Sampled {len(sample_data)} items for evaluation ({len(class_0)} class 0, {len(class_1)} class 1)")
    
    # Load few-shot examples for all configurations
    examples_data = load_examples(args.examples_path)
    
    # Define all configurations to run
    # Each configuration is defined by: (prompt_type, shot_count)
    configurations = list(itertools.product(
        ["prompt_no_LT", "prompt_LT"],  # prompt types
        args.shot_counts         # shot counts
    ))
    
    logger.info(f"Running {len(configurations)} different configurations")
    
    # Prepare all tasks
    all_tasks = []
    for item in sample_data:
        for prompt_type, shot_count in configurations:
            all_tasks.append((
                item, model, tokenizer, prompt_type, args.temperature, args.top_p,
                shot_count, examples_data
            ))
    
    # Process all tasks with a progress bar
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item_with_config, task) for task in all_tasks]
        
        for future in tqdm(futures, total=len(all_tasks), desc="Processing items"):
            try:
                result = future.result()
                results.append(result)
                
                # Save intermediate results after each batch
                if len(results) % (len(sample_data) * 2) == 0:  # Save after every 2 configurations
                    save_config_results(results, exp_root, "llama3.1-8b-instruct", args.temperature, args.top_p)
                    
            except Exception as e:
                logger.error(f"Error processing task: {e}")
    
    # Save final results
    save_config_results(results, exp_root, "llama3.1-8b-instruct", args.temperature, args.top_p)
    
    # Save the configuration used
    with open(f'{exp_root}/config_used.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info(f"All configurations completed. Results saved to {exp_root}/")
    
    return exp_root

# Entry point
if __name__ == "__main__":
    main()