import json
import re
import os
import yaml
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
import time

# Load configuration from YAML file
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Process the dataset
def process_dataset(file_path: str) -> List[Dict[str, Any]]:
    print("Processing dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        # Filter out hyperpartisan texts
        if item['labels'] == 1:
            processed_item = {
                'text': item['text'],
                'labels': item['labels'],
                'linguistic_traits': {}
            }
            
            # Extract linguistic traits without spans and exclude Agents and Entities
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

# Load few-shot examples from file
def load_examples(examples_path: str) -> Dict[str, Any]:
    print("Loading few-shot examples...")
    with open(examples_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    return examples_data

# Format examples for few-shot learning
def format_examples(examples_data: Dict[str, Any], shot_count: int, use_explanations: bool) -> Tuple[str, str]:
    if shot_count == 0:
        return "", ""
    
    # Determine which dictionaries to use based on whether explanations are needed
    if use_explanations:
        original_dict = examples_data["DPP"]["dic_h"]
        paraphrased_dict = examples_data["DPP"]["dic_h_paraphrased__expl"]
        # Get the keys based on explanations pattern
        example_keys = [key for key in original_dict.keys() if key.startswith("PAREXPL_h_shot_run_")][:shot_count]
    else:
        original_dict = examples_data["DPP"]["dic_h"]
        paraphrased_dict = examples_data["DPP"]["dic_h_paraphrased_no_expl"]
        # Get the keys based on no explanations pattern
        example_keys = [key for key in original_dict.keys() if key.startswith("PAR_h_shot_run_")][:shot_count]
    
    # If we don't have enough examples for the requested shot count, take what we have
    example_keys = example_keys[:shot_count]
    
    # Format examples for both prompts
    examples_text_prompt1 = ""
    examples_text_prompt2 = ""
    
    for key in example_keys:
        original = original_dict.get(key, {})
        paraphrased = paraphrased_dict.get(key, {})
        
        if not original or not paraphrased:
            continue
        
        # Extract data
        original_text = original.get("text", "")
        linguistic_traits = original.get("linguistic_traits", {})
        traits_description = ", ".join([f"{k}: {v}" for k, v in linguistic_traits.items() if v])
        
        # Ensure we have a modified text
        modified_text = paraphrased.get("Testo_modificato", "")
        if not modified_text:
            modified_text = paraphrased.get("modified_text", "")
        
        # Format linguistic traits for example JSON
        json_format = {
            "label": "1",  # Since we're only using hyperpartisan texts
            "Figurative_Speech": linguistic_traits.get("Figurative_Speech", []),
            "Irony/Sarcasm": linguistic_traits.get("Irony/Sarcasm", []),
            "Epithet": linguistic_traits.get("Epithet", ""),
            "Neologism": linguistic_traits.get("Neologism", []),
            "Loaded_language": linguistic_traits.get("Loaded_language", []),
            "Hyperbolic_Language": linguistic_traits.get("Hyperbolic_Language", []),
            "Testo_modificato": modified_text
        }
        
        # Add explanation if available and requested
        if use_explanations:
            explanation = paraphrased.get("Spiegazione", "")
            json_format["Spiegazione"] = explanation
        
        # Format the example for prompt 1
        examples_text_prompt1 += f"""
### Esempio:
Testo originale: "{original_text}"

Output:
```json
{json.dumps(json_format, ensure_ascii=False, indent=2)}
```

"""
        
        # Format the example for prompt 2
        examples_text_prompt2 += f"""
### Esempio:
Testo originale: "{original_text}"
Tratti linguistici: {traits_description}

Output:
```json
{json.dumps(json_format, ensure_ascii=False, indent=2)}
```

"""
    
    return examples_text_prompt1, examples_text_prompt2

# Create prompts for paraphrasing
def create_paraphrase_prompts(item: Dict[str, Any], examples_text_prompt1: str, examples_text_prompt2: str, use_explanations: bool) -> Tuple[str, str]:
    text = item['text']
    linguistic_traits = item['linguistic_traits']
    
    # First prompt with examples if available
    explanation_instruction = ""
    if use_explanations:
        explanation_instruction = "\n4. **Spiegazione**: Rispondi alla domanda: \"Perché questa frase è meno iperpartigiana rispetto all'originale?\" Evidenzia i cambiamenti effettuati e il loro impatto sul tono del testo."
    
    few_shot_instruction = ""
    if examples_text_prompt1:
        few_shot_instruction = "\n\nEcco alcuni esempi di come dovresti analizzare e modificare il testo:"
    
    prompt1 = f"""Sei un assistente italiano esperto in comunicazione politica e analisi del linguaggio. Il tuo compito è identificare e ridurre i tratti linguistici che rendono un testo hyperpartisan, senza alterarne il significato originale.

Un testo è considerato hyperpartisan quando esprime un'opinione estremizzata su una categoria ideologica, politica o sociale, spesso polarizzando il pubblico attraverso un linguaggio carico emotivamente, ironico/sarcastico o eccessivamente iperbolico.  

### Istruzioni:
1. **Classificazione**: Determina se il testo è "hyperpartisan" o "neutral". Se hyperpartisan, la label è 1, altrimenti 0.
2. **Analisi linguistica**: Identifica e classifica i tratti linguistici che contribuiscono all'hyperpartisan bias. I tratti da analizzare sono:  
   - **Metafore**  
   - **Ironia/Sarcasmo**  
   - **Loaded Language** (termini caricati emotivamente: possono essere singole o più parole)  
   - **Neologismi**  
   - **Linguaggio Iperbolico** (uso di iperboli, esagerazioni)  
   Elenca le parole o frasi che li veicolano.
3. **Testo modificato**: Se il testo è identificato come hyperpartisan (1), riscrivi il testo in modo che risulti più neutrale, mantenendo il significato originale. Inserisci il nuovo testo nella variabile `"Testo_modificato"`.{explanation_instruction}{few_shot_instruction}

{examples_text_prompt1}
### Output richiesto:
Rispondi nel seguente formato JSON:

json:
{{
  "label": "1/0",
  "Metafore": ["esempio1", "esempio2"],
  "Ironia/Sarcasmo": ["esempio1", "esempio2"],
  "Neologismi": ["esempio1", "esempio2"],
  "Loaded Language": ["esempio1", "esempio2"],
  "Linguaggio Iperbolico": ["esempio1", "esempio2"],
  "Testo_modificato": "testo riscritto in modo neutrale"
  {'"Spiegazione": "Analisi dei cambiamenti effettuati e motivazione."' if use_explanations else ''}
}}

Questo è il testo originale da analizzare: "{text}"
"""

    # Second prompt with linguistic traits focus
    traits_description = ", ".join([f"{k}: {v}" for k, v in linguistic_traits.items() if v])
    
    few_shot_instruction2 = ""
    if examples_text_prompt2:
        few_shot_instruction2 = "\n\nEcco alcuni esempi di come dovresti analizzare e modificare il testo:"
        
    explanation_point = ""
    if use_explanations:
        explanation_point = "\n4. Fornisci una spiegazione dettagliata dei cambiamenti effettuati."
        
    prompt2 = f"""Sei un assistente italiano esperto in comunicazione politica e analisi del linguaggio. Il tuo compito è identificare e ridurre i tratti linguistici che rendono un testo hyperpartisan, senza alterarne il significato originale.

Vorrei che parafrasassi il seguente testo, agendo esclusivamente sui tratti linguistici indicati. La frase non deve cambiare di significato. Si tratta di ridurre dei bias linguistici che si avverano attraverso alcuni fenomeni particolari.

Testo originale: "{text}"

Contiene i seguenti tratti linguistici a cui prestare attenzione: {traits_description}

Segui le istruzioni di seguito:
1. **Classificazione**: Determina se il testo è "hyperpartisan" o "neutral". Se hyperpartisan, la label è 1, altrimenti 0.
2. Se il testo è classificato come hyperpartisan, riscrivilo in maniera oggettiva, mantenendo il significato generale della frase.
3. Limitati a neutralizzare le parti di testo indicate fornendo spiegazioni sul tuo processo.{explanation_point}{few_shot_instruction2}

{examples_text_prompt2}
### Output richiesto:
Rispondi nel seguente formato JSON:

json:
{{
  "label": "1/0",
  "Metafore": ["esempio1", "esempio2"],
  "Ironia/Sarcasmo": ["esempio1", "esempio2"],
  "Neologismi": ["esempio1", "esempio2"],
  "Loaded Language": ["esempio1", "esempio2"],
  "Linguaggio Iperbolico": ["esempio1", "esempio2"],
  "Testo_modificato": "testo riscritto in modo neutrale"
  {'"Spiegazione": "Analisi dei cambiamenti effettuati e motivazione."' if use_explanations else ''}
}}
"""

    return prompt1, prompt2

# Extract JSON from the model response
def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    # Look for a JSON object in the response
    try:
        # Try to find JSON with regex first
        json_pattern = r'({[\s\S]*?})'
        json_matches = re.findall(json_pattern, response_text)
        
        if json_matches:
            for json_str in json_matches:
                try:
                    # Try to parse each match as JSON
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found with regex, try to extract JSON from code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_blocks = re.findall(code_block_pattern, response_text)
        
        if code_blocks:
            for block in code_blocks:
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
                
        # Final attempt: try to find JSON-like structure in the entire text
        try:
            cleaned_text = response_text.strip()
            # Check if the response contains proper JSON structure
            if '{' in cleaned_text and '}' in cleaned_text:
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}') + 1
                json_text = cleaned_text[start_idx:end_idx]
                return json.loads(json_text)
        except json.JSONDecodeError:
            pass
            
        # If all fails, return a default structure
        return {
            "label": None,
            "Metafore": [],
            "Ironia/Sarcasmo": [],
            "Neologismi": [],
            "Loaded Language": [],
            "Linguaggio Iperbolico": [],
            "Testo_modificato": "Failed to extract modified text",
            "Spiegazione": "Failed to extract explanation" if "Spiegazione" in response_text else None
        }
            
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        print(f"Response text: {response_text}")
        return {
            "label": None,
            "Metafore": [],
            "Ironia/Sarcasmo": [],
            "Neologismi": [],
            "Loaded Language": [],
            "Linguaggio Iperbolico": [],
            "Testo_modificato": "Failed to extract modified text",
            "Spiegazione": "Failed to extract explanation" if "Spiegazione" in response_text else None,
            "raw_response": response_text
        }

# Generate paraphrases using OpenAI API
def generate_paraphrase(client: OpenAI, prompt: str, model: str, temperature: float, top_p: float) -> Tuple[str, Dict[str, Any], str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in political communication and language analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=1000,
            response_format={"type": "text"}
        )
        
        # Get the response text
        assistant_response = response.choices[0].message.content
        
        # Extract the JSON content
        json_data = extract_json_from_response(assistant_response)
        
        # Get the modified text from the JSON using the correct key "Testo_modificato"
        if json_data and "Testo_modificato" in json_data:
            modified_text = json_data["Testo_modificato"]
        else:
            modified_text = "Failed to extract modified text"
        
        return modified_text, json_data, assistant_response
    
    except Exception as e:
        print(f"Error generating paraphrase: {e}")
        return "API request failed", {"error": str(e)}, f"API request failed: {str(e)}"

# Create experiment directories
def create_experiment_dirs(output_path: str, shot_count: int, use_explanations: bool) -> str:
    # Format the experiment identifier
    if shot_count == 0:
        exp_id = "0-shot"
    else:
        exp_type = "with_expl" if use_explanations else "no_expl"
        exp_id = f"{shot_count}-shot_{exp_type}"
    
    # Create experiment directory
    exp_dir = os.path.join(output_path, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

# Main function to run the experiment
def main(config_path: str):
    # Load configuration from YAML
    config = load_config(config_path)
    
    # Extract configuration parameters
    file_path = config.get("file_path")
    examples_path = config.get("examples_path")
    output_path = config.get("output_path")
    num_samples = config.get("num_samples", 20)
    shot_count = config.get("shot_count", 0)
    use_explanations = config.get("use_explanations", False)
    temperature = config.get("temperature", 0)
    top_p = config.get("top_p", 0.3)
    model = config.get("model", "gpt-4o-mini")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=config.get("openai_api_key"))
    
    # Process the dataset (hyperpartisan texts only)
    processed_data = process_dataset(file_path)
    
    # Take the requested number of hyperpartisan entries
    sample_data = processed_data[:num_samples]
    
    # Create experiment directory
    exp_dir = create_experiment_dirs(output_path, shot_count, use_explanations)
    
    # Load few-shot examples if needed
    examples_text_prompt1 = ""
    examples_text_prompt2 = ""
    if shot_count > 0:
        examples_data = load_examples(examples_path)
        examples_text_prompt1, examples_text_prompt2 = format_examples(
            examples_data, shot_count, use_explanations
        )
    
    # Prepare results for both prompts
    results_prompt1 = []
    results_prompt2 = []
    
    print(f"Generating modified texts for {len(sample_data)} samples...")
    print(f"Configuration: {shot_count}-shot, " + 
          f"{'with explanations' if use_explanations else 'without explanations'}, " +
          f"temp={temperature}, top_p={top_p}, model={model}")
    
    for i, item in enumerate(sample_data):
        print(f"Processing sample {i+1}/{len(sample_data)}...")
        
        # Generate both prompts with examples if available
        prompt1, prompt2 = create_paraphrase_prompts(
            item, examples_text_prompt1, examples_text_prompt2, use_explanations
        )
        
        # Generate modifications for prompt 1
        modified_text1, json_response1, raw_response1 = generate_paraphrase(
            client, prompt1, model, temperature, top_p
        )
        results_prompt1.append({
            'original': item['text'],
            'modified': modified_text1,
            'linguistic_traits': item['linguistic_traits'],
            'model_json_response': json_response1,
            'raw_response': raw_response1
        })
        
        # Add a delay to avoid rate limits
        time.sleep(1)
        
        # Generate modifications for prompt 2
        modified_text2, json_response2, raw_response2 = generate_paraphrase(
            client, prompt2, model, temperature, top_p
        )
        results_prompt2.append({
            'original': item['text'],
            'modified': modified_text2,
            'linguistic_traits': item['linguistic_traits'],
            'model_json_response': json_response2,
            'raw_response': raw_response2
        })
        
        # Add a delay to avoid rate limits
        time.sleep(1)
    
    # Convert to DataFrames
    df_prompt1 = pd.DataFrame(results_prompt1)
    df_prompt2 = pd.DataFrame(results_prompt2)
    
    # Define experiment-specific filename components
    config_str = f"{model}_temp{temperature}_topp{top_p}"
    
    # Save results for Prompt 1
    df_prompt1.to_csv(f'{exp_dir}/modification_results_prompt1_{config_str}.csv', index=False)
    with open(f'{exp_dir}/modification_results_prompt1_{config_str}.json', 'w', encoding='utf-8') as f:
        json.dump(results_prompt1, f, indent=4, ensure_ascii=False)
    
    # Save results for Prompt 2
    df_prompt2.to_csv(f'{exp_dir}/modification_results_prompt2_{config_str}.csv', index=False)
    with open(f'{exp_dir}/modification_results_prompt2_{config_str}.json', 'w', encoding='utf-8') as f:
        json.dump(results_prompt2, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {exp_dir}/")
    
    return df_prompt1, df_prompt2

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)