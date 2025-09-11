# json_agent_poc/build_knowledge_base.py

import json
import os

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "data"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "metrics"
MODEL_NAME_EMBEDDING = 'all-MiniLM-L6-v2'
MODEL_NAME_LLM = "local-model"  # Using your local model for generation
ENHANCED_DESCRIPTIONS_CACHE = "enhanced_descriptions.json"


# --- Initialize LLM Client ---
# This script now also needs to talk to the LLM
openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def generate_improved_description(metric_data):
    """
    Uses an LLM to generate a high-quality description based on the metric's JSON.
    """
    # The original description is now sent to the LLM for refinement.
    original_description = metric_data.get('description', 'N/A')
    metric_data_for_prompt = metric_data.copy()

    prompt = f"""
        Please refine and improve the following description for a streaming metric based on the provided JSON data.
        The new description should be a clear and concise one-sentence explanation that incorporates details from the JSON,
        such as what the metric does, what it's grouped by, its time window, its topic, its filters and events used.
        Ensure that the core business meaning and terms are preserved.

Original Description: {original_description}

JSON Object:
{json.dumps(metric_data_for_prompt, indent=2)}

Improved Description:
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        new_description = response.choices[0].message.content.strip()
        return new_description
    except Exception as e:
        print(f"  - Warning: Could not generate new description for {metric_data.get('name')}. Error: {e}")
        return metric_data.get('description', 'No description available.') # Fallback to old description

def process_json_to_text(metric_data):
    """
    Converts a metric's JSON data into a human-readable text description for embedding.
    """
    name = metric_data.get('name', 'N/A')
    description = metric_data.get('description', 'No description') # This will now be the NEW description
    operation = metric_data.get('function_operation', 'N/A')
    window = f"{metric_data.get('duration_value')} {metric_data.get('duration_type', '').lower()}"
    
    keys = metric_data.get('keys', [])
    key_aliases = [key.get('alias', 'N/A') for key in keys]
    group_by_clause = f"grouped by {', '.join(key_aliases)}" if key_aliases else ""

    return (
        f"Metric named '{name}' calculates the {operation} over a {window} window. "
        f"Description: {description}. "
        f"It is {group_by_clause}."
    )

def main():
    print("🚀 Starting knowledge base creation with description enhancement...")

    # --- Load description cache ---
    if os.path.exists(ENHANCED_DESCRIPTIONS_CACHE):
        with open(ENHANCED_DESCRIPTIONS_CACHE, 'r') as f:
            description_cache = json.load(f)
    else:
        description_cache = {}

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    # --- Consolidate all metrics from all files ---
    all_metrics = []
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files. Consolidating metrics...")
    for filename in json_files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'data' in data and 'metrics' in data['data']:
                all_metrics.extend(data['data']['metrics'])
        except Exception as e:
            print(f"An error occurred while reading {filename}: {e}")

    if not all_metrics:
        print("No metrics found to process.")
        return

    print(f"Found a total of {len(all_metrics)} metrics to process.")

    documents, metadatas, ids = [], [], []
    new_descriptions_since_last_save = 0

    # --- Process all metrics with a single progress bar ---
    for metric in tqdm(all_metrics, desc="Processing & Enhancing Metrics"):
        metric_id = metric.get('id')
        if not metric_id: continue

        # --- ENHANCEMENT STEP ---
        if metric_id in description_cache:
            improved_description = description_cache[metric_id]
        else:
            # 1. Generate a new, better description
            improved_description = generate_improved_description(metric)
            description_cache[metric_id] = improved_description
            new_descriptions_since_last_save += 1

            # 2. Persist cache in batches
            if new_descriptions_since_last_save >= 10:
                # Use tqdm.write to avoid breaking the progress bar
                tqdm.write(f"\nSaving cache to disk ({len(description_cache)} items)...")
                with open(ENHANCED_DESCRIPTIONS_CACHE, 'w') as f:
                    json.dump(description_cache, f, indent=2)
                new_descriptions_since_last_save = 0

        # 3. Replace the old description with the new one
        metric['description'] = improved_description

        text_representation = process_json_to_text(metric)
        documents.append(text_representation)
        metadatas.append({'original_json': json.dumps(metric)})
        ids.append(metric_id)

    # --- Final cache save ---
    if new_descriptions_since_last_save > 0:
        print(f"\nSaving final cache changes to disk ({len(description_cache)} items)...")
        with open(ENHANCED_DESCRIPTIONS_CACHE, 'w') as f:
            json.dump(description_cache, f, indent=2)

    if documents:
        print(f"\nAdding {len(documents)} enhanced documents to the collection...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print("✅ Enhanced knowledge base created successfully!")
    else:
        print("No documents were processed.")



if __name__ == "__main__":
    main()