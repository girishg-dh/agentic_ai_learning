import argparse
import json
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "metrics"
MODEL_NAME = "local-model" 

# --- Initialize Clients ---
load_dotenv()
openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


# --- NEW: Define the list of keys the API accepts ---
ALLOWED_KEYS = [
    "name", "description", "window_type", "duration_type", "duration_value",
    "keys", "function_type", "function_key", "function_data_type", "function_value",
    "function_operation", "topic", "filter", "historical_data_required",
    "active", "global_entity_ids", "created_by", "default_value", "is_platform_level",
    "metric_group" # Added metric_group as it was in the initial full schema
]


def build_prompt(query, context_metrics):
    """Builds the prompt for the LLM with the user query and retrieved context."""
    
    examples = "\n\n".join([json.dumps(metric, indent=2) for metric in context_metrics])
    
    # --- MODIFIED: Updated the system prompt ---
    system_prompt = f"""
You are an expert AI assistant that generates JSON configurations for a streaming aggregation service.
Your task is to create a JSON configuration based on the user's request.
Use the provided examples to understand the structure, but only include the fields necessary for the API.

--- EXAMPLES ---
{examples}
--- END EXAMPLES ---

Your response MUST be ONLY the final JSON object.

IMPORTANT: The final JSON must NOT include the following keys: `id`, `version`, `status`, `created_date`, `updated_by`, `updated_date`, `deleted`, `aggregation_start_time`, `exp_aggregation_start_time`.
"""
    return system_prompt, query


def main(user_query):
    """
    Main function to handle user query and generate JSON.
    """
    print(f"🔍 Received query: '{user_query}'")
    
    print("🧠 Searching knowledge base for relevant examples...")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )
    
    context_metrics = [json.loads(meta['original_json']) for meta in results['metadatas'][0]]
    
    system_prompt, user_prompt = build_prompt(user_query, context_metrics)
    
    print("🤖 Calling the local LLM via LM Studio to generate the configuration...")
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        generated_json_str = response.choices[0].message.content
        
        print("\n✅ Generated JSON Configuration (Cleaned):")
        
        # --- MODIFIED: Parse, clean, and then print the result ---
        parsed_json = json.loads(generated_json_str)
        
        # Create a new dictionary with only the allowed keys
        cleaned_json = {key: value for key, value in parsed_json.items() if key in ALLOWED_KEYS}
        
        print(json.dumps(cleaned_json, indent=4))
        
    except json.JSONDecodeError:
        print("\n❌ Error: The LLM did not return a valid JSON. Here is the raw output:")
        print(generated_json_str)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSON metric configuration from a natural language query.")
    parser.add_argument("query", type=str, help="The natural language query describing the metric.")
    args = parser.parse_args()
    main(args.query)