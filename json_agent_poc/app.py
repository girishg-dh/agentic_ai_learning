# In app.py

import json
import re 
import chromadb
from dotenv import load_dotenv
from llm_interface import get_llm_response 

# --- Configuration & Clients (no changes here) ---
load_dotenv()
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "metrics"
RULES_FILE = "validation_rules.json"
MODEL_NAME = "local-model"
ALLOWED_KEYS = [
    "name", "description", "window_type", "duration_type", "duration_value",
    "keys", "function_type", "function_key", "function_data_type", "function_value",
    "function_operation", "topic", "filter", "historical_data_required",
    "active", "global_entity_ids", "created_by", "default_value", "is_platform_level",
    "metric_group"
]

chroma_client = None
validation_rules = None

def get_chroma_client():
    global chroma_client
    if chroma_client is None:
        print("🚀 Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return chroma_client

def get_validation_rules():
    global validation_rules
    if validation_rules is None:
        print("🚀 Loading validation rules...")
        try:
            with open(RULES_FILE, 'r') as f:
                validation_rules = json.load(f)
        except FileNotFoundError:
            print("Warning: validation_rules.json not found. Skipping validation.")
            validation_rules = {}
    return validation_rules


def validate_metric(metric_json):
    warnings = []
    rules = get_validation_rules()
    if not rules:
        return warnings
    if 'topic' in metric_json and metric_json['topic'] not in rules.get('topic', []):
        warnings.append(f"Warning: The topic '{metric_json['topic']}' is not a known topic.")
    if 'function_operation' in metric_json and metric_json['function_operation'] not in rules.get('function_operation', []):
        warnings.append(f"Warning: The function '{metric_json['function_operation']}' is not a standard operation.")
    if 'duration_type' in metric_json and metric_json['duration_type'] not in rules.get('duration_type', []):
        warnings.append(f"Warning: The duration type '{metric_json['duration_type']}' is not a standard duration.")
    return warnings

def get_ai_response(history):
    """
    This is the main "brain" of the agent. It returns the generated JSON and any validation warnings.
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    last_user_message = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
    results = collection.query(query_texts=[last_user_message], n_results=4)
    context_metrics = [json.loads(meta['original_json']) for meta in results['metadatas'][0]] if results['metadatas'] else []
    examples = "\n\n".join([json.dumps(metric, indent=2) for metric in context_metrics])

    # Get validation rules for the prompt
    rules = get_validation_rules()
    rules_text = json.dumps(rules, indent=2) if rules else "No validation rules available"
    
    # --- MODIFIED: Added validation rules and stricter instructions to the prompt ---
    system_prompt = f"""
You are an expert AI assistant that helps users create JSON configurations for a streaming service.
Your goal is to gather all necessary information through a conversation. Remember user are not technical and has less information about schema

1.  Analyze the user's request and the entire conversation history.
2.  Identify missing information needed for a complete metric (e.g., duration, grouping keys, function).
3.  If information is missing, ask ONE clear, specific question to the user.
4.  Once you have all the necessary information, generate the final JSON object.
5.  **IMPORTANT: Do not engage in conversational chit-chat or greetings. Your ONLY valid outputs are either a single question or a single, raw JSON object.**
6.  Your final response should ONLY be the JSON object, with no extra text or markdown.
7.  Do NOT include these keys in the final JSON: `id`, `version`, `status`, and any `_date` or `_time` keys.
8.  When users are not sure use your best judgment to return a valid response
9.  **VALIDATION RULES: You MUST use only these allowed values:**
{rules_text}

--- EXAMPLES OF COMPLETE JSON ---
{examples}
--- END EXAMPLES ---
"""
    

    try:
        # --- Single call to our new LLM interface ---
        response_content = get_llm_response(history, system_prompt)

        # Handle LM Studio special tokens - extract content after <|message|>
        if '<|message|>' in response_content:
            message_start = response_content.find('<|message|>') + len('<|message|>')
            response_content = response_content[message_start:]
        
        # Clean any remaining special tokens
        cleaned_content = re.sub(r'<\|[^|]*\|>', '', response_content).strip()
        
        json_start_index = cleaned_content.find('{')
        json_end_index = cleaned_content.rfind('}') + 1

        if json_start_index != -1 and json_end_index != -1:
            json_string = cleaned_content[json_start_index:json_end_index]
            parsed_json = json.loads(json_string)
            cleaned_json = {key: value for key, value in parsed_json.items() if key in ALLOWED_KEYS}
            warnings = validate_metric(cleaned_json)
            
            # If validation fails, retry with feedback
            if warnings:
                feedback_prompt = f"The JSON you generated has validation errors: {'; '.join(warnings)}. Please fix these issues and return a corrected JSON using only the allowed values from the validation rules."
                history_with_feedback = history + [{"role": "assistant", "content": json.dumps(cleaned_json)}, {"role": "user", "content": feedback_prompt}]
                
                try:
                    retry_response = get_llm_response(history_with_feedback, system_prompt)
                    if '<|message|>' in retry_response:
                        message_start = retry_response.find('<|message|>') + len('<|message|>')
                        retry_response = retry_response[message_start:]
                    
                    retry_cleaned = re.sub(r'<\|[^|]*\|>', '', retry_response).strip()
                    retry_json_start = retry_cleaned.find('{')
                    retry_json_end = retry_cleaned.rfind('}') + 1
                    
                    if retry_json_start != -1 and retry_json_end != -1:
                        retry_json_string = retry_cleaned[retry_json_start:retry_json_end]
                        retry_parsed = json.loads(retry_json_string)
                        retry_cleaned_json = {key: value for key, value in retry_parsed.items() if key in ALLOWED_KEYS}
                        retry_warnings = validate_metric(retry_cleaned_json)
                        return retry_cleaned_json, retry_warnings
                except:
                    pass  # If retry fails, return original
            
            return cleaned_json, warnings
        else:
            return cleaned_content, []

    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return {"error": f"An error occurred: {e}"}, []

