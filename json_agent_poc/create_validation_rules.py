# In create_validation_rules.py

import json
import os
from collections import defaultdict

DATA_DIR = "data"
RULES_FILE = "validation_rules.json"

def main():
    """
    Analyzes all metric files and extracts the set of possible values for key fields.
    """
    print(f"🔍 Analyzing source files in '{DATA_DIR}' to create validation rules...")


    rules = defaultdict(set)
    
    all_metrics = []
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]

    # --- Consolidate all metrics from all files ---
    for filename in json_files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'data' in data and 'metrics' in data['data']:
                all_metrics.extend(data['data']['metrics'])
        except Exception as e:
            print(f"Warning: Could not process {filename}. Error: {e}")

    if not all_metrics:
        print("❌ No metrics found. Cannot generate rules.")
        return

    # --- Extract unique values for specific keys ---
    for metric in all_metrics:
        # We only add values if they exist in the metric definition
        if 'topic' in metric:
            rules['topic'].add(metric['topic'])
        if 'function_operation' in metric:
            rules['function_operation'].add(metric['function_operation'])
        if 'duration_type' in metric:
            rules['duration_type'].add(metric['duration_type'])
        if 'window_type' in metric:
            rules['window_type'].add(metric['window_type'])
        
        # For nested keys like the grouping aliases
        if 'keys' in metric and isinstance(metric['keys'], list):
            for key in metric['keys']:
                if 'alias' in key:
                    rules['key_alias'].add(key['alias'])

    # --- Convert sets to sorted lists for clean JSON output ---
    final_rules = {key: sorted(list(value)) for key, value in rules.items()}

    # --- Save the rules to a file ---
    with open(RULES_FILE, 'w') as f:
        json.dump(final_rules, f, indent=4)

    print(f"✅ Successfully created '{RULES_FILE}' with the following rules:")
    print(json.dumps(final_rules, indent=4))

if __name__ == "__main__":
    main()