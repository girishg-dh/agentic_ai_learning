#!/usr/bin/env python3

import json
from langchain.document_loaders import JSONLoader

# Test the jq schema
json_file_path = "data/metrics.json"

jq_schema = (
    '.data.metrics[] | "Metric Name: \\(.name) | '
    'Description: \\(.description) | '
    'Calculation: \\(.function_operation) on key \\(.function_key) | '
    'Metric Group: \\(.metric_group // "Not specified") | '
    'Status: \\(.status) | '
    'Duration: \\(.duration_value) \\(.duration_type)"'
)

try:
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema=jq_schema,
        text_content=False
    )
    documents = loader.load()
    
    print(f"Successfully loaded {len(documents)} documents")
    
    # Find the specific metric
    for doc in documents:
        if "cntd_cash_users_device_30d" in doc.page_content:
            print("Found the metric:")
            print(doc.page_content)
            break
    else:
        print("Metric not found")
        
except Exception as e:
    print(f"Error: {e}")