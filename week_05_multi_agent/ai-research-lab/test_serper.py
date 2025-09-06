
import os

import requests
from dotenv import load_dotenv

load_dotenv("week_05_multi_agent/ai-research-lab/.env")
print(os.getenv("SERPER_API_KEY"))
url = "https://google.serper.dev/search"
headers = {
    'X-API-KEY': os.getenv("SERPER_API_KEY"),
    'Content-Type': 'application/json'
}
data = {"q": "test query"}
response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())