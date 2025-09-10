# backend/llm.py

import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


def get_llm():
    # Your get_llm function remains here...
    print("The key is: ", os.getenv("LLM_PROVIDER").lower())
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if llm_provider == "ollama":
        print("Using Ollama model")
        from langchain_community.llms import Ollama
        return Ollama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
    elif llm_provider == "lmstudio":
        print("Using LM Studio model")
        return LLM(model=os.getenv("LMSTUDIO_MODEL"), base_url=os.getenv("LMSTUDIO_BASE_URL"), api_key=os.getenv("LMSTUDIO_API_KEY"))
    else:
        print("Using Gemini model")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return LLM(model="gemini/gemini-2.0-flash-lite", temperature=0.7, api_key=gemini_api_key)