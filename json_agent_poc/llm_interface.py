# In llm_interface.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import anthropic

# --- Load Configurations ---
load_dotenv()

# --- Client is initialized on-demand ---
llm_client = None
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "LMSTUDIO").upper()


def get_llm_client():
    """Initializes and returns the LLM client based on the .env config."""
    global llm_client
    if llm_client is not None:
        return llm_client

    print(f"🚀 Initializing LLM Provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "CLAUDE":
        llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif LLM_PROVIDER == "GEMINI":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        llm_client = genai.GenerativeModel(model_name)
    elif LLM_PROVIDER == "OLLAMA":
        base_url = os.getenv("OLLAMA_BASE_URL")
        llm_client = OpenAI(base_url=base_url, api_key="not-needed")
    else:  # Default to LMSTUDIO
        base_url = os.getenv("LMSTUDIO_BASE_URL")
        llm_client = OpenAI(base_url=base_url, api_key="not-needed")

    return llm_client

def get_llm_response(history, system_prompt):
    """Calls the appropriate LLM to get a response."""
    client = get_llm_client()

    if LLM_PROVIDER == "CLAUDE":
        model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=history,
                temperature=0.3,
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: Could not get response from Claude. Details: {e}"
    elif LLM_PROVIDER == "GEMINI":
        gemini_messages = []
        gemini_messages.append({'role': 'user', 'parts': [system_prompt]})
        gemini_messages.append({'role': 'model', 'parts': ['I understand. I will help create JSON configurations following the rules and examples provided.']})
        for msg in history:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_messages.append({'role': role, 'parts': [msg['content']]})
        try:
            response = client.generate_content(gemini_messages, generation_config={"temperature": 0.3})
            if response and response.text:
                return response.text
            else:
                return "Error: Gemini returned an empty response."
        except Exception as e:
            return f"Error: Could not get response from Gemini. Details: {e}"
    else:
        # OpenAI-Compatible Logic (LMSTUDIO / OLLAMA)
        model_name = os.getenv(f"{LLM_PROVIDER}_MODEL")
        messages_for_api = [{"role": "system", "content": system_prompt}] + history
        try:
            response = client.chat.completions.create(model=model_name, messages=messages_for_api, temperature=0.3)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: Could not get response from local LLM. Is the server running? Details: {e}"