# frontend/app.py
import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Header and Title ---
st.title("🤖 Financial Analyst Agent")
st.markdown("""
<style>
    body {
        color: #000000;
    }
    .stApp {
        background-color: #000000;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- API Communication Function ---
def call_analysis_api(company_name: str):
    """
    Calls the backend API to perform financial analysis.
    """
    api_url = "http://127.0.0.1:8000/analyze"
    payload = {"company": company_name}
    try:
        response = requests.post(api_url, json=payload, timeout=600)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None
      
    

# --- Main Application ---
st.header("Company Analysis")

# Input field for the company name
company_name = st.text_input(
    "Enter the name of the company you want to analyze:",
    placeholder="e.g., Apple, Tesla, Microsoft"
)

# Button to trigger the analysis
if st.button("Analyze Company"):
    if company_name:
        with st.spinner(f"🔍 Analyzing {company_name}... This may take a few minutes."):
            analysis_result = call_analysis_api(company_name)
            if analysis_result and "result" in analysis_result:
                st.markdown(analysis_result["result"])
            elif analysis_result and "error" in analysis_result:
                st.error(f"An error occurred during analysis: {analysis_result['error']}")
            else:
                st.error("Failed to get a valid response from the analysis service.")
            
    else:
        st.warning("Please enter a company name to analyze.")

       