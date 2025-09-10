# frontend/app.py
import json

import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanceIQ Research",
    page_icon="📊",
    layout="wide"
)

# --- Professional Styling ---
st.markdown("""
<style>
    /* Light Mode Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .header-container {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 2px solid #e8f4f8;
        margin-bottom: 2rem;
    }
    
    .company-title {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    .analysis-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Dark Mode Styles */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .main-container {
            background: rgba(30, 30, 46, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header-container {
            border-bottom: 2px solid #374151;
        }
        
        .company-title {
            color: #60a5fa;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            color: #9ca3af;
        }
        
        .analysis-container {
            background: #1f2937;
            border-left: 4px solid #60a5fa;
        }
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stMarkdown {
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="main-container">
    <div class="header-container">
        <h1 class="company-title">📊 FinanceIQ Research</h1>
        <p class="subtitle">Professional Financial Analysis & Market Intelligence</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- API Communication Function ---
def call_analysis_api(company_name: str):
    """
    Calls the backend API to perform financial analysis.
    """
    api_url = "https://financial-agent-backend.onrender.com/analyze""
    payload = {"company": company_name}
    
    try:
        response = requests.post(api_url, json=payload, timeout=600) # 10-minute timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None


# --- Main Application ---
st.markdown("""
<div class="main-container">
    <h2 style="color: #1e3a8a; margin-bottom: 1.5rem; font-weight: 600;">🏢 Company Analysis Dashboard</h2>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Input field for the company name
        company_name = st.text_input(
            "Enter Company Name for Analysis",
            placeholder="e.g., Apple, Tesla, Microsoft, Amazon",
            help="Enter the full company name or stock ticker symbol"
        )
        
        # Button to trigger the analysis
        if st.button("🔍 Start Financial Analysis", type="primary"):
            if company_name:
                # Show a spinner while the analysis is in progress
                with st.spinner(f"📊 Analyzing {company_name}... Gathering financial data and market insights..."):
                    analysis_result = call_analysis_api(company_name)

                # Display results in styled container
                st.markdown("""
                <div class="analysis-container">
                    <h3 style="color: #1e3a8a; margin-bottom: 1rem;">📈 Analysis Report</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if analysis_result and "result" in analysis_result:
                    # Display the result using markdown for better formatting
                    st.markdown(analysis_result["result"])
                elif analysis_result and "error" in analysis_result:
                    st.error(f"⚠️ Analysis Error: {analysis_result['error']}")
                else:
                    st.error("❌ Failed to retrieve analysis. Please check the service connection.")

            else:
                st.warning("⚠️ Please enter a company name to proceed with the analysis.")