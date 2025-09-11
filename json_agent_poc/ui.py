# In ui.py

import streamlit as st
import json
from app import build_prompt, generate_json_config, ALLOWED_KEYS # We'll reuse some logic

# --- UI Configuration ---
st.set_page_config(page_title="Conversational JSON Agent", layout="wide")
st.title("🤖 Conversational AI for Metrics")
st.write("Start by describing the metric you need. The AI will ask clarifying questions until it has enough information to generate the JSON.")

# --- Session State Initialization ---
# This is the "memory" of our application
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you create a metric today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if the content is a dict (our final JSON) or a string (a question)
        if isinstance(message["content"], dict):
            st.json(message["content"])
        else:
            st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("What metric do you want to create?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Get AI Response ---
    with st.chat_message("assistant"):
        with st.spinner("🧠 The AI is thinking..."):
            # We create a new function to handle the conversational logic
            # Note: We are now calling the core generation function directly
            # For a more advanced agent, this logic would be more complex
            # For this PoC, we will assume a simplified flow
            # If the initial prompt is vague, the LLM will likely generate a flawed JSON
            # which we can then refine. A true conversational flow is more complex.
            
            # Let's simulate a check for required info for this PoC
            if "count" not in prompt.lower() and "sum" not in prompt.lower():
                 response_content = "What is the aggregation function you want to use (e.g., COUNT, SUM)?"
            elif "day" not in prompt.lower() and "week" not in prompt.lower() and "month" not in prompt.lower() and "hour" not in prompt.lower():
                response_content = "What is the time window for this metric (e.g., 7 days, 1 month)?"
            else:
                # If the prompt seems specific enough, try to generate
                result = generate_json_config(prompt)
                if "error" in result:
                    response_content = f"I encountered an error: {result['error']}. Can you please provide more details or clarify your request?"
                else:
                    response_content = result

        # Display the AI's response (either a question or the final JSON)
        if isinstance(response_content, dict):
            st.json(response_content)
        else:
            st.markdown(response_content)
    
    # Add the AI's response to the session history
    st.session_state.messages.append({"role": "assistant", "content": response_content})