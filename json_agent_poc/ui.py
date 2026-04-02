# In ui.py

import json
import streamlit as st
from app import get_ai_response

# --- UI Configuration ---
st.set_page_config(page_title="Conversational JSON Agent", layout="wide")
st.title("🤖 Conversational AI for Metrics")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you create a metric today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Unpack the content, which might be a tuple or a single item
        content = message["content"]
        
        # Check if content is the tuple (json_data, warnings)
        if isinstance(content, tuple):
            json_data, warnings = content
            st.json(json_data)
            for warning in warnings:
                st.warning(warning)
        else:
            # Otherwise it's a string (question) or a dict (error)
            st.markdown(content)


# --- User Input & Command Handling ---
if prompt := st.chat_input("Describe the metric you need..."):
    # --- Command Handling (no changes here) ---
    if prompt.lower() in ["/new", "/bye"]:
        st.session_state.messages = [{"role": "assistant", "content": "Ok, let's start over. How can I help you create a new metric?"}]
        st.rerun()
    elif prompt.lower() == "/help":
        help_message = "..." # help message text
        st.session_state.messages.append({"role": "assistant", "content": help_message})
        st.rerun()
    # --- End Command Handling ---

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Get AI Response ---
        with st.chat_message("assistant"):
            with st.spinner("🧠 The AI is thinking..."):
                response = get_ai_response(st.session_state.messages)

            # --- MODIFIED: New, robust response handling ---
            # The response is always a tuple: (content, warnings)
            content, warnings = response

            if isinstance(content, dict):
                # It's a JSON object (the final metric or an error)
                st.json(content)
                # Store the string version in history for the next turn
                st.session_state.messages.append({"role": "assistant", "content": json.dumps(content, indent=4)})
            else:
                # It's a string (a clarifying question)
                st.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content})
            
            # Display any validation warnings (they are not saved to history)
            for warning in warnings:
                st.warning(warning)