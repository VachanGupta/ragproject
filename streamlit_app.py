# In streamlit_app.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantumLeap X Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- App Title ---
st.title("🤖 QuantumLeap X - RAG Assistant")
st.caption("This chat assistant is powered by a local RAG pipeline and the Groq API.")

# --- API Endpoint ---
API_URL = "http://127.0.0.1:8000/api/v1/chat"

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("Ask about the QuantumLeap X..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the FastAPI backend
            response = requests.post(API_URL, json={"query": prompt})
            response.raise_for_status() # Raise an exception for bad status codes
            
            full_response = response.json()
            answer = full_response.get("answer", "Sorry, I encountered an error.")
            
            message_placeholder.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            error_message = f"Failed to connect to the backend: {e}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})