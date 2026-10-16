import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="Financial RAG Analytics Assistant",
    page_icon="📈",
    layout="wide"
)

API_URL = "http://localhost:8000/v1/chat"

st.title("📈 Docling Financial RAG Analytics Assistant")
st.caption("Hybrid Vector + BM25 Search with Table-Aware Context Boost")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "stream_sess_001"

if "state" not in st.session_state:
    st.session_state.state = {
        "company": None,
        "fiscal_year": None,
        "target_metric": None
    }

# Sidebar Controls & Metadata Inspector
with st.sidebar:
    st.header("⚙️ Session Controls")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.subheader("🔍 Active State Inspector")
    st.json(st.session_state.state)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If retrieved sources exist, show inside an expander
        if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
            with st.expander("📚 View Retrieved Sources & Citation Chunks"):
                for idx, chunk in enumerate(msg["retrieved_chunks"]):
                    chunk_type = chunk.get("chunk_type", "text").upper()
                    st.markdown(f"**Source [{idx+1}] — Type: `{chunk_type}` | Page: `{chunk.get('page_number', 'N/A')}` | Score: `{chunk.get('final_score', 0):.4f}`**")
                    st.code(chunk.get("content", ""), language="markdown")

# Chat Input & Interaction Handling
# Chat Input & Interaction Handling
if user_input := st.chat_input("Ask a financial question (e.g., 'What was Duolingo's subscription revenue in 2023?')..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format history payload for FastAPI API
    formatted_history = [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages[:-1]
    ]

    payload = {
        "session_id": st.session_state.session_id,
        "query": user_input,
        "history": formatted_history,
        "state": st.session_state.state
    }

    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Retrieval (Vector + BM25) & Reranking..."):
            try:
                # 1. First check if backend is ready (prevents cold-start gateway drops)
                try:
                    ready_res = requests.get("http://localhost:8000/health/ready", timeout=5)
                    if ready_res.status_code != 200:
                        st.warning("Server is still warming up models. Please wait a moment and try again.")
                        st.stop()
                except requests.exceptions.RequestException:
                    st.error("FastAPI server is offline or starting up. Ensure backend is running.")
                    st.stop()

                # 2. Make chat request with a generous timeout (60s for cold-start model calls)
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated.")
                    retrieved_chunks = data.get("retrieved_chunks", [])
                    updated_state = data.get("state", st.session_state.state)

                    # Update local state
                    st.session_state.state = updated_state

                    # Render answer
                    st.markdown(answer)

                    # Render expandable source chunks
                    if retrieved_chunks:
                        with st.expander("📚 View Retrieved Sources & Citation Chunks"):
                            for idx, chunk in enumerate(retrieved_chunks):
                                chunk_type = chunk.get("chunk_type", "text").upper()
                                st.markdown(f"**Source [{idx+1}] — Type: `{chunk_type}` | Page: `{chunk.get('page_number', 'N/A')}` | Score: `{chunk.get('final_score', 0):.4f}`**")
                                st.code(chunk.get("content", ""), language="markdown")

                    # Append assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "retrieved_chunks": retrieved_chunks
                    })
                elif response.status_code == 503:
                    st.warning("The application is finishing its initialization warmup. Please try sending your message again in 5 seconds.")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("The request took too long to process. The first query loads embedding models and vector indexes into memory—please try submitting your query again now that initialization is complete.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to FastAPI Backend at {API_URL}: {str(e)}")