# admin/app.py
import asyncio
import sys
from pathlib import Path

# Silence Windows 10054 connection reset warnings on socket termination
def silence_win_error_10054(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
        return
    loop.default_exception_handler(context)

if sys.platform == "win32":
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silence_win_error_10054)
    except RuntimeError:
        pass

import streamlit as st
from parse_pdf import run_pipeline
from seed_qdrant import seed_database

# Base directory configuration
BASE_DIR = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot")
DATA_DIR = BASE_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
QDRANT_STORAGE_PATH = str(DATA_DIR / "qdrant_storage")

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Financial Document Ingestion Admin", layout="wide")
st.title("Multi-Document Financial RAG Ingestion Pipeline")

st.sidebar.header("Pipeline Configuration")
doc_id = st.sidebar.text_input("Document Identifier", value="duolingo_2024_10k")
qdrant_path = st.sidebar.text_input("Qdrant Storage Path", value=QDRANT_STORAGE_PATH)

uploaded_files = st.file_uploader(
    "Upload Financial PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_dir = DATA_DIR / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        pdf_file_path = temp_dir / uploaded_file.name

        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Ready: {uploaded_file.name}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(f"1. Parse {uploaded_file.name}", key=f"parse_{uploaded_file.name}"):
                with st.spinner("Extracting content with Docling..."):
                    chunks, manifest_path = run_pipeline(
                        pdf_path=pdf_file_path,
                        doc_id=doc_id,
                        out_dir=MANIFEST_DIR,
                    )
                    st.session_state[f"manifest_{doc_id}"] = manifest_path
                    st.success(f"Extracted {len(chunks)} chunks into {manifest_path.name}")

        with col2:
            if st.button(f"2. Index into Qdrant", key=f"seed_{uploaded_file.name}"):
                target_manifest = st.session_state.get(
                    f"manifest_{doc_id}", MANIFEST_DIR / f"{doc_id}_manifest.json"
                )
                if not Path(target_manifest).exists():
                    st.error("Please run PDF extraction first.")
                else:
                    with st.spinner("Indexing embeddings into local Qdrant..."):
                        indexed_count = seed_database(
                            manifest_path=target_manifest,
                            storage_path=qdrant_path,
                        )
                        st.success(f"Indexed {indexed_count} chunks into central vector store!")