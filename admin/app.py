# admin/app.py
import asyncio
import hashlib
import re
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


def make_document_id(filename: str, content_hash: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_-")
    return f"{stem or 'document'}_{content_hash[:16]}"


st.set_page_config(page_title="Financial Document Ingestion Admin", layout="wide")
st.title("Multi-Document Financial RAG Ingestion Pipeline")

st.sidebar.header("Pipeline Configuration")
qdrant_path = st.sidebar.text_input("Qdrant Storage Path", value=QDRANT_STORAGE_PATH)

uploaded_files = st.file_uploader(
    "Upload Financial PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for upload_index, uploaded_file in enumerate(uploaded_files):
        uploaded_bytes = uploaded_file.getvalue()
        content_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        upload_key = f"{content_hash[:16]}_{upload_index}"
        default_doc_id = make_document_id(uploaded_file.name, content_hash)
        doc_id_input = st.text_input(
            "Document Identifier",
            value=default_doc_id,
            key=f"doc_id_{upload_key}",
        )
        doc_id = re.sub(r"[^A-Za-z0-9_-]+", "_", doc_id_input).strip("_-")
        if not doc_id:
            doc_id = default_doc_id

        temp_dir = DATA_DIR / "temp_uploads"
        pdf_file_path = temp_dir / upload_key / Path(uploaded_file.name).name
        pdf_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_bytes)

        st.success(f"Ready: {uploaded_file.name}")

        col1, col2 = st.columns(2)
        manifest_state_key = f"manifest_{upload_key}"

        with col1:
            if st.button(f"1. Parse {uploaded_file.name}", key=f"parse_{upload_key}"):
                with st.spinner("Extracting content with Docling..."):
                    chunks, manifest_path = run_pipeline(
                        pdf_path=pdf_file_path,
                        doc_id=doc_id,
                        out_dir=MANIFEST_DIR,
                    )
                    st.session_state[manifest_state_key] = {
                        "doc_id": doc_id,
                        "path": str(manifest_path),
                    }
                    st.success(f"Extracted {len(chunks)} chunks into {manifest_path.name}")

        with col2:
            if st.button("2. Index into Qdrant", key=f"seed_{upload_key}"):
                manifest_info = st.session_state.get(manifest_state_key)
                if not manifest_info or manifest_info["doc_id"] != doc_id:
                    st.error("Please parse this file with the current document identifier first.")
                elif not Path(manifest_info["path"]).exists():
                    st.error("Please run PDF extraction first.")
                else:
                    with st.spinner("Indexing embeddings into local Qdrant..."):
                        indexed_count = seed_database(
                            manifest_path=manifest_info["path"],
                            storage_path=qdrant_path,
                        )
                        st.success(f"Indexed {indexed_count} chunks into central vector store!")