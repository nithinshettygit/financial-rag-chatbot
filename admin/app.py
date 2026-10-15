# admin/app.py
import os
from dotenv import load_dotenv
import logfire

# Load the .env file so os.environ can find LOGFIRE_TOKEN
load_dotenv() 

# Globally configure Logfire once for the entire application lifecycle
logfire.configure(
    service_name="financial-rag-engine",
    send_to_logfire=bool(os.environ.get("LOGFIRE_TOKEN"))
)
logfire.instrument_pydantic()

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import streamlit as st
from seed_qdrant import seed_database 

# Ensure the admin directory is in sys.path to guarantee imports resolve
ADMIN_DIR = Path(__file__).resolve().parent
if str(ADMIN_DIR) not in sys.path:
    sys.path.insert(0, str(ADMIN_DIR))

# Silence Windows 10054 connection reset warnings on socket termination
def silence_win_error_10054(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
        return
    loop.default_exception_handler(context)

if sys.platform == "win32":
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(silence_win_error_10054)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.set_exception_handler(silence_win_error_10054)
        except Exception:
            pass

from parse_pdf import run_pipeline
from seed_qdrant import seed_database

# Dynamic base directory configuration (relative to project root)
BASE_DIR = ADMIN_DIR.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
TEMP_DIR = DATA_DIR / "temp_uploads"
QDRANT_STORAGE_PATH = str(DATA_DIR / "qdrant_storage")

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def make_document_id(filename: str, content_hash: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_-")
    return f"{stem or 'document'}_{content_hash[:16]}"


st.set_page_config(
    page_title="Financial Document Ingestion Admin",
    layout="wide",
    page_icon="📊",
)

st.title("Multi-Document Financial RAG Ingestion Pipeline")
st.caption("Extract, structure, and index financial PDFs or pre-parsed manifests into Qdrant")

st.sidebar.header("Pipeline Configuration")
qdrant_path = st.sidebar.text_input("Qdrant Storage Path", value=QDRANT_STORAGE_PATH)
recreate_collection = st.sidebar.checkbox(
    "Recreate Collection on Indexing",
    value=False,
    help="If enabled, deletes the existing collection before re-indexing.",
)

# Use Tabs to cleanly separate PDF parsing vs. Manifest direct seeding
tab_pdf, tab_manifest = st.tabs(["📄 1. Upload & Parse PDF", "📂 2. Upload Existing Manifest"])

# ==========================================
# TAB 1: UPLOAD & PARSE PDF WORKFLOW
# ==========================================
with tab_pdf:
    st.subheader("End-to-End PDF Parsing & Indexing")
    st.markdown("Upload raw financial PDFs to extract tables, generate chunks, and seed Qdrant.")

    uploaded_files = st.file_uploader(
        "Upload Financial PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        for upload_index, uploaded_file in enumerate(uploaded_files):
            uploaded_bytes = uploaded_file.getvalue()
            content_hash = hashlib.sha256(uploaded_bytes).hexdigest()
            upload_key = f"{content_hash[:16]}_{upload_index}"
            default_doc_id = make_document_id(uploaded_file.name, content_hash)

            st.markdown(f"### 📄 {uploaded_file.name}")

            col_id, _ = st.columns([2, 1])
            with col_id:
                doc_id_input = st.text_input(
                    "Document Identifier",
                    value=default_doc_id,
                    key=f"doc_id_{upload_key}",
                )
            doc_id = re.sub(r"[^A-Za-z0-9_-]+", "_", doc_id_input).strip("_-") or default_doc_id

            # Write uploaded PDF file only if not already written
            pdf_file_path = TEMP_DIR / upload_key / Path(uploaded_file.name).name
            if not pdf_file_path.exists():
                pdf_file_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_file_path.write_bytes(uploaded_bytes)

            col1, col2 = st.columns(2)
            manifest_state_key = f"manifest_{upload_key}"

            with col1:
                if st.button("Parse Document via Docling", key=f"parse_{upload_key}"):
                    with logfire.span("ui_parse_pdf_action", filename=uploaded_file.name, doc_id=doc_id):
                        with st.spinner("Extracting financial tables and content..."):
                            chunks, manifest_path = run_pipeline(
                                pdf_path=pdf_file_path,
                                doc_id=doc_id,
                                out_dir=MANIFEST_DIR,
                            )
                            st.session_state[manifest_state_key] = {
                                "doc_id": doc_id,
                                "path": str(manifest_path),
                                "total_chunks": len(chunks),
                                "table_chunks": sum(1 for c in chunks if c.chunk_type == "table"),
                                "table_facts": sum(1 for c in chunks if c.chunk_type == "table_fact"),
                                "text_chunks": sum(1 for c in chunks if c.chunk_type == "text"),
                            }
                            st.success(f"Successfully parsed {len(chunks)} chunks!")

            with col2:
                if st.button("Index into Qdrant", key=f"seed_{upload_key}"):
                    manifest_info = st.session_state.get(manifest_state_key)
                    if not manifest_info or manifest_info["doc_id"] != doc_id:
                        st.error("Please parse this file first.")
                    elif not Path(manifest_info["path"]).exists():
                        st.error("Manifest file not found.")
                    else:
                        with logfire.span("ui_seed_qdrant_action", doc_id=doc_id):
                            with st.spinner("Embedding and indexing into Qdrant..."):
                                indexed_count = seed_database(
                                    manifest_path=manifest_info["path"],
                                    storage_path=qdrant_path,
                                    recreate_collection=recreate_collection,
                                )
                                logfire.info("UI Qdrant indexing completed", indexed_count=indexed_count, doc_id=doc_id)
                                st.success(f"Successfully indexed {indexed_count} vectors into Qdrant!")

            if manifest_state_key in st.session_state:
                info = st.session_state[manifest_state_key]
                if info.get("doc_id") == doc_id:
                    st.info(
                        f"**Parsed Summary:** {info['total_chunks']} total chunks "
                        f"({info['table_chunks']} table blocks, {info.get('table_facts', 0)} multi-year table facts, {info['text_chunks']} text blocks) "
                        f"| Saved to `{Path(info['path']).name}`"
                    )

            st.divider()

# ==========================================
# TAB 2: UPLOAD MANIFEST & SEED WORKFLOW
# ==========================================
with tab_manifest:
    st.subheader("Direct Seeding from Existing Manifest Files")
    st.markdown("Upload pre-parsed JSON manifest files (e.g., exported from Google Colab) to seed Qdrant instantly without local re-parsing.")

    manifest_file_upload = st.file_uploader(
        "Upload Manifest JSON File", type=["json"], key="manifest_uploader"
    )

    if manifest_file_upload:
        manifest_bytes = manifest_file_upload.getvalue()
        
        # Parse JSON preview to validate format and extract default doc_id
        try:
            manifest_chunks = json.loads(manifest_bytes.decode("utf-8"))
            first_chunk = manifest_chunks[0] if manifest_chunks else {}
            suggested_doc_id = first_chunk.get("document_id", Path(manifest_file_upload.name).stem.replace("_manifest", ""))
        except Exception as e:
            manifest_chunks = []
            suggested_doc_id = Path(manifest_file_upload.name).stem
            st.error(f"Invalid JSON file format: {e}")

        st.markdown(f"### 📂 {manifest_file_upload.name}")
        
        col_m_id, _ = st.columns([2, 1])
        with col_m_id:
            manifest_doc_id = st.text_input(
                "Target Document Identifier",
                value=suggested_doc_id,
                key="manifest_doc_id_input"
            )

        if manifest_chunks:
            table_fact_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "table_fact")
            table_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "table")
            text_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "text")
            st.info(
                f"**Manifest Preview:** {len(manifest_chunks)} total chunks "
                f"({table_count} table blocks, {table_fact_count} multi-year table facts, {text_count} text blocks)."
            )

            if st.button("Seed Qdrant from Uploaded Manifest", key="seed_from_manifest_btn"):
                # Save uploaded manifest locally to manifests dir for persistence
                saved_manifest_path = MANIFEST_DIR / manifest_file_upload.name
                saved_manifest_path.write_bytes(manifest_bytes)
                with logfire.span("ui_seed_from_uploaded_manifest", filename=manifest_file_upload.name):
                    with st.spinner("Indexing pre-parsed chunks into Qdrant vector database..."):
                        indexed_count = seed_database(
                            manifest_path=saved_manifest_path,
                            storage_path=qdrant_path,
                            recreate_collection=recreate_collection,
                        )
                        st.success(f"Successfully indexed {indexed_count} vectors from `{manifest_file_upload.name}` into Qdrant!")