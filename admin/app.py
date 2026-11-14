# admin/app.py
import hmac
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

import hashlib
import json
import re
import sys
from pathlib import Path

import streamlit as st
# Ensure the admin directory is in sys.path to guarantee imports resolve
ADMIN_DIR = Path(__file__).resolve().parent
if str(ADMIN_DIR) not in sys.path:
    sys.path.insert(0, str(ADMIN_DIR))

from parse_pdf import run_pipeline
from seed_qdrant import seed_database, validate_manifest_chunks

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


def require_admin_login() -> None:
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_username or not admin_password:
        st.error("Admin access is disabled. Configure ADMIN_USERNAME and ADMIN_PASSWORD.")
        st.stop()

    if st.session_state.get("admin_authenticated"):
        if st.sidebar.button("Log out", key="admin_logout"):
            del st.session_state["admin_authenticated"]
            st.rerun()
        return

    st.title("Admin sign in")
    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        username_matches = hmac.compare_digest(
            username.encode("utf-8"),
            admin_username.encode("utf-8"),
        )
        password_matches = hmac.compare_digest(
            password.encode("utf-8"),
            admin_password.encode("utf-8"),
        )
        if username_matches and password_matches:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        st.error("Invalid username or password.")
    st.stop()


require_admin_login()

st.title("Multi-Document Financial RAG Ingestion Pipeline")
st.caption("Extract, structure, and index financial PDFs or pre-parsed manifests into Qdrant")

st.sidebar.header("Pipeline Configuration")
qdrant_path = st.sidebar.text_input(
    "Qdrant Storage Path",
    value=QDRANT_STORAGE_PATH,
    help="Used for local Qdrant. QDRANT_URL in the environment takes precedence when configured.",
)
qdrant_url = os.environ.get("QDRANT_URL")
if qdrant_url:
    st.sidebar.caption("Using remote Qdrant configured by QDRANT_URL.")
else:
    st.sidebar.caption("Using local Qdrant storage.")

recreate_collection_requested = st.sidebar.checkbox(
    "Recreate Collection on Indexing",
    value=False,
    help="If enabled, deletes the existing collection before re-indexing.",
)
confirm_recreate_collection = False
if recreate_collection_requested:
    st.sidebar.warning(
        "Recreating the collection deletes every indexed document, not just the uploaded one."
    )
    confirm_recreate_collection = st.sidebar.checkbox(
        "Confirm deletion of the entire Qdrant collection"
    )
recreate_collection = recreate_collection_requested and confirm_recreate_collection

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
                    if recreate_collection_requested and not confirm_recreate_collection:
                        st.error("Confirm collection deletion in the sidebar before indexing.")
                    elif not manifest_info or manifest_info["doc_id"] != doc_id:
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
        manifest_error = None
        try:
            manifest_chunks = validate_manifest_chunks(
                json.loads(manifest_bytes.decode("utf-8"))
            )
            first_chunk = manifest_chunks[0] if manifest_chunks else {}
            suggested_doc_id = first_chunk.get("document_id", Path(manifest_file_upload.name).stem.replace("_manifest", ""))
        except (UnicodeDecodeError, ValueError) as e:
            manifest_chunks = []
            suggested_doc_id = Path(manifest_file_upload.name).stem
            manifest_error = f"Invalid manifest file: {e}"

        st.markdown(f"### 📂 {manifest_file_upload.name}")
        if manifest_error:
            st.error(manifest_error)
        
        col_m_id, _ = st.columns([2, 1])
        with col_m_id:
            manifest_doc_id = st.text_input(
                "Target Document Identifier",
                value=suggested_doc_id,
                key="manifest_doc_id_input"
            )

        normalized_manifest_doc_id = re.sub(
            r"[^A-Za-z0-9_-]+", "_", manifest_doc_id
        ).strip("_-")
        if manifest_chunks and not normalized_manifest_doc_id:
            st.error("Target Document Identifier must contain letters, numbers, '_' or '-'.")

        if manifest_chunks:
            table_fact_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "table_fact")
            table_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "table")
            text_count = sum(1 for c in manifest_chunks if c.get("chunk_type") == "text")
            st.info(
                f"**Manifest Preview:** {len(manifest_chunks)} total chunks "
                f"({table_count} table blocks, {table_fact_count} multi-year table facts, {text_count} text blocks)."
            )

            if st.button(
                "Seed Qdrant from Uploaded Manifest",
                key="seed_from_manifest_btn",
                disabled=not normalized_manifest_doc_id,
            ):
                # Save uploaded manifest locally to manifests dir for persistence
                if recreate_collection_requested and not confirm_recreate_collection:
                    st.error("Confirm collection deletion in the sidebar before indexing.")
                else:
                    chunks_to_seed = [
                        {**chunk, "document_id": normalized_manifest_doc_id}
                        for chunk in manifest_chunks
                    ]
                    saved_manifest_path = MANIFEST_DIR / Path(manifest_file_upload.name).name
                    saved_manifest_path.write_text(
                        json.dumps(chunks_to_seed, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    with logfire.span("ui_seed_from_uploaded_manifest", filename=manifest_file_upload.name):
                        with st.spinner("Indexing pre-parsed chunks into Qdrant vector database..."):
                            indexed_count = seed_database(
                                manifest_path=saved_manifest_path,
                                storage_path=qdrant_path,
                                recreate_collection=recreate_collection,
                            )
                            st.success(f"Successfully indexed {indexed_count} vectors from `{manifest_file_upload.name}` into Qdrant!")
