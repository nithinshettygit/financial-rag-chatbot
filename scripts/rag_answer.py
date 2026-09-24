from importlib.metadata import metadata
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss.index")
METADATA_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Start small.
TOP_K = 5
NEIGHBOR_PAGES = 1
MAX_CONTEXT_CHUNKS = 15

# Use a Groq model available in your account.
GROQ_MODEL = "openai/gpt-oss-20b" 
# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Add it to the .env file."
    )


# ============================================================
# LOAD FAISS
# ============================================================

def load_faiss_index():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    return faiss.read_index(
        str(INDEX_PATH)
    )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

# def expand_with_neighbor_pages(results, metadata, neighbor_pages=1):
#     """
#     Expand retrieved chunks with neighboring PDF pages.

#     Example:
#         Retrieved page 72
#         neighbor_pages=1

#         -> include pages 71, 72, 73

#     Only existing pages/chunks from metadata are added.
#     """

#     # Map PDF page -> all chunks belonging to that page
#     page_to_chunks = {}

#     for item in metadata:
#         page = item.get("pdf_page")

#         if page is None:
#             continue

#         try:
#             page = int(page)
#         except (ValueError, TypeError):
#             continue

#         page_to_chunks.setdefault(page, []).append(item)

#     # Keep track of chunks already added
#     seen_chunk_ids = set()

#     expanded = []

#     # Process the original FAISS results first
#     for result in results:

#         chunk_id = result.get("chunk_id")

#         if chunk_id and chunk_id not in seen_chunk_ids:
#             expanded.append(result)
#             seen_chunk_ids.add(chunk_id)

#         page = result.get("pdf_page")

#         if page is None:
#             continue

#         try:
#             page = int(page)
#         except (ValueError, TypeError):
#             continue

#         # Add neighboring pages
#         for offset in range(-neighbor_pages, neighbor_pages + 1):

#             neighbor_page = page + offset

#             if neighbor_page == page:
#                 continue

#             for chunk in page_to_chunks.get(neighbor_page, []):

#                 chunk_id = chunk.get("chunk_id")

#                 if not chunk_id or chunk_id in seen_chunk_ids:
#                     continue

#                 expanded.append({
#                     **chunk,
#                     "score": None,
#                     "expanded_from_page": page,
#                     "is_neighbor": True
#                 })

#                 seen_chunk_ids.add(chunk_id)

#     return expanded



# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=TOP_K
):

    # Create normalized query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        # Safety check
        if index_id < 0:
            continue

        item = metadata[index_id]

        results.append({
            "faiss_id": int(index_id),
            "score": float(score),
            "chunk_id": item["chunk_id"],
            "pdf_page": item.get("pdf_page"),
            "section": item.get("section"),
            "text": item["text"]
        })
    return results
    # expanded_results = expand_with_neighbor_pages(
    #     results,
    #     metadata,
    #     neighbor_pages=NEIGHBOR_PAGES
    # )

    # expanded_results = expanded_results[:MAX_CONTEXT_CHUNKS]
    # return expanded_results



# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        page = result.get("pdf_page", "Unknown")
        chunk_id = result.get("chunk_id", "Unknown")
        text = result.get("text", "")

        context_parts.append(
            f"""
    [Context {i}]
    PDF Page: {page}
    Chunk ID: {chunk_id}

    {text}
    """
        )

    context = "\n".join(context_parts)
    return context.strip()

# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
    client
):

    system_prompt = """
    You are a financial document question-answering assistant.

    Answer the user's question ONLY using the provided document context.

    Important rules:

    1. Use information from multiple context sections when necessary.
    2. Combine information across pages when it is required to answer the question.
    3. Do not invent numbers, dates, percentages, or financial facts.
    4. Preserve numerical values and their original units exactly.
    5. Do not convert thousands to millions or millions to thousands unless explicitly asked.
    6. If the supplied context does not contain enough information to answer the question, clearly say so.
    7. If information from different pages conflicts, mention the conflict.
    8. Do not generate citations or source references. The application will add sources separately.
    """.strip()

    user_prompt = f"""
Document context:

{context}

Question:

{question}

Answer the question using only the document context.
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        max_tokens=800
    )

    return response.choices[0].message.content


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading FAISS index...")

    index = load_faiss_index()

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print("Loading metadata...")

    metadata = load_metadata()

    print(
        f"Metadata records: {len(metadata)}"
    )

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    print("\nFinancial RAG chatbot ready.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        expanded_results = retrieve(
            question,
            model,
            index,
            metadata,
            TOP_K
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = build_context(
            expanded_results
        )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        print("\nGenerating answer...\n")

        answer = generate_answer(
            question,
            context,
            client
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        print("\n" + "=" * 70)
        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        for result in expanded_results:

            page = result.get("pdf_page", "Unknown")
            chunk_id = result.get("chunk_id", "Unknown")

            if result.get("is_neighbor"):
                print(
                    f"- Page {page} | Chunk ID: {chunk_id} | "
                    f"Neighbor context from Page {result.get('expanded_from_page')}"
                )
            else:
                print(
                    f"- Page {page} | Chunk ID: {chunk_id} | "
                    f"Score: {result.get('score', 0):.4f}"
                )

if __name__ == "__main__":
    main()