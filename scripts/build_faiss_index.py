import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\embeddings.npz")
CHUNKS_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\chunks.json")

FAISS_INDEX_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss.index")
FAISS_METADATA_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss_metadata.json")


def load_embeddings():
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]

    return embeddings.astype("float32")


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faiss_index():

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    print(f"Loaded embeddings: {embeddings.shape}")

    # FAISS expects:
    # (number_of_vectors, embedding_dimension)

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(dimension)

    # Add vectors
    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal}")

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    FAISS_INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    # --------------------------------------------------------
    # Save chunk metadata
    # --------------------------------------------------------

    chunks = load_chunks()

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: "
            f"{len(chunks)} chunks but "
            f"{embeddings.shape[0]} embeddings"
        )

    metadata = []

    for i, chunk in enumerate(chunks):

        metadata.append({
            "faiss_id": i,
            "chunk_id": chunk["chunk_id"],
            "source_type": chunk.get("source_type"),
            "pdf_page": chunk.get("pdf_page"),
            "document_page": chunk.get("document_page"),
            "section": chunk.get("section"),
            "table_id": chunk.get("table_id"),
            "table_title": chunk.get("table_title"),
            "years": chunk.get("years"),
            "text": chunk["text"]
        })

    with open(
        FAISS_METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFAISS index creation complete")
    print(f"Index    : {FAISS_INDEX_PATH}")
    print(f"Metadata : {FAISS_METADATA_PATH}")


if __name__ == "__main__":
    build_faiss_index()