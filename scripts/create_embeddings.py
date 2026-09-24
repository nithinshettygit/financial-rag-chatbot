import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\chunks.json")
OUTPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\embeddings.npz")
# METADATA_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\embedding_metadata.json")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings():

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Save vectors
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embeddings
    )

    # --------------------------------------------------------
    # Save metadata separately
    # --------------------------------------------------------

    # metadata = []

    # for chunk in chunks:

    #     metadata.append({
    #         "chunk_id": chunk["chunk_id"],
    #         "source_type": chunk["source_type"],
    #         "pdf_page": chunk["pdf_page"],
    #         "document_page": chunk["document_page"],
    #         "section": chunk["section"],
    #         "table_id": chunk["table_id"],
    #         "table_title": chunk["table_title"],
    #         "years": chunk["years"],
    #         "text": chunk["text"]
    #     })

    # with open(
    #     METADATA_PATH,
    #     "w",
    #     encoding="utf-8"
    # ) as f:

    #     json.dump(
    #         metadata,
    #         f,
    #         ensure_ascii=False,
    #         indent=2
    #     )

    print("\nEmbedding stage complete")
    print(f"Vectors : {OUTPUT_PATH}")
    # print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    create_embeddings()