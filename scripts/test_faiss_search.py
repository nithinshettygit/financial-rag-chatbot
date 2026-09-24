import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss.index")
METADATA_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\faiss\faiss_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(f"FAISS vectors: {index.ntotal}")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # Load same embedding model
    # --------------------------------------------------------

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query = input("\nEnter your question: ")

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FAISS SEARCH RESULTS")
    print("=" * 70)

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        item = metadata[index_id]

        print(f"\nRank       : {rank}")
        print(f"FAISS ID   : {index_id}")
        print(f"Score      : {score:.4f}")
        print(f"Chunk ID   : {item['chunk_id']}")
        print(f"PDF page   : {item.get('pdf_page')}")
        print(f"Section    : {item.get('section')}")

        print("\nText:")
        print(item["text"][:1000])


if __name__ == "__main__":
    main()