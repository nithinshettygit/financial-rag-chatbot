import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ============================================================
# COMPACT ENTERPRISE DATA MODEL
#  creating a custom Data Structure using Python's @dataclass
# ============================================================
@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_type: str = "text"
    page_number: Optional[int] = None
    headings: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

# ============================================================
# OPTIMIZED ENTERPRISE CHUNKER ENGINE
# ============================================================
class OptimizedMarkdownChunker:
    def __init__(self, target_chars: int = 2500, overlap_chars: int = 300):
        # Step 1: Use a Header Splitter to split along our exact structural anchors(## Page Context Anchor: Page, ### Table:
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("## Page Context Anchor: Page", "page_number"),
                ("### Table:", "table_title"),
                ("###", "section_heading")
            ],
            strip_headers=False  # Keep headers inside the text so the LLM reads them
        )
        # Step 2: Use a Recursive Splitter to break down massive text blocks safely without splitting tables
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_file(self, input_path: Path, document_id: str, source_file: str) -> List[DocumentChunk]:
        markdown_text = input_path.read_text(encoding="utf-8")
        
        # Fast structural extraction via LangChain's compiled underlying implementation
        initial_sections = self.header_splitter.split_text(markdown_text)
        final_chunks: List[DocumentChunk] = []

        for section in initial_sections:
            meta = section.metadata
            content = section.page_content
            
            # Extract internal structural components
            page_num = int(meta["page_number"]) if "page_number" in meta else None
            
            # Reconstruct clean heading hierarchy array
            heading_path = []
            if "section_heading" in meta: heading_path.append(meta["section_heading"])
            if "table_title" in meta: heading_path.append(meta["table_title"])

            # DETERMINISTIC RULE: If it contains a markdown table structure, keep it completely ATOMIC
            if "|" in content and "---" in content:
                final_chunks.append(self._build_chunk_node(
                    content=content, doc_id=document_id, src_file=source_file,
                    chunk_type="table", page=page_num, headings=heading_path
                ))
            else:
                # Standard narrative prose gets safely broken down using standard slide windows
                sub_chunks = self.text_splitter.split_text(content)
                for chunk_text in sub_chunks:
                    final_chunks.append(self._build_chunk_node(
                        content=chunk_text, doc_id=document_id, src_file=source_file,
                        chunk_type="text", page=page_num, headings=heading_path
                    ))

        return self._assign_deterministic_ids(final_chunks)

    def _build_chunk_node(self, content: str, doc_id: str, src_file: str, chunk_type: str, page: Optional[int], headings: List[str]) -> DocumentChunk:
        """Injects clean meta contextual routing blocks straight into text entries."""
        context_header = f"Source: {src_file} | Location: Page {page or 'Unknown'}"
        if headings:
            context_header += f" > {' > '.join(headings)}"
        
        # Prepend contextual layout right into the string payload for embedding models
        enriched_content = f"Context: {context_header}\n\n{content.strip()}"
        
        return DocumentChunk(
            chunk_id="", document_id=doc_id, content=enriched_content,
            chunk_type=chunk_type, page_number=page, headings=headings, source_file=src_file,
            metadata={
                "chunk_type": chunk_type, "page_number": page, "section_hierarchy": headings, "source_file": src_file
            }
        )

    @staticmethod
    def _assign_deterministic_ids(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Calculates ultra-fast deterministic hash IDs to eliminate duplicate pipeline items."""
        for idx, chunk in enumerate(chunks):
            seed = f"{chunk.document_id}:{idx}:{chunk.content}"
            chunk.chunk_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return chunks

# ============================================================
# EXECUTION CONVENIENCE BLOCK
# ============================================================
# if __name__ == "__main__":
#     INPUT_MD = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\extracted_data.md")
    
#     chunker = OptimizedMarkdownChunker(target_chars=2500, overlap_chars=300)
#     chunks = chunker.chunk_file(
#         input_path=INPUT_MD,
#         document_id="duolingo_10k",
#         source_file="Form_10_K_for_Duolingo_INC.pdf"
#     )

#     print(f"\nProcessing Complete. Extracted {len(chunks)} Enterprise-Grade Payload Chunks.\n")
    
#     # Inspect a structural sample node
#     sample = chunks[0]
#     print("=" * 80 + f"\nCHUNK ID: {sample.chunk_id} ({sample.chunk_type.upper()})\n" + "-" * 80)
#     print(sample.content[:600] + "\n...")
import json

# ============================================================
# EXECUTION CONVENIENCE BLOCK
# ============================================================
if __name__ == "__main__":
    # Define your source and output paths
    INPUT_MD = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\extracted_data.md")
    OUTPUT_CHUNKS_JSON = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\chunks_manifest.json")
    
    # Run the optimized chunker engine
    chunker = OptimizedMarkdownChunker(target_chars=2500, overlap_chars=300)
    chunks = chunker.chunk_file(
        input_path=INPUT_MD,
        document_id="duolingo_10k",
        source_file="Form_10_K_for_Duolingo_INC.pdf"
    )

    print(f"\nProcessing Complete. Extracted {len(chunks)} Enterprise-Grade Payload Chunks.")
    
    # --------------------------------------------------------
    # NEW: CONVERT DATACLASS OBJECTS TO SERIABLE DICTS AND SAVE
    # --------------------------------------------------------
    serializable_chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "headings": chunk.headings,
            "source_file": chunk.source_file,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    # Ensure the parent folders exist and dump the payload manifest
    OUTPUT_CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Saved highly-optimized chunk manifest to:\n -> {OUTPUT_CHUNKS_JSON}\n")
    
    # Inspect a structural sample node
    if chunks:
        sample = chunks[0]
        print("=" * 80 + f"\nSAMPLE CHUNK ID: {sample.chunk_id} ({sample.chunk_type.upper()})\n" + "-" * 80)
        print(sample.content[:500] + "\n...")
