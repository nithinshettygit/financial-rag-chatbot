# parse_pdf.py
import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 400
EMBED_MAX_TOKENS = 480
MIN_ALNUM_CHARS = 30

_SEPARATOR_ROW = re.compile(r"^\|?[\s:\-|]+\|?$")
_NUMERIC_CELL = re.compile(r"^[\s$€£(\-–—+]*\d[\d,.\s]*[%)]?\s*[a-zA-Z]{0,2}$")


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    embedding_text: str
    chunk_type: str = "text"
    page_number: Optional[int] = None
    headings: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def profile_table(text: str) -> Dict:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("|") and not _SEPARATOR_ROW.match(line)
    ]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines if line]

    if not rows:
        return {"columns": [], "row_labels": [], "meaningful": False}

    columns = [c for c in rows[0] if c and not _NUMERIC_CELL.match(c)][:20]
    row_labels = [
        r[0] for r in rows[1:] if r and r[0] and not _NUMERIC_CELL.match(r[0])
    ][:40]

    return {
        "columns": columns,
        "row_labels": row_labels,
        "meaningful": len(rows) >= 2 and (len(columns) > 0 or len(row_labels) > 0),
    }


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


class FastDoclingChunker:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        doc_title: Optional[str] = None,
    ):
        self.doc_title = doc_title

        # Fix: Enforce token truncation on the tokenizer to avoid transformers warnings
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model,
            model_max_length=MAX_TOKENS,
            truncation=True,
        )
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=self.hf_tokenizer, max_tokens=MAX_TOKENS
        )

        # Force CPU execution for local processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )
        pipeline_options.do_ocr = False  # Disabled for faster processing of digital PDFs

        # Fix: Pass max_tokens explicitly to HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=MAX_TOKENS,
            repeat_table_header=True,
            merge_peers=True,
            serializer_provider=MarkdownTableSerializerProvider(),
        )
        self.converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _truncate(self, text: str, max_tokens: int = EMBED_MAX_TOKENS) -> str:
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            return self.hf_tokenizer.decode(ids[:max_tokens])
        return text

    def process_pdf(
        self, pdf_path: Union[str, Path], doc_id: str
    ) -> Tuple[List[DocumentChunk], str]:
        pdf_path = Path(pdf_path)
        doc = self.converter.convert(source=pdf_path).document
        raw_chunks = list(self.chunker.chunk(dl_doc=doc))

        final_chunks: List[DocumentChunk] = []
        for idx, raw in enumerate(raw_chunks):
            text = (raw.text or "").strip()
            headings = list(raw.meta.headings or [])

            labels = [
                str(getattr(i, "label", "")).lower()
                for i in getattr(raw.meta, "doc_items", []) or []
                if getattr(i, "label", None)
            ]
            chunk_type = "table" if any("table" in l for l in labels) else "text"

            if not text or len([c for c in text if c.isalnum()]) < MIN_ALNUM_CHARS:
                continue

            profile = profile_table(text) if chunk_type == "table" else None
            if chunk_type == "table" and not profile["meaningful"]:
                continue

            pages = sorted(
                {
                    int(prov.page_no)
                    for item in getattr(raw.meta, "doc_items", []) or []
                    for prov in getattr(item, "prov", []) or []
                    if getattr(prov, "page_no", None)
                }
            )
            page_num = pages[0] if pages else None

            emb_parts = []
            if self.doc_title:
                emb_parts.append(f"Doc: {self.doc_title}")
            if headings:
                emb_parts.append(f"Headings: {' > '.join(headings)}")
            if profile:
                if profile["columns"]:
                    emb_parts.append(f"Cols: {', '.join(profile['columns'])}")
                if profile["row_labels"]:
                    emb_parts.append(f"Rows: {'; '.join(profile['row_labels'])}")
            emb_parts.append(text)

            embedding_text = self._truncate("\n".join(emb_parts))
            chunk_id = hashlib.sha256(
                f"{doc_id}:{idx}:{text}".encode()
            ).hexdigest()[:16]

            final_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=text,
                    embedding_text=embedding_text,
                    chunk_type=chunk_type,
                    page_number=page_num,
                    headings=headings,
                    source_file=pdf_path.name,
                    metadata={
                        "page_numbers": pages,
                        "columns": profile["columns"] if profile else [],
                        "row_labels": profile["row_labels"] if profile else [],
                    },
                )
            )

        return final_chunks, doc.export_to_markdown()


def run_pipeline(
    pdf_path: Union[str, Path],
    doc_id: str,
    out_dir: Union[str, Path] = Path("data"),
) -> Tuple[List[DocumentChunk], Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunker = FastDoclingChunker(doc_title=doc_id)
    chunks, markdown_text = chunker.process_pdf(pdf_path, doc_id)

    (out_dir / f"{doc_id}_document.md").write_text(markdown_text, encoding="utf-8")
    manifest_path = out_dir / f"{doc_id}_manifest.json"
    serialized = [c.__dict__ for c in chunks]
    manifest_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return chunks, manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF using Docling chunker")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--doc-id", required=True, help="Document ID name")
    parser.add_argument("--out-dir", default="data", help="Output directory")

    args = parser.parse_args()
    chunks, manifest = run_pipeline(
        pdf_path=args.pdf, doc_id=args.doc_id, out_dir=args.out_dir
    )
    print(f"Successfully generated {len(chunks)} chunks in {manifest}")