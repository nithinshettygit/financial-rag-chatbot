# admin/parse_pdf.py
import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 2048
MIN_ALNUM_CHARS = 10

_SEPARATOR_ROW = re.compile(r"^\|?[\s:\-|]+\|?$")


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    embedding_text: str
    chunk_type: str = "text"
    page_number: Optional[int] = None
    headings: List[str] = field(default_factory=list)
    table_caption: Optional[str] = None
    source_file: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def extract_table_structure(text: str) -> Dict[str, List[str]]:
    """Extracts column headers and non-numeric row labels from a markdown table."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("|") and not _SEPARATOR_ROW.match(line)
    ]
    if not lines:
        return {"columns": [], "row_labels": []}

    rows = []
    for line in lines:
        parts = line.split("|")
        # Standard markdown table lines have leading and trailing empty elements
        cells = [c.strip() for c in (parts[1:-1] if len(parts) > 2 else parts)]
        if any(cells):
            rows.append(cells)

    if not rows:
        return {"columns": [], "row_labels": []}

    columns = [c for c in rows[0] if c]

    # Collect row labels across cells (handles indented/nested row headers)
    row_labels = []
    for r in rows[1:]:
        row_label = None
        for cell in r:
            if cell and not re.match(r"^[\d\s.,$()%–\-+]+$", cell):
                row_label = cell
                break
        if row_label and row_label not in row_labels:
            row_labels.append(row_label)

    return {"columns": columns[:15], "row_labels": row_labels[:30]}


class FastDoclingChunker:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        doc_title: Optional[str] = None,
    ):
        self.doc_title = doc_title

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model,
            model_max_length=MAX_TOKENS,
            truncation=True,
        )
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=self.hf_tokenizer,
            max_tokens=MAX_TOKENS,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )
        pipeline_options.do_ocr = False

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=MAX_TOKENS,
            repeat_table_header=True,
            merge_peers=False,
        )
        self.converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _truncate(self, text: str, max_tokens: int = MAX_TOKENS) -> str:
        """Safely truncates embedding text strings exceeding model sequence limit."""
        ids = self.hf_tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=max_tokens
        )
        return self.hf_tokenizer.decode(ids)

    def process_pdf(
        self, pdf_path: Union[str, Path], doc_id: str
    ) -> Tuple[List[DocumentChunk], str]:
        pdf_path = Path(pdf_path)
        doc = self.converter.convert(source=pdf_path).document
        raw_chunks = list(self.chunker.chunk(dl_doc=doc))

        final_chunks: List[DocumentChunk] = []
        for idx, raw in enumerate(raw_chunks):
            text = (raw.text or "").strip()
            headings = list(getattr(raw.meta, "headings", []) or [])

            labels = [
                str(getattr(i, "label", "")).lower()
                for i in getattr(raw.meta, "doc_items", []) or []
                if getattr(i, "label", None)
            ]
            chunk_type = "table" if any("table" in l for l in labels) else "text"

            if not text:
                continue
            if chunk_type != "table" and len([c for c in text if c.isalnum()]) < MIN_ALNUM_CHARS:
                continue

            pages = sorted(
                {
                    int(prov.page_no)
                    for item in getattr(raw.meta, "doc_items", []) or []
                    for prov in getattr(item, "prov", []) or []
                    if getattr(prov, "page_no", None) is not None
                }
            )
            page_num = pages[0] if pages else None

            table_info = extract_table_structure(text) if chunk_type == "table" else {}

            emb_parts = []
            if self.doc_title:
                emb_parts.append(f"Document: {self.doc_title}")
            if page_num:
                emb_parts.append(f"Page: {page_num}")
            if headings:
                emb_parts.append(f"Section: {' > '.join(headings)}")
            if chunk_type == "table":
                if table_info.get("columns"):
                    emb_parts.append(f"Table Columns: {', '.join(table_info['columns'])}")
                if table_info.get("row_labels"):
                    emb_parts.append(f"Table Metrics: {'; '.join(table_info['row_labels'])}")

            emb_parts.append(text)
            full_embedding_text = "\n".join(emb_parts)
            embedding_text = self._truncate(full_embedding_text)

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
                        "columns": table_info.get("columns", []),
                        "row_labels": table_info.get("row_labels", []),
                        "is_financial_table": chunk_type == "table",
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