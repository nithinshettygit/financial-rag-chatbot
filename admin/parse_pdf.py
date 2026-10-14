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
_SERIALIZED_TABLE_CELL = re.compile(
    r"(?P<label>[^=]+?),\s*YEAR ENDED MAY 31,\.?\s*(?P<year>20\d{2})\s*=\s*"
    r"(?P<value>.*?)(?=(?:\.\s*)?[A-Z][^=]{0,120}?,\s*YEAR ENDED MAY 31,\.?\s*20\d{2}\s*=|$)",
    re.IGNORECASE,
)


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


def extract_serialized_table_rows(text: str) -> List[Dict[str, str]]:
    rows = []
    current_section = ""
    for match in _SERIALIZED_TABLE_CELL.finditer(text):
        label = match.group("label").strip(" .:;,\t\r\n")
        value = match.group("value").strip(" .\t\r\n")
        year = match.group("year")
        if not label:
            continue

        if label.endswith(":") or not value:
            current_section = label.rstrip(":").strip()
            continue

        normalized_label = label
        if label.casefold() in {"basic", "diluted"} and current_section:
            section_key = current_section.casefold()
            if "earnings per common share" in section_key:
                normalized_label = f"{label.capitalize()} earnings per share"
            elif "weighted average common shares outstanding" in section_key:
                normalized_label = f"{label.capitalize()} weighted average common shares outstanding"
        else:
            current_section = ""

        rows.append({"metric": normalized_label, "year": year, "value": value})
    return rows


def extract_table_structure(text: str) -> Dict[str, List[str]]:
    """Robustly extracts column headers and row labels from markdown or serialized table text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    columns = []
    row_labels = []

    # Check if table is in Markdown pipe format
    pipe_lines = [l for l in lines if l.startswith("|") and not _SEPARATOR_ROW.match(l)]
    if pipe_lines:
        for idx, line in enumerate(pipe_lines):
            parts = [c.strip() for c in line.split("|")[1:-1]]
            parts = [p for p in parts if p]
            if not parts:
                continue
            if idx == 0:
                columns = parts[:15]
            else:
                # First non-numeric cell is typically the row label/metric
                for cell in parts:
                    if cell and not re.match(r"^[\d\s.,$()%–\-+]+$", cell):
                        if cell not in row_labels:
                            row_labels.append(cell)
                        break
    else:
        rows = extract_serialized_table_rows(text)
        row_labels = list(dict.fromkeys(row["metric"] for row in rows))
        columns = list(dict.fromkeys(row["year"] for row in rows))

    return {"columns": columns[:15], "row_labels": row_labels[:40]}


def build_table_fact_chunks(
    rows: List[Dict[str, str]],
    doc_id: str,
    parent_chunk_id: str,
    page_number: Optional[int],
    headings: List[str],
    source_file: str,
) -> List[DocumentChunk]:
    facts = []
    section = " > ".join(headings)
    for row in rows:
        metric = row["metric"]
        year = row["year"]
        value = row["value"]
        content = (
            f"{doc_id}: {metric}, fiscal year ended May 31, {year} = {value}"
            f"{' per share' if 'earnings per share' in metric.casefold() else ''}."
        )
        embedding_parts = [f"Document: {doc_id}"]
        if page_number:
            embedding_parts.append(f"Page: {page_number}")
        if section:
            embedding_parts.append(f"Section: {section}")
        embedding_parts.extend(
            [f"Financial metric: {metric}", f"Fiscal year: {year}", content]
        )
        fact_id = hashlib.sha256(
            f"{parent_chunk_id}:{metric}:{year}:{value}".encode()
        ).hexdigest()[:16]
        facts.append(
            DocumentChunk(
                chunk_id=fact_id,
                document_id=doc_id,
                content=content,
                embedding_text="\n".join(embedding_parts),
                chunk_type="table_fact",
                page_number=page_number,
                headings=headings,
                source_file=source_file,
                metadata={
                    "page_numbers": [page_number] if page_number else [],
                    "columns": [year],
                    "row_labels": [metric],
                    "is_financial_table": True,
                    "is_financial_fact": True,
                    "financial_metric": metric,
                    "fiscal_year": int(year),
                    "financial_value": value,
                    "parent_chunk_id": parent_chunk_id,
                },
            )
        )
    return facts


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
            # Export raw item or text to markdown if it's a table
            text = (raw.text or "").strip()
            if hasattr(raw, "export_to_markdown"):
                try:
                    text = raw.export_to_markdown() or text
                except Exception:
                    pass

            headings = list(getattr(raw.meta, "headings", []) or [])

            labels = [
                str(getattr(i, "label", "")).lower()
                for i in getattr(raw.meta, "doc_items", []) or []
                if getattr(i, "label", None)
            ]
            chunk_type = "table" if any("table" in l for l in labels) or "|" in text or "=" in text else "text"

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

            table_info = extract_table_structure(text) if chunk_type == "table" else {"columns": [], "row_labels": []}

            emb_parts = []
            if self.doc_title:
                emb_parts.append(f"Document: {self.doc_title}")
            if page_num:
                emb_parts.append(f"Page: {page_num}")
            if headings:
                emb_parts.append(f"Section: {' > '.join(headings)}")
            if chunk_type == "table":
                if table_info.get("columns"):
                    emb_parts.append(f"Table Columns / Years: {', '.join(table_info['columns'])}")
                if table_info.get("row_labels"):
                    emb_parts.append(f"Table Metrics: {'; '.join(table_info['row_labels'])}")

            emb_parts.append(text)
            full_embedding_text = "\n".join(emb_parts)
            embedding_text = self._truncate(full_embedding_text)

            chunk_id = hashlib.sha256(
                f"{doc_id}:{idx}:{text}".encode()
            ).hexdigest()[:16]

            table_chunk = DocumentChunk(
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
            final_chunks.append(table_chunk)
            if chunk_type == "table":
                final_chunks.extend(
                    build_table_fact_chunks(
                        extract_serialized_table_rows(text),
                        doc_id=doc_id,
                        parent_chunk_id=chunk_id,
                        page_number=page_num,
                        headings=headings,
                        source_file=pdf_path.name,
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
    parser = argparse.ArgumentParser(description="Process PDF using robust Docling chunker")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--doc-id", required=True, help="Document ID name")
    parser.add_argument("--out-dir", default="data", help="Output directory")

    args = parser.parse_args()
    chunks, manifest = run_pipeline(
        pdf_path=args.pdf, doc_id=args.doc_id, out_dir=args.out_dir
    )
    print(f"Successfully generated {len(chunks)} chunks in {manifest}")