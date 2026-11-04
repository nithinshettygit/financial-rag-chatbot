# admin/parse_pdf.py
import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logfire
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Strict 500 token limit to guarantee safe bounds within BGE-small-en-v1.5 (512 token ceiling)
MAX_EMBEDDING_TOKENS = 500
MIN_TEXT_CHARS = 35
MIN_TEXT_WORDS = 6

_CELL_SPLIT = re.compile(
    r"\.(?!\d)\s+(?=[A-Za-z0-9][^=,]{0,100}?,\s*[^=]{1,80}?=)"
)
_YEAR_RE = re.compile(r"\b(19|20|21)\d{2}\b")
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
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_year(col_str: str) -> Optional[int]:
    match = _YEAR_RE.search(col_str)
    return int(match.group(0)) if match else None


def parse_cell_left(left: str) -> Tuple[str, str]:
    """Separates metric name from column header in serialized key-value table cells."""
    m = re.search(
        r",\s*((?:YEAR ENDED\s+(?:MAY|DECEMBER|JANUARY|JUNE|SEPTEMBER)\s+\d{1,2}|MAY\s+\d{1,2}|DECEMBER\s+\d{1,2}|FISCAL|FOR\s+THE\s+|AVERAGE\s+|AT\s+|AS\s+OF\s+)?[\s.,]*(?:20\d{2}|NUMBER|1|% CHANGE[^\s=]*|[A-Z0-9\s/–\-\.()]+?))$",
        left,
        re.IGNORECASE,
    )
    if m:
        metric = left[: m.start()].strip(" .:;\t\r\n")
        col = m.group(1).strip(" .:;\t\r\n")
        col = re.sub(r"^[.\s]+", "", col)
        col = col.replace(".20", " 20")
        return metric, col
    elif "," in left:
        metric, col = left.rsplit(",", 1)
        return metric.strip(" .:;\t\r\n"), col.strip(" .:;\t\r\n")
    return left.strip(), ""


def extract_structured_table(text: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Universal robust parser: handles both Markdown pipe tables and serialized table strings."""
    rows_by_metric: Dict[str, Dict[str, str]] = {}
    cols_order: List[str] = []

    # 1. Pipe-separated Markdown Table
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    pipe_lines = [l for l in lines if l.startswith("|") and not _SEPARATOR_ROW.match(l)]
    if pipe_lines:
        header_parts = [c.strip() for c in pipe_lines[0].split("|")[1:-1]]
        cols_order = [p for p in header_parts[1:] if p]
        for line in pipe_lines[1:]:
            parts = [c.strip() for c in line.split("|")[1:-1]]
            if not parts:
                continue
            metric = parts[0]
            if not metric or re.match(r"^[\d\s.,$()%–\-+]+$", metric):
                continue
            rows_by_metric[metric] = {}
            for col_idx, val in enumerate(parts[1:]):
                if col_idx < len(cols_order):
                    rows_by_metric[metric][cols_order[col_idx]] = val
        return cols_order, rows_by_metric

    # 2. Serialized format: Metric, Col = Val
    cells = _CELL_SPLIT.split(text)
    for c in cells:
        c = c.strip(" .\t\r\n")
        if "=" not in c:
            continue
        parts = c.split("=", 1)
        left = parts[0].strip()
        val = parts[1].strip(" .\t\r\n")
        if val.endswith(".") and not re.search(r"\d\.\d", val[-3:]):
            val = val[:-1].strip()

        metric, col = parse_cell_left(left)
        if not metric or len(metric) < 2 or metric.startswith("%") or metric.isdigit():
            continue
        if not col:
            continue

        if col not in cols_order:
            cols_order.append(col)

        if metric not in rows_by_metric:
            rows_by_metric[metric] = {}
        rows_by_metric[metric][col] = val

    return cols_order, rows_by_metric


def build_multiyear_fact_chunks(
    rows_by_metric: Dict[str, Dict[str, str]],
    doc_id: str,
    parent_chunk_id: str,
    page_number: Optional[int],
    headings: List[str],
    source_file: str,
) -> List[DocumentChunk]:
    """Generates atomic multi-year row facts preserving all columns/years together."""
    facts = []
    section = " > ".join(headings)

    for metric, row_data in rows_by_metric.items():
        non_empty = {k: v for k, v in row_data.items() if v and v.strip()}
        if not non_empty:
            continue

        years = [clean_year(c) for c in non_empty.keys() if clean_year(c)]
        years = sorted(set(years))

        # Consolidated Multi-Year Row Fact (Primary retrieval target)
        val_pairs = [f"{col} = {val}" for col, val in non_empty.items()]
        content = f"{doc_id}: {metric} | " + " | ".join(val_pairs) + "."

        emb_parts = [f"Document: {doc_id}"]
        if page_number:
            emb_parts.append(f"Page: {page_number}")
        if section:
            emb_parts.append(f"Section: {section}")
        emb_parts.append(f"Financial Metric: {metric}")
        if years:
            emb_parts.append(f"Fiscal Years: {', '.join(str(y) for y in years)}")
        emb_parts.append(content)

        fact_id = hashlib.sha256(
            f"{parent_chunk_id}:{metric}:multiyear".encode()
        ).hexdigest()[:16]

        facts.append(
            DocumentChunk(
                chunk_id=fact_id,
                document_id=doc_id,
                content=content,
                embedding_text="\n".join(emb_parts),
                chunk_type="table_fact",
                page_number=page_number,
                headings=headings,
                source_file=source_file,
                metadata={
                    "page_numbers": [page_number] if page_number else [],
                    "columns": list(non_empty.keys()),
                    "row_labels": [metric],
                    "is_financial_table": True,
                    "is_financial_fact": True,
                    "financial_metric": metric,
                    "fiscal_years": years,
                    "fiscal_year": max(years) if years else None,
                    "values_by_column": non_empty,
                    "parent_chunk_id": parent_chunk_id,
                },
            )
        )

    return facts


def split_large_table_content(text: str, max_words_per_block: int = 120) -> List[str]:
    """Splits large financial tables into bounded row blocks to prevent embedding truncation."""
    cells = _CELL_SPLIT.split(text)
    if len(cells) <= 12 or len(text.split()) <= max_words_per_block:
        return [text]

    blocks = []
    current_block: List[str] = []
    current_words = 0

    for c in cells:
        cell_str = c.strip(" .\t\r\n")
        if not cell_str:
            continue
        c_words = len(cell_str.split())
        if current_words + c_words > max_words_per_block and current_block:
            blocks.append(". ".join(current_block) + ".")
            current_block = [cell_str]
            current_words = c_words
        else:
            current_block.append(cell_str)
            current_words += c_words

    if current_block:
        blocks.append(". ".join(current_block) + ".")

    return blocks if blocks else [text]


class FastDoclingChunker:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        doc_title: Optional[str] = None,
    ):
        self.doc_title = doc_title

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model,
            model_max_length=MAX_EMBEDDING_TOKENS,
            truncation=True,
        )
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=self.hf_tokenizer,
            max_tokens=MAX_EMBEDDING_TOKENS,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )
        pipeline_options.do_ocr = False

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=MAX_EMBEDDING_TOKENS,
            repeat_table_header=True,
            merge_peers=False,
        )
        self.converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _truncate(self, text: str, max_tokens: int = MAX_EMBEDDING_TOKENS) -> str:
        """Safely truncates text within the embedding model's sequence length."""
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
            # If doc item has markdown export capability, use it
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

            # Filter out boilerplate / micro-chunk noise
            if chunk_type != "table":
                words = text.split()
                if len(text) < MIN_TEXT_CHARS or len(words) < MIN_TEXT_WORDS:
                    continue
                # Skip SEC header/boilerplate single lines
                if re.match(r"^\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}$", text.strip()):
                    continue
                if "IRS Employer Identification No" in text or "principal executive offices" in text:
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
            with logfire.span("process_pdf_page_chunk", chunk_index=idx, page_number=page_num, chunk_type=chunk_type):
                if chunk_type == "table":
                    cols, structured_rows = extract_structured_table(text)
                    row_labels = list(structured_rows.keys())
                    col_labels = cols[:15]

                    # Split large tables into bounded sub-blocks (< 500 tokens)
                    table_blocks = split_large_table_content(text)
                    for b_idx, block_text in enumerate(table_blocks):
                        chunk_id = hashlib.sha256(
                            f"{doc_id}:{idx}:block_{b_idx}:{block_text}".encode()
                        ).hexdigest()[:16]

                        emb_parts = []
                        if self.doc_title:
                            emb_parts.append(f"Document: {self.doc_title}")
                        if page_num:
                            emb_parts.append(f"Page: {page_num}")
                        if headings:
                            emb_parts.append(f"Section: {' > '.join(headings)}")
                        if col_labels:
                            emb_parts.append(f"Columns: {', '.join(col_labels)}")
                        if row_labels:
                            emb_parts.append(f"Metrics: {'; '.join(row_labels[:20])}")
                        emb_parts.append(block_text)

                        final_chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                content=block_text,
                                embedding_text=self._truncate("\n".join(emb_parts)),
                                chunk_type="table",
                                page_number=page_num,
                                headings=headings,
                                source_file=pdf_path.name,
                                metadata={
                                    "page_numbers": pages,
                                    "columns": col_labels,
                                    "row_labels": row_labels,
                                    "is_financial_table": True,
                                    "table_block_index": b_idx,
                                },
                            )
                        )

                    # Generate atomic multi-year row facts
                    facts = build_multiyear_fact_chunks(
                        rows_by_metric=structured_rows,
                        doc_id=doc_id,
                        parent_chunk_id=f"{doc_id}:{idx}",
                        page_number=page_num,
                        headings=headings,
                        source_file=pdf_path.name,
                    )
                    final_chunks.extend(facts)

                else:
                    chunk_id = hashlib.sha256(
                        f"{doc_id}:{idx}:{text}".encode()
                    ).hexdigest()[:16]

                    emb_parts = []
                    if self.doc_title:
                        emb_parts.append(f"Document: {self.doc_title}")
                    if page_num:
                        emb_parts.append(f"Page: {page_num}")
                    if headings:
                        emb_parts.append(f"Section: {' > '.join(headings)}")
                    emb_parts.append(text)

                    final_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            content=text,
                            embedding_text=self._truncate("\n".join(emb_parts)),
                            chunk_type="text",
                            page_number=page_num,
                            headings=headings,
                            source_file=pdf_path.name,
                            metadata={
                                "page_numbers": pages,
                                "columns": [],
                                "row_labels": [],
                                "is_financial_table": False,
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
    parser = argparse.ArgumentParser(description="Process PDF using robust Docling chunker")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--doc-id", required=True, help="Document ID name")
    parser.add_argument("--out-dir", default="data", help="Output directory")

    args = parser.parse_args()
    chunks, manifest = run_pipeline(
        pdf_path=args.pdf, doc_id=args.doc_id, out_dir=args.out_dir
    )
    print(f"Successfully generated {len(chunks)} chunks in {manifest}")