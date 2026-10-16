# admin/parse_pdf.py
"""
Production PDF -> RAG chunk pipeline (Docling).

What changed vs. the previous version
-------------------------------------
1. Tables are read from Docling's *structured* TableItem (cells, spans, header
   flags) instead of regex-parsing flattened "Metric, Col = Val" text.
   -> no hardcoded month/year patterns; works for any table layout.
2. TableFormer ACCURATE mode + PDF cell matching (best table fidelity).
3. Nothing is silently truncated. Every embedding_text is packed to the token
   budget; anything bigger is split (loss-free), never cut.
4. Automatic OCR fallback for scanned pages (born-digital PDFs stay fast).
5. Quality report + lossless Docling JSON saved next to the manifest.

Public API (unchanged): DocumentChunk, FastDoclingChunker.process_pdf,
run_pipeline, and the CLI. Manifest schema is backward compatible; new
metadata keys are additive (row_group, table_caption, fact_part, ...).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import logfire
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE-small ceiling is 512 tokens; 500 leaves room for [CLS]/[SEP] + safety.
MAX_EMBEDDING_TOKENS = 500
MIN_TEXT_CHARS = 35  # below this (or MIN_TEXT_WORDS) a chunk is a 'fragment' -> merged, not dropped
MIN_TEXT_WORDS = 6
OCR_FALLBACK_MISSING_PAGE_RATIO = 0.20

_YEAR_RE = re.compile(r"\b(19|20|21)\d{2}\b")
# numbers, (negatives), percents, currency, and dash placeholders
_BARE_YEAR = re.compile(r"(19|20|21)\d{2}")
_NUMERIC_CELL = re.compile(r"[(\-−–—$€£¥\s]*[\d.,]+\s*[)%x]*|[\-−–—\s]+")
_SYMBOL_ONLY = re.compile(r"[$€£¥()%\s]+")  # '$', ')', '%', ')%', '(' ...
# Checkbox glyphs (Wingdings þ ¨ ☐ ...) and lone currency signs carry no content.
_GLYPH_ONLY = re.compile(r"[\s\u00fe\u00a8\u2610-\u2612\u2713\u2714\u25a0\u25a1$]+")
_BOILERPLATE = [
    re.compile(r"^\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}$"),
    re.compile(r"IRS Employer Identification No"),
    re.compile(r"principal executive offices"),
]


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
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


@dataclass
class TableRow:
    label: str
    values: Dict[str, str]
    group: Optional[str] = None
    is_group: bool = False
    unlabeled: bool = False


@dataclass
class ParsedTable:
    columns: List[str]
    rows: List[TableRow]
    unlabeled_rows: int = 0
    unlabeled_samples: List[Dict[str, Any]] = field(default_factory=list)


def clean_year(col_str: str) -> Optional[int]:
    match = _YEAR_RE.search(col_str)
    return int(match.group(0)) if match else None


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# Structured table parsing (replaces regex parsing of serialized text)
# --------------------------------------------------------------------------- #
def _cell_matrix(table: Any) -> Tuple[List[List[Any]], int, int]:
    """Expand Docling table cells (with row/col spans) into a dense grid."""
    data = table.data
    n_r, n_c = int(data.num_rows), int(data.num_cols)
    grid: List[List[Any]] = [[None] * n_c for _ in range(n_r)]
    for cell in data.table_cells:
        for r in range(cell.start_row_offset_idx, min(cell.end_row_offset_idx, n_r)):
            for c in range(cell.start_col_offset_idx, min(cell.end_col_offset_idx, n_c)):
                grid[r][c] = cell
    return grid, n_r, n_c


def _reading_order(pieces: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
    """Order cell fragments as a reader sees them (top-to-bottom, then left-to-right)
    using the cells' bounding boxes. Falls back to column order without geometry."""
    if len(pieces) < 2:
        return pieces
    ys: List[float] = []
    hs: List[float] = []
    for cell, _ in pieces:
        bb = getattr(cell, "bbox", None)
        if bb is None:
            return pieces
        origin = getattr(bb, "coord_origin", "")
        origin = str(getattr(origin, "value", origin)).upper()
        ys.append(-bb.t if "BOTTOM" in origin else bb.t)  # smaller = higher on the page
        hs.append(abs(bb.t - bb.b))
    tol = 0.5 * min(hs)

    def cmp(i: int, j: int) -> int:
        if ys[i] + tol < ys[j]:
            return -1
        if ys[j] + tol < ys[i]:
            return 1
        return i - j

    order = sorted(range(len(pieces)), key=functools.cmp_to_key(cmp))
    return [pieces[k] for k in order]


def parse_docling_table(table: Any) -> Optional[ParsedTable]:
    """
    Convert a Docling TableItem into (column labels, rows) from the model's own
    structure (header flags, row-header flags, spans, cell geometry). Handles:
      * multi-row / spanning headers          -> "Year Ended May 31, 2024"
      * '$' / ')' / '%' split into own columns (SEC filings)
      * wrapped row labels that the table model spread over several columns
        (header-less text columns are merged into the label, in reading order)
      * section rows ("Revenues:")            -> carried as `group`
      * rows with values but no label         -> kept (often subtotals), never dropped
    Returns None when the grid is unusable (caller falls back to markdown).
    """
    grid, n_r, n_c = _cell_matrix(table)
    if n_r < 1 or n_c < 2:
        return None

    def text_at(r: int, c: int) -> str:
        cell = grid[r][c]
        return _clean(cell.text) if cell is not None else ""

    def _single_col(cell: Any) -> bool:
        return cell is not None and (cell.end_col_offset_idx - cell.start_col_offset_idx) == 1

    # Header rows = contiguous rows from the top flagged as column headers.
    header_rows: List[int] = []
    for r in range(n_r):
        if any(c is not None and getattr(c, "column_header", False) for c in grid[r]):
            header_rows.append(r)
        else:
            break
    header_flagged = bool(header_rows)
    if not header_rows:
        # No flagged header. Row 0 is a header unless it carries real data
        # (numbers that are not bare years) -> then the table has no header row.
        first = [text_at(0, c) for c in range(n_c) if text_at(0, c)]
        numeric = [t for t in first if _NUMERIC_CELL.fullmatch(t)]
        has_data = any(not _BARE_YEAR.fullmatch(t) for t in numeric)
        header_rows = [] if has_data else [0]
    body_rows = list(range((header_rows[-1] + 1) if header_rows else 0, n_r))
    if not body_rows:
        return None

    def header_label(c: int) -> str:
        parts: List[str] = []
        seen = set()
        for r in header_rows:
            cell = grid[r][c]
            if cell is None or id(cell) in seen:
                continue
            seen.add(id(cell))
            t = _clean(cell.text)
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        return " ".join(parts).replace(" ,", ",")

    # Columns that only hold currency/percent/paren fragments get merged away.
    # Cells spanning several columns (section rows) are ignored for this test.
    sym_cols = set()
    for c in range(n_c):
        vals = {text_at(r, c) for r in body_rows if _single_col(grid[r][c])} - {""}
        if vals and all(_SYMBOL_ONLY.fullmatch(v) for v in vals):
            sym_cols.add(c)

    candidates = [c for c in range(n_c) if c not in sym_cols]
    flagged: List[int] = []
    for c in candidates:
        filled = [
            grid[r][c] for r in body_rows
            if grid[r][c] is not None and _clean(grid[r][c].text)
        ]
        if filled and sum(1 for x in filled if getattr(x, "row_header", False)) / len(filled) >= 0.5:
            flagged.append(c)
    label_cols = flagged or candidates[:1]
    value_cols = [c for c in candidates if c not in label_cols]

    # Header-less, mostly-text columns are wrapped label lines, not data.
    cont_cols: List[int] = []
    for c in value_cols:
        filled = [text_at(r, c) for r in body_rows if _single_col(grid[r][c]) and text_at(r, c)]
        if (
            filled
            and not header_label(c)
            and sum(1 for t in filled if not _NUMERIC_CELL.fullmatch(t)) / len(filled) >= 0.7
        ):
            cont_cols.append(c)
    value_cols = [c for c in value_cols if c not in cont_cols]
    label_like = sorted(set(label_cols) | set(cont_cols))
    if not value_cols:
        return None

    labels: List[str] = []
    counts: Dict[str, int] = {}
    for c in value_cols:
        lab = header_label(c) or f"Column {len(labels) + 1}"
        counts[lab] = counts.get(lab, 0) + 1
        labels.append(lab if counts[lab] == 1 else f"{lab} ({counts[lab]})")

    rows: List[TableRow] = []
    unlabeled = 0
    samples: List[Dict[str, Any]] = []
    group: Optional[str] = None
    prev_label: Optional[str] = None
    for r in body_rows:
        used = set()
        pieces: List[Tuple[Any, str]] = []
        for c in label_like:
            cell = grid[r][c]
            if cell is None or id(cell) in used:
                continue
            used.add(id(cell))
            t = _clean(cell.text)
            if t:
                pieces.append((cell, t))
        parts: List[str] = []
        for _, t in _reading_order(pieces):
            if not parts or parts[-1] != t:
                parts.append(t)
        label = " ".join(parts)

        row_vals: Dict[int, str] = {}
        for c in value_cols:
            cell = grid[r][c]
            if cell is None or id(cell) in used:  # spanning cell: keep once
                continue
            used.add(id(cell))
            t = re.sub(r"^([$€£¥])\s+", r"\1", _clean(cell.text))
            if t:
                row_vals[c] = t
        for c in sorted(sym_cols):
            if not _single_col(grid[r][c]):
                continue
            sym = text_at(r, c).replace(" ", "")
            if sym.startswith("("):  # opening paren column -> value on the right
                nxt = min((vc for vc in row_vals if vc > c), default=None)
                if nxt is not None and not row_vals[nxt].startswith("("):
                    row_vals[nxt] = "(" + row_vals[nxt]
            elif ")" in sym or "%" in sym:  # closing paren / percent -> value on the left
                sym = sym.lstrip("$€£¥")
                prev = max((vc for vc in row_vals if vc < c), default=None)
                if prev is not None:
                    row_vals[prev] += sym

        values = {labels[i]: row_vals[c] for i, c in enumerate(value_cols) if c in row_vals}
        if not values:
            if label:  # section / sub-header row
                group = label.rstrip(":").strip()
                rows.append(TableRow(label=group, values={}, is_group=True))
            continue
        if label:
            rows.append(TableRow(label=label, values=values, group=group))
            prev_label = label
        else:  # values without a label: keep them (subtotals etc.), flag them
            unlabeled += 1
            synth = f"(unlabeled row after {prev_label})" if prev_label else "(unlabeled row)"
            rows.append(TableRow(label=synth, values=values, group=group, unlabeled=True))
            if len(samples) < 5:
                samples.append({"after": prev_label, "values": dict(list(values.items())[:3])})

    if not any(not r.is_group for r in rows):
        return None
    if not header_flagged:
        # No header and (almost) no numbers = a layout table (signatures, cover
        # blocks, address lists). Row/column facts would be meaningless, so let the
        # caller keep it as plain markdown instead.
        vals = [v for r in rows if not r.is_group for v in r.values.values()]
        if vals and sum(1 for v in vals if _NUMERIC_CELL.fullmatch(v)) / len(vals) < 0.1:
            return None
    return ParsedTable(columns=labels, rows=rows, unlabeled_rows=unlabeled, unlabeled_samples=samples)


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|")


def _render_row(parsed: ParsedTable, row: TableRow) -> str:
    cells = [row.values.get(c, "") for c in parsed.columns]
    return "| " + " | ".join([_md_cell(row.label)] + [_md_cell(c) for c in cells]) + " |"


def _is_table_item(item: Any) -> bool:
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label)).lower() == "table"


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
class FastDoclingChunker:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        doc_title: Optional[str] = None,
        device: str = "auto",
        num_threads: int = 4,
        ocr_mode: str = "auto",  # "auto" | "on" | "off"
        hf_tokenizer: Any = None,
    ):
        self.doc_title = doc_title
        self.device = device
        self.num_threads = num_threads
        self.ocr_mode = ocr_mode
        self.last_doc: Any = None
        self.last_report: Dict[str, Any] = {}
        self._unlabeled_samples: List[Dict[str, Any]] = []
        self._fallback_pages: List[Optional[int]] = []
        self._converters: Dict[bool, DocumentConverter] = {}

        self.hf_tokenizer = hf_tokenizer or AutoTokenizer.from_pretrained(embedding_model)
        # We count tokens ourselves and never rely on silent truncation.
        self.hf_tokenizer.model_max_length = 10**6

        # Reserve room for the "Document/Page/Section" prefix we add on top of
        # the chunker's own (heading-contextualized) token count.
        overhead = self._count(f"Document: {doc_title or ''}\nPage: 99999\nSection: ") + 8
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=self.hf_tokenizer,
            max_tokens=MAX_EMBEDDING_TOKENS - overhead,
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=False)

    # ---- tokenization helpers ------------------------------------------- #
    def _count(self, text: str) -> int:
        return len(self.hf_tokenizer.encode(text, add_special_tokens=False))

    def _split_to_budget(self, text: str, budget: int) -> List[str]:
        """Loss-free split: lines -> sentences -> words -> raw token slices."""
        if self._count(text) <= budget:
            return [text]
        for pattern, sep in ((r"\n", "\n"), (r"(?<=[.!?;])\s+", " "), (r"\s+", " ")):
            parts = [p for p in re.split(pattern, text) if p.strip()]
            if len(parts) <= 1:
                continue
            out: List[str] = []
            cur: List[str] = []
            cur_n = 0
            for p in parts:
                n = self._count(p) + 1
                if n > budget:
                    if cur:
                        out.append(sep.join(cur))
                        cur, cur_n = [], 0
                    out.extend(self._split_to_budget(p, budget))
                    continue
                if cur and cur_n + n > budget:
                    out.append(sep.join(cur))
                    cur, cur_n = [], 0
                cur.append(p)
                cur_n += n
            if cur:
                out.append(sep.join(cur))
            return out
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return [
            self.hf_tokenizer.decode(ids[i : i + budget]) for i in range(0, len(ids), budget)
        ]

    def _prefix(
        self, page: Optional[int], headings: List[str], extra: Optional[List[str]] = None
    ) -> str:
        parts: List[str] = []
        if self.doc_title:
            parts.append(f"Document: {self.doc_title}")
        if page:
            parts.append(f"Page: {page}")
        if headings:
            parts.append(f"Section: {' > '.join(headings)}")
        parts.extend(extra or [])
        prefix = "\n".join(parts)
        cap = MAX_EMBEDDING_TOKENS // 3  # metadata may never starve the content
        while prefix and self._count(prefix) > cap and len(prefix.split()) > 1:
            prefix = prefix.rsplit(None, 1)[0]  # drop trailing words, keep original casing
        return prefix

    @staticmethod
    def _heading_map(doc: Any) -> Dict[str, List[str]]:
        """table self_ref -> active section headings, from document order."""
        out: Dict[str, List[str]] = {}
        try:
            try:
                from docling_core.types.doc import ContentLayer

                items = doc.iterate_items(included_content_layers=set(ContentLayer))
            except Exception:
                items = doc.iterate_items()
            by_level: Dict[int, str] = {}
            for item, _ in items:
                raw_label = getattr(item, "label", "")
                label = str(getattr(raw_label, "value", raw_label)).lower()
                if label in ("section_header", "title"):
                    level = 0 if label == "title" else int(getattr(item, "level", 1) or 1)
                    txt = _clean(getattr(item, "text", ""))
                    if txt:
                        by_level[level] = txt
                        for k in [k for k in by_level if k > level]:
                            del by_level[k]
                elif label == "table":
                    out[item.self_ref] = [by_level[k] for k in sorted(by_level)]
        except Exception as exc:
            logfire.warn("heading_map_failed {err}", err=str(exc))
        return out

    def _clip_tail(self, text: str, max_tokens: int) -> str:
        """Keep the END of a text (where a lead-in like '...following:' lives)."""
        words = text.split()[-120:]
        while len(words) > 1 and self._count(" ".join(words)) > max_tokens:
            words.pop(0)
        return " ".join(words)

    def _pieces(self, content: str, prefix: str) -> List[str]:
        budget = MAX_EMBEDDING_TOKENS - self._count(prefix) - 4
        return self._split_to_budget(content, max(budget, 64))

    # ---- converter ------------------------------------------------------- #
    def _converter(self, do_ocr: bool) -> DocumentConverter:
        if do_ocr not in self._converters:
            opts = PdfPipelineOptions()
            opts.accelerator_options = AcceleratorOptions(
                num_threads=self.num_threads,
                device=getattr(AcceleratorDevice, self.device.upper(), AcceleratorDevice.AUTO),
            )
            opts.do_ocr = do_ocr
            opts.do_table_structure = True
            opts.table_structure_options.mode = TableFormerMode.ACCURATE
            opts.table_structure_options.do_cell_matching = True
            self._converters[do_ocr] = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
            )
        return self._converters[do_ocr]

    @staticmethod
    def _pages_with_content(doc: Any) -> set:
        pages = {p.page_no for t in doc.texts for p in (t.prov or [])}
        pages |= {p.page_no for t in doc.tables for p in (t.prov or [])}
        return pages

    def _convert(self, pdf_path: Path) -> Tuple[Any, bool]:
        if self.ocr_mode == "on":
            return self._converter(True).convert(source=pdf_path).document, True
        doc = self._converter(False).convert(source=pdf_path).document
        if self.ocr_mode == "auto" and len(doc.pages):
            missing = len(doc.pages) - len(self._pages_with_content(doc))
            if missing / len(doc.pages) > OCR_FALLBACK_MISSING_PAGE_RATIO:
                try:
                    logfire.info("ocr_fallback pages_missing={missing}", missing=missing)
                    return self._converter(True).convert(source=pdf_path).document, True
                except Exception as exc:  # keep the non-OCR result
                    logfire.warn("ocr_fallback_failed {err}", err=str(exc))
        return doc, False

    # ---- table -> chunks -------------------------------------------------- #
    def _table_chunks(
        self,
        table: Any,
        doc: Any,
        doc_id: str,
        ref: str,
        headings: List[str],
        source_file: str,
        stats: Counter,
        lead_in: Optional[Tuple[Optional[int], str]] = None,
    ) -> List[DocumentChunk]:
        pages = sorted({int(p.page_no) for p in (table.prov or [])})
        page_num = pages[0] if pages else None
        try:
            caption = _clean(table.caption_text(doc)) or None
        except Exception:
            caption = None
        extra = [f"Table: {caption}"] if caption else []
        # Text directly above the table (units, super-headers, "...following:") is
        # its context, not a separate topic.
        if lead_in and lead_in[1] and (lead_in[0] is None or page_num is None or lead_in[0] == page_num):
            extra.append(f"Context: {self._clip_tail(lead_in[1], 48)}")
        prefix = self._prefix(page_num, headings, extra)
        parsed = parse_docling_table(table)
        out: List[DocumentChunk] = []
        stats["tables_total"] += 1

        base_meta = {
            "page_numbers": pages,
            "is_financial_table": True,
            "table_ref": ref,
        }

        # Fallback: structure unusable -> keep content as markdown, never drop it.
        if parsed is None:
            stats["tables_fallback_markdown"] += 1
            self._fallback_pages.append(page_num)
            try:
                md = table.export_to_markdown(doc=doc)
            except Exception:
                md = ""
            for n, piece in enumerate(self._pieces(md, prefix)) if md.strip() else []:
                out.append(
                    DocumentChunk(
                        chunk_id=hashlib.sha256(f"{doc_id}:{ref}:md:{n}:{piece}".encode()).hexdigest()[:16],
                        document_id=doc_id,
                        content=piece,
                        embedding_text=f"{prefix}\n{piece}" if prefix else piece,
                        chunk_type="table",
                        page_number=page_num,
                        headings=headings,
                        table_caption=caption,
                        source_file=source_file,
                        metadata={**base_meta, "columns": [], "row_labels": [], "structured": False},
                    )
                )
            return out

        stats["tables_structured"] += 1
        stats["table_rows_unlabeled"] += parsed.unlabeled_rows
        for smp in parsed.unlabeled_samples:
            self._unlabeled_samples.append({"page": page_num, **smp})
        data_rows = [r for r in parsed.rows if not r.is_group]
        stats["table_rows"] += len(data_rows)

        # ---- 1) table blocks: whole rows only, header repeated per block ---- #
        head_lines = [
            "| Metric | " + " | ".join(_md_cell(c) for c in parsed.columns) + " |",
            "|" + " --- |" * (len(parsed.columns) + 1),
        ]
        head_n = self._count("\n".join(head_lines))
        budget = MAX_EMBEDDING_TOKENS - self._count(prefix) - 8

        blocks: List[Tuple[List[str], List[str]]] = []  # (lines, row_labels)
        cur: List[str] = []
        cur_labels: List[str] = []
        cur_n = head_n
        for row in parsed.rows:
            line = _render_row(parsed, row) if not row.is_group else (
                f"| {_md_cell(row.label)} |" + " |" * len(parsed.columns)
            )
            n = self._count(line) + 1
            if cur and cur_n + n > budget:
                blocks.append((cur, cur_labels))
                cur, cur_labels, cur_n = [], [], head_n
            if not cur and row.group and not row.is_group:  # restore section context
                g_line = f"| {_md_cell(row.group)} |" + " |" * len(parsed.columns)
                cur.append(g_line)
                cur_n += self._count(g_line) + 1
            cur.append(line)
            cur_n += n
            if not row.is_group:
                cur_labels.append(row.label)
        if cur:
            blocks.append((cur, cur_labels))

        for b_idx, (lines, row_labels) in enumerate(blocks):
            block_text = "\n".join(head_lines + lines)
            for p_idx, piece in enumerate(self._pieces(block_text, prefix)):
                out.append(
                    DocumentChunk(
                        chunk_id=hashlib.sha256(
                            f"{doc_id}:{ref}:block_{b_idx}:{p_idx}:{piece}".encode()
                        ).hexdigest()[:16],
                        document_id=doc_id,
                        content=piece,
                        embedding_text=f"{prefix}\n{piece}" if prefix else piece,
                        chunk_type="table",
                        page_number=page_num,
                        headings=headings,
                        table_caption=caption,
                        source_file=source_file,
                        metadata={
                            **base_meta,
                            "columns": parsed.columns,
                            "row_labels": row_labels,
                            "table_block_index": b_idx,
                            "structured": True,
                        },
                    )
                )

        # ---- 2) atomic multi-year row facts (primary retrieval target) ------ #
        label_counts = Counter(r.label for r in data_rows)
        for r_idx, row in enumerate(data_rows):
            display = f"{row.group} > {row.label}" if (row.group and label_counts[row.label] > 1) else row.label
            pairs = [(col, val) for col, val in row.values.items()]
            years_all = sorted({y for col, _ in pairs if (y := clean_year(col))})

            extra_f = [f"Financial Metric: {display}"]
            if years_all:
                extra_f.append(f"Fiscal Years: {', '.join(str(y) for y in years_all)}")
            f_prefix = self._prefix(page_num, headings, extra_f + extra)
            lead = f"{doc_id}: {display} | "
            f_budget = MAX_EMBEDDING_TOKENS - self._count(f_prefix) - self._count(lead) - 10

            parts: List[List[Tuple[str, str]]] = []
            cur_p: List[Tuple[str, str]] = []
            cur_pn = 0
            for col, val in pairs:
                n = self._count(f"{col} = {val}") + 2
                if cur_p and cur_pn + n > max(f_budget, 32):
                    parts.append(cur_p)
                    cur_p, cur_pn = [], 0
                cur_p.append((col, val))
                cur_pn += n
            if cur_p:
                parts.append(cur_p)

            for part_idx, part in enumerate(parts):
                content = lead + " | ".join(f"{c} = {v}" for c, v in part) + "."
                part_years = sorted({y for c, _ in part if (y := clean_year(c))})
                out.append(
                    DocumentChunk(
                        chunk_id=hashlib.sha256(
                            f"{doc_id}:{ref}:row{r_idx}:{row.label}:p{part_idx}".encode()
                        ).hexdigest()[:16],
                        document_id=doc_id,
                        content=content,
                        embedding_text=f"{f_prefix}\n{content}",
                        chunk_type="table_fact",
                        page_number=page_num,
                        headings=headings,
                        table_caption=caption,
                        source_file=source_file,
                        metadata={
                            "page_numbers": pages,
                            "columns": [c for c, _ in part],
                            "row_labels": [row.label],
                            "is_financial_table": True,
                            "is_financial_fact": True,
                            "financial_metric": row.label,
                            "row_group": row.group,
                            "fiscal_years": part_years,
                            "fiscal_year": max(part_years) if part_years else None,
                            "values_by_column": dict(part),
                            "parent_chunk_id": f"{doc_id}:{ref}",
                            "unlabeled_row": row.unlabeled,
                            "fact_part": part_idx,
                            "fact_parts": len(parts),
                        },
                    )
                )
                stats["facts"] += 1
        return out

    # ---- main entry ------------------------------------------------------- #
    def process_pdf(
        self, pdf_path: Union[str, Path], doc_id: str
    ) -> Tuple[List[DocumentChunk], str]:
        pdf_path = Path(pdf_path)
        doc, ocr_used = self._convert(pdf_path)
        self.last_doc = doc

        tables_by_ref = {t.self_ref: t for t in doc.tables}
        seen_tables: set = set()
        stats: Counter = Counter()
        chunks: List[DocumentChunk] = []
        self._unlabeled_samples = []
        self._fallback_pages = []
        last_text: Dict[str, Any] = {"page": None, "text": None}
        frag: Dict[str, Any] = {"texts": [], "page": None, "headings": None, "pages": [], "idx": 0}

        def emit_text(text, page_num, headings, pages, idx, merged=0):
            if merged or len(text.split()) <= 60 or text.rstrip().endswith(":"):
                last_text.update(page=page_num, text=text)  # candidate lead-in for a table
            else:
                last_text.update(page=None, text=None)
            prefix = self._prefix(page_num, headings)
            for n, piece in enumerate(self._pieces(text, prefix)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=hashlib.sha256(f"{doc_id}:{idx}:{n}:{piece}".encode()).hexdigest()[:16],
                        document_id=doc_id,
                        content=piece,
                        embedding_text=f"{prefix}\n{piece}" if prefix else piece,
                        chunk_type="text",
                        page_number=page_num,
                        headings=headings,
                        source_file=pdf_path.name,
                        metadata={
                            "page_numbers": pages,
                            "columns": [],
                            "row_labels": [],
                            "is_financial_table": False,
                            "merged_fragments": merged,
                        },
                    )
                )
                stats["text_chunks"] += 1

        def flush_frag():
            if frag["texts"]:
                emit_text("\n".join(frag["texts"]), frag["page"], frag["headings"],
                          frag["pages"], frag["idx"], len(frag["texts"]))
                stats["text_fragments_merged"] += len(frag["texts"])
            frag.update(texts=[], page=None, headings=None, pages=[], idx=0)

        for idx, raw in enumerate(self.chunker.chunk(dl_doc=doc)):
            items = list(getattr(raw.meta, "doc_items", None) or [])
            headings = list(getattr(raw.meta, "headings", None) or [])

            with logfire.span("process_pdf_chunk", chunk_index=idx):
                # Table chunk(s): handled from the structured TableItem.
                if items and all(_is_table_item(i) for i in items):
                    flush_frag()
                    for item in items:
                        ref = getattr(item, "self_ref", None)
                        if ref in seen_tables:
                            continue
                        seen_tables.add(ref)
                        table = tables_by_ref.get(ref)
                        if table is not None:
                            lead = (last_text["page"], last_text["text"]) if last_text["text"] else None
                            chunks.extend(
                                self._table_chunks(
                                    table, doc, doc_id, ref, headings, pdf_path.name, stats, lead
                                )
                            )
                            last_text.update(page=None, text=None)
                    continue

                text = (raw.text or "").strip()
                if not text:
                    continue
                if _GLYPH_ONLY.fullmatch(text):
                    stats["text_dropped_glyph"] += 1
                    continue
                if any(p.search(text) for p in _BOILERPLATE):
                    stats["text_dropped_boilerplate"] += 1
                    continue

                pages = sorted(
                    {
                        int(prov.page_no)
                        for item in items
                        for prov in (getattr(item, "prov", None) or [])
                        if getattr(prov, "page_no", None) is not None
                    }
                )
                page_num = pages[0] if pages else None

                if len(text) < MIN_TEXT_CHARS or len(text.split()) < MIN_TEXT_WORDS:
                    # Short fragment: never dropped, merged with its neighbours
                    # (same page + same section) into one chunk.
                    if frag["texts"] and (frag["page"] != page_num or frag["headings"] != headings):
                        flush_frag()
                    if not frag["texts"]:
                        frag.update(page=page_num, headings=headings, pages=pages, idx=idx)
                    frag["texts"].append(text)
                    if self._count("\n".join(frag["texts"])) > MAX_EMBEDDING_TOKENS // 2:
                        flush_frag()
                    continue

                flush_frag()
                emit_text(text, page_num, headings, pages, idx)
        flush_frag()

        # Safety net: any table the chunker did not surface is still ingested.
        missed_pages: List[Optional[int]] = []
        heading_map: Optional[Dict[str, List[str]]] = None
        for ref, table in tables_by_ref.items():
            if ref not in seen_tables:
                stats["tables_missed_by_chunker"] += 1
                if heading_map is None:
                    heading_map = self._heading_map(doc)
                pg = next((int(p.page_no) for p in (table.prov or [])), None)
                missed_pages.append(pg)
                chunks.extend(
                    self._table_chunks(
                        table, doc, doc_id, ref, heading_map.get(ref, []), pdf_path.name, stats
                    )
                )

        # Quality gate: nothing may exceed the embedding window.
        over = [c.chunk_id for c in chunks if self._count(c.embedding_text) > MAX_EMBEDDING_TOKENS]
        if over:
            logfire.error("chunks_over_token_budget {n}", n=len(over))
        pages_total = len(doc.pages)
        empty_pages = sorted(set(doc.pages.keys()) - self._pages_with_content(doc)) if pages_total else []
        self.last_report = {
            "document_id": doc_id,
            "source_file": pdf_path.name,
            "pages": pages_total,
            "pages_without_content": empty_pages,
            "ocr_used": ocr_used,
            "chunks_total": len(chunks),
            "chunks_by_type": dict(Counter(c.chunk_type for c in chunks)),
            "max_embedding_tokens": max((self._count(c.embedding_text) for c in chunks), default=0),
            "token_budget": MAX_EMBEDDING_TOKENS,
            "chunks_over_budget": over,
            "tables_fallback_pages": self._fallback_pages,
            "tables_missed_pages": missed_pages,
            "unlabeled_row_samples": self._unlabeled_samples[:20],
            **{k: v for k, v in sorted(stats.items())},
        }
        return chunks, doc.export_to_markdown()


# --------------------------------------------------------------------------- #
# Pipeline / CLI
# --------------------------------------------------------------------------- #
def run_pipeline(
    pdf_path: Union[str, Path],
    doc_id: str,
    out_dir: Union[str, Path] = Path("data"),
    device: str = "auto",
    ocr_mode: str = "auto",
) -> Tuple[List[DocumentChunk], Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunker = FastDoclingChunker(doc_title=doc_id, device=device, ocr_mode=ocr_mode)
    chunks, markdown_text = chunker.process_pdf(pdf_path, doc_id)

    # (out_dir / f"{doc_id}_document.md").write_text(markdown_text, encoding="utf-8")
    manifest_path = out_dir / f"{doc_id}_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"{doc_id}_quality.json").write_text(
        json.dumps(chunker.last_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # try:  # lossless Docling doc: re-chunk later without re-parsing the PDF
    #     chunker.last_doc.save_as_json(out_dir / f"{doc_id}_docling.json")
    # except Exception as exc:
    #     logfire.warn("docling_json_save_failed {err}", err=str(exc))
    # return chunks, manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF using the Docling structured chunker")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--doc-id", required=True, help="Document ID name")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--ocr", default="auto", choices=["auto", "on", "off"])

    args = parser.parse_args()
    chunks, manifest = run_pipeline(
        pdf_path=args.pdf,
        doc_id=args.doc_id,
        out_dir=args.out_dir,
        device=args.device,
        ocr_mode=args.ocr,
    )
    print(f"Successfully generated {len(chunks)} chunks in {manifest}")