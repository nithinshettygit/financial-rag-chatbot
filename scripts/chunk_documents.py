import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CLEAN_PAGES_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\clean_pages.json")
TABLES_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\tables.json")
OUTPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\chunks.json")


# ============================================================
# CHUNK SETTINGS
# ============================================================

TARGET_CHARS = 1000
OVERLAP_CHARS = 150
MINIMUM_CHUNK_LENGTH = 50


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# REMOVE KNOWN PDF LAYOUT ARTIFACTS
# ============================================================

def remove_layout_artifacts(text):
    """
    Remove known repeated PDF headers while preserving
    legitimate financial content.

    Do NOT use frequency-based removal here because
    values such as 2024, 2023, Total, etc. can legitimately
    repeat throughout financial tables.
    """

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Repeated company header
        if line == "DUOLINGO, INC.":
            continue

        # Repeated company header variant
        if line == "DUOLINGO, INC. AND SUBSIDIARIES":
            continue

        lines.append(line)

    return "\n".join(lines)


# ============================================================
# DETECT STANDALONE PDF ARTIFACTS
# ============================================================

def is_artifact(text):
    """
    Detect obvious standalone PDF extraction artifacts.
    """

    text = text.strip()

    # Empty text
    if not text:
        return True

    # Standalone numbering:
    # 1.
    # 9.
    # 12.
    if re.fullmatch(r"\d+\.", text):
        return True

    # Standalone bullets
    if re.fullmatch(r"[-•▪●]", text):
        return True

    return False


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(text):
    """
    Detect major 10-K sections using simple rules.
    No LLM required.
    """

    patterns = [
        r"^(ITEM\s+\d+[A-Z]?\.\s+.+)$",
        r"^(PART\s+[IVX]+)$",
    ]

    for line in text.splitlines():

        line = line.strip()

        for pattern in patterns:

            match = re.match(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:
                return match.group(1)

    return None


# ============================================================
# PARAGRAPH SPLITTING
# ============================================================

def split_into_paragraphs(text):
    """
    Split text while preserving paragraph boundaries.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ============================================================
# MERGE SMALL CHUNKS
# ============================================================

def merge_small_chunks(
    chunks,
    minimum_length=MINIMUM_CHUNK_LENGTH
):
    """
    Merge meaningful small chunks into the previous chunk.
    """

    if not chunks:
        return []

    merged = []

    for chunk in chunks:

        if (
            len(chunk) < minimum_length
            and merged
        ):
            merged[-1] += " " + chunk

        else:
            merged.append(chunk)

    return merged


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_text_chunks(page):
    """
    Create retrieval chunks from normal page text.
    """

    text = page["text"].strip()

    # --------------------------------------------------------
    # Remove known PDF layout artifacts
    # --------------------------------------------------------

    text = remove_layout_artifacts(text)

    if not text:
        return []

    # --------------------------------------------------------
    # Exclude actual Table of Contents page
    # --------------------------------------------------------

    if "Table of Contents Page" in text:
        return []

    # --------------------------------------------------------
    # Split into paragraphs
    # --------------------------------------------------------

    paragraphs = split_into_paragraphs(text)

    # Remove standalone artifacts
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not is_artifact(paragraph)
    ]

    if not paragraphs:
        return []

    # --------------------------------------------------------
    # Build chunks
    # --------------------------------------------------------

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # Large paragraph → split into sentences
        if len(paragraph) > TARGET_CHARS:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph
            )

        else:
            sentences = [paragraph]

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # First sentence
            if not current:

                current = sentence

                continue

            candidate = (
                current
                + " "
                + sentence
            )

            # Fits current chunk
            if len(candidate) <= TARGET_CHARS:

                current = candidate

            else:

                # Save current chunk
                chunks.append(current)

                # Create overlap
                overlap = current[-OVERLAP_CHARS:]

                current = (
                    overlap
                    + " "
                    + sentence
                )

    # --------------------------------------------------------
    # Save final chunk
    # --------------------------------------------------------

    if current:
        chunks.append(current)

    # --------------------------------------------------------
    # Merge small chunks
    # --------------------------------------------------------

    chunks = merge_small_chunks(chunks)

    return chunks


# ============================================================
# TABLE CHUNKING
# ============================================================

def create_table_chunks(table):
    """
    Convert structured financial table rows into
    retrieval-friendly chunks.
    """

    chunks = []

    title = table["title"]
    units = table.get("units")
    columns = table["columns"]

    current_rows = []

    for row in table["rows"]:

        current_rows.append(row)

        row_text = format_table_rows(
            title,
            units,
            columns,
            current_rows
        )

        # Current group reached target size
        if len(row_text) >= TARGET_CHARS:

            last_row = current_rows.pop()

            if current_rows:

                chunks.append(
                    format_table_rows(
                        title,
                        units,
                        columns,
                        current_rows
                    )
                )

            current_rows = [last_row]

    # Remaining rows
    if current_rows:

        chunks.append(
            format_table_rows(
                title,
                units,
                columns,
                current_rows
            )
        )

    return chunks


# ============================================================
# FORMAT TABLE
# ============================================================

def format_table_rows(
    title,
    units,
    columns,
    rows
):

    parts = [
        f"Table: {title}"
    ]

    if units:

        parts.append(
            f"Units: {units}"
        )

    parts.append(
        f"Columns: {', '.join(columns)}"
    )

    for row in rows:

        label = row["label"]

        values = []

        for column in columns:

            value = row.get(column)

            values.append(
                f"{column} = {value}"
            )

        parts.append(
            f"{label}: "
            + ", ".join(values)
        )

    return " | ".join(parts)


# ============================================================
# BUILD CHUNKS
# ============================================================

def build_chunks():

    pages = load_json(
        CLEAN_PAGES_PATH
    )

    tables = load_json(
        TABLES_PATH
    )

    chunks = []

    # ========================================================
    # TEXT CHUNKS
    # ========================================================

    text_counter = 1

    for page in pages:

        page_chunks = create_text_chunks(page)

        section = detect_section(
            page["text"]
        )

        for chunk_text in page_chunks:

            chunks.append({

                "chunk_id":
                    f"text_{text_counter:05d}",

                "source_type":
                    "text",

                "pdf_page":
                    page["pdf_page"],

                "document_page":
                    page.get("document_page"),

                "section":
                    section,

                "table_id":
                    None,

                "table_title":
                    None,

                "years":
                    [],

                "text":
                    chunk_text
            })

            text_counter += 1

    # ========================================================
    # TABLE CHUNKS
    # ========================================================

    table_counter = 1

    for table in tables:

        table_chunks = create_table_chunks(
            table
        )

        for chunk_text in table_chunks:

            chunks.append({

                "chunk_id":
                    f"table_{table_counter:05d}",

                "source_type":
                    "table",

                "pdf_page":
                    table["pdf_page"],

                "document_page":
                    table["document_page"],

                "section":
                    "Item 8. Financial Statements",

                "table_id":
                    table["table_id"],

                "table_title":
                    table["title"],

                "years":
                    table["columns"],

                "text":
                    chunk_text
            })

            table_counter += 1

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    text_chunks = sum(
        1
        for chunk in chunks
        if chunk["source_type"] == "text"
    )

    table_chunks = sum(
        1
        for chunk in chunks
        if chunk["source_type"] == "table"
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    print(
        f"Text chunks  : {text_chunks}"
    )

    print(
        f"Table chunks : {table_chunks}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_chunks()