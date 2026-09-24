import json
import re
from pathlib import Path


INPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\raw_pages.json")                                                                     
OUTPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\clean_pages.json")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_whitespace(text):
    """Normalize spaces while preserving line structure."""

    text = text.replace("\xa0", " ")

    # Normalize spaces/tabs inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Prevent excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
 
    return text.strip()


def normalize_bullets(text):
    """Normalize common bullet characters."""

    text = text.replace("•", "-")
    text = text.replace("▪", "-")
    text = text.replace("●", "-")

    return text


def remove_table_of_contents_artifact(text):
    """
    'Table of Contents' appears repeatedly in extracted pages
    because it is part of the PDF header/footer.

    Remove only the standalone header.
    """

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        if line.strip().lower() == "table of contents":
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_trailing_page_number(text):
    """
    Remove the standalone printed page number at the end.

    Example:

        Some document text...
        84

    The page number is already preserved as metadata.
    """

    lines = text.splitlines()

    # Remove empty lines at the end
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    # Only remove a standalone number at the END.
    if re.fullmatch(r"\d{1,3}", lines[-1].strip()):
        lines.pop()

    return "\n".join(lines)


def clean_page_text(text):
    """Apply safe cleaning operations."""

    text = normalize_bullets(text)

    text = remove_table_of_contents_artifact(text)

    text = remove_trailing_page_number(text)

    text = normalize_whitespace(text)

    return text


# ------------------------------------------------------------
# Document page number
# ------------------------------------------------------------

def detect_document_page(text):
    """
    Extract the printed document page number from the original
    text before it is removed.

    Example:
        PDF page 86 -> printed page 85
    """

    lines = text.splitlines()

    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return None

    last_line = lines[-1].strip()

    if re.fullmatch(r"\d{1,3}", last_line):
        return int(last_line)

    return None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    clean_pages = []

    for page in raw_pages:

        raw_text = page["text"]

        document_page = detect_document_page(raw_text)

        cleaned_text = clean_page_text(raw_text)

        clean_pages.append(
            {
                "pdf_page": page["page"],
                "document_page": document_page,
                "text": cleaned_text
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            clean_pages,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Cleaned {len(clean_pages)} pages")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()