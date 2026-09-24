import fitz
import json
from pathlib import Path


PDF_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\raw\Form_10_K_for_Duolingo_INC.pdf")
OUTPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\raw_pages.json")


def extract_pdf():
    doc = fitz.open(PDF_PATH)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    doc.close()

    print(f"Extracted {len(pages)} pages")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_pdf()