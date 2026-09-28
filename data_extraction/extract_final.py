import re
from pathlib import Path
import pymupdf

# ============================================================
# LAYOUT & TEXT CLEANING HELPERS• UNITS_PATTERN: Looks for typical financial phrases like "Amounts in thousands" or "In millions" so it can capture the scale of a table.
# • FOOTER_HEADER_PATTERN: Detects page numbers, item sections (like "Item 1A"), or headers (like "Form 10-K") so they can be ignored or stripped out.

# ============================================================
UNITS_PATTERN = re.compile(
    r"(Amounts?\s+in\s+\w+|Values?\s+in\s+\w+|In\s+thousands|In\s+millions|In\s+billions|In\s+USD|In\s+dollars|In\s+percent)",
    re.IGNORECASE
)

FOOTER_HEADER_PATTERN = re.compile(
    r"^(\d+|form\s+10-k|part\s+[i|v]+|item\s+\d+[a-z]?)$", 
    re.IGNORECASE
)

# • What it does: Extracts a title and financial unit for a detected table. • How it works: It looks at a 120-pixel visual zone directly above the top boundary (bbox_top) of a table. It extracts any text in that box, assumes the very last non-numeric line is the table's header/title, and runs the UNITS_PATTERN regex to find out if the table is in thousands, millions, or percent.

def infer_table_metadata(page, bbox_top, page_num, table_num):
    """Scans the text context immediately above the table bounding box."""
    search_rect = pymupdf.Rect(0, max(0, bbox_top - 120), page.rect.width, bbox_top)
    text = page.get_text("text", clip=search_rect)
    
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().isdigit()]
    title = lines[-1] if lines else f"Financial Table Data {table_num}"
    
    match = UNITS_PATTERN.search(text)
    units = match.group(0) if match else "Not specified"
    
    return title, units

# • What it does: Filters out "fake" tables (like paragraphs that just happen to look boxy).• How it works: It checks if a table has at least 2 rows and 2 columns. Then, it calculates a density ratio—the number of text-filled cells divided by total cells. If less than 25% of the grid has data, it rejects it as an empty or invalid table layout.

def is_valid_table(table, min_rows=2, min_columns=2):
    """Verifies that the structural object meets minimum matrix rules and data density."""
    if table.row_count < min_rows or table.col_count < min_columns:
        return False
    cells = [c.strip() for r in table.extract() for c in r if c]
    total_cells = table.row_count * table.col_count
    return (len(cells) / total_cells) >= 0.25 if total_cells > 0 else False

# • What it does: Cleans up text spacing and highlights uppercase sections.• How it works: It removes accidental duplicate tabs and spaces. It also checks if a text block is short, completely uppercase, and doesn't end with a period. If so, it assumes it is a section header and automatically formats it as a Markdown header (### Section Title).

def clean_and_format_text(text_block):
    """Normalizes structural formatting and removes common layout artifacts."""
    text = text_block.strip()
    if not text or FOOTER_HEADER_PATTERN.match(text):
        return ""
        
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    
    # Simple markdown formatting heuristic for major bold section boundaries
    if len(text) < 100 and text.isupper() and not text.endswith((".", "?", "!")):
        return f"\n### {text.title()}\n"
        
    return text

# ============================================================
# UNIFIED PIPELINE ENGINE (Callable from external files)
# cross-checks the visual boundaries of every text block against the boundaries of the known tables (table_bboxes). If a text block falls inside a table area, the script skips it. If it is outside, it cleans it and pairs it with its vertical position (by0 vertical coordinate) in a list called page_content_sequence.
# ============================================================
def extract_pdf_to_single_markdown(pdf_path: str | Path, output_md_path: str | Path):
    """
    Orchestrates table extraction and standard text layout generation page-by-page.
    Interleaves textual data blocks and structured tables natively in natural reading order
    while explicitly filtering layout overlap to safeguard against RAG duplicate content noise.
    """
    pdf_path = Path(pdf_path)
    output_md_path = Path(output_md_path)
    
    print(f"{'='*70}\nSTARTING UNIFIED EXTRACTION PIPELINE\n{'='*70}")
    print(f"Source PDF: {pdf_path}")
    
    md_document_lines = [
        f"# Unified Document Digest: {pdf_path.stem}",
        f"- **Primary File Source:** {pdf_path.name}",
        "---",
        ""
    ]
    
    total_tables = 0
    
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            print(f"Analyzing Canvas Structure - Page {page_num}/{len(doc)}...")
            
            # Step 1: Detect and index valid tables on this current page frame
            finder = page.find_tables()
            valid_tables_on_page = []
            
            for t_num, table_obj in enumerate(finder.tables, start=1):
                if is_valid_table(table_obj):
                    valid_tables_on_page.append((table_obj, t_num))
                    total_tables += 1
            
            # Extract basic bounding coordinates for visual tracking boundaries
            table_bboxes = [t.bbox for t, _ in valid_tables_on_page]
            
            # This collection array tracks items on the current page: (y_coordinate, markdown_content)
            page_content_sequence = []
            
            # Step 2: Extract Layout text blocks while removing table overlap regions
            blocks = page.get_text("blocks")  
            for b in blocks:
                bx0, by0, bx1, by1, text_content, _, _ = b
                
                # Cross-reference bounds: Verify if current block boundaries sit within a table element
                inside_table = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if (bx0 >= tx0 - 5 and by0 >= ty0 - 5 and bx1 <= tx1 + 5 and by1 <= ty1 + 5):
                        inside_table = True
                        break
                
                if inside_table:
                    continue
                    
                cleaned_text = clean_and_format_text(text_content)
                if cleaned_text:
                    # Capture the horizontal reading order sequence alongside visual boundary reference (by0)
                    page_content_sequence.append((by0, cleaned_text))
            
            # Step 3: Mix the native markdown tabular formats straight into the sequence array
            for table_obj, t_id in valid_tables_on_page:
                bbox_top = float(table_obj.bbox[1])  # Get the top vertical coordinate (y0) of the table
                title, units = infer_table_metadata(page, bbox_top, page_num, t_id)
                markdown_table = table_obj.to_markdown(clean=True, fill_empty=True)
                
                table_md_block = (
                    f"### Table: {title}\n"
                    f"- **Reference Identifier:** Table_P{page_num}_{t_id}\n"
                    f"- **Financial Scale Unit:** {units}\n\n"
                    f"{markdown_table}\n"
                )
                page_content_sequence.append((bbox_top, table_md_block))
            
            # Step 4: Physical layout sort - Sort items based on their Y-axis position on the document page
            # This ensures items are ordered top-to-bottom exactly as a human reads them
            page_content_sequence.sort(key=lambda x: x[0])
            
            # Step 5: Append organized page blocks to the primary string stack
            if page_content_sequence:
                md_document_lines.extend([
                    f"## Page Context Anchor: Page {page_num}",
                    ""
                ])
                for _, content in page_content_sequence:
                    md_document_lines.append(content)
                    
                md_document_lines.append("\n---\n")  # Explicit page chunking marker boundary
                
    # Persist layout payload directly to the unified file
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n\n".join(md_document_lines), encoding="utf-8")
    
    print(f"\n{'='*70}\nPIPELINE EXECUTION COMPLETE")
    print(f"File Saved: {output_md_path}")
    print(f"Total Structural Tables Consolidated: {total_tables}\n{'='*70}")

# ============================================================
# SELF-RUNNER RUNTIME TESTING HOOK
# ============================================================
if __name__ == "__main__":
    # Internal paths configuration
    INPUT_PDF = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\raw\Form_10_K_for_Duolingo_INC.pdf")
    OUTPUT_SINGLE_MD = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\extracted_data.md")
    
    # Internal module test invocation
    extract_pdf_to_single_markdown(INPUT_PDF, OUTPUT_SINGLE_MD)
