import json
import re
from pathlib import Path


INPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\extracted\raw_pages.json")
OUTPUT_PATH = Path(r"D:\xfcatr\srinivasan_venkataramanan_mentoring\project-financial-rag-chatbot\data\processed\tables.json")


# ----------------------------------- -------------------------
# Helpers
# ------------------------------------------------------------

def clean_text(text):
    """Normalize whitespace without changing financial content."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_number(value):
    """
    Convert financial table values into numbers.

    Examples:
        785,791     -> 785791
        (125,847)   -> 125847
        —           -> None
        2.04        -> 2.04
    """
    value = value.strip()

    if value in {"—", "-", "–", ""}:
        return None

    negative = value.startswith("(") and value.endswith(")")

    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.replace("(", "")
    value = value.replace(")", "")
    value = value.strip()

    try:
        number = float(value)

        if number.is_integer():
            number = int(number)

        return -number if negative else number

    except ValueError:
        return None


def load_pages():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Page 86 — Balance Sheet
# ------------------------------------------------------------

def parse_balance_sheet(page):
    text = clean_text(page["text"])

    table = {
        "table_id": "balance_sheet",
        "pdf_page": page["page"],
        "document_page": 85,
        "title": "Consolidated Balance Sheets",
        "units": "Amounts in thousands, except par value amounts",
        "columns": ["2024", "2023"],
        "rows": []
    }

    # Explicitly define the financial rows we expect.
    # This is safer than trying to blindly interpret every
    # number on the page.
    patterns = [
        ("Cash and cash equivalents", r"Cash and cash equivalents\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)"),
        ("Short-term investments", r"Short-term investments\s+([\d,]+)\s+—"),
        ("Accounts receivable", r"Accounts receivable\s+([\d,]+)\s+([\d,]+)"),
        ("Deferred cost of revenues", r"Deferred cost of revenues\s+([\d,]+)\s+([\d,]+)"),
        ("Prepaid expenses and other current assets",
         r"Prepaid expenses and other current assets\s+([\d,]+)\s+([\d,]+)"),
        ("Total current assets",
         r"Total current assets\s+([\d,]+)\s+([\d,]+)"),
        ("Operating lease right-of-use assets",
         r"Operating lease right-of-use assets\s+([\d,]+)\s+([\d,]+)"),
        ("Intangible assets, net",
         r"Intangible assets, net\s+([\d,]+)\s+([\d,]+)"),
        ("Long-term investments",
         r"Long-term investments\s+([\d,]+)\s+—"),
        ("Property and equipment, net",
         r"Property and equipment, net\s+([\d,]+)\s+([\d,]+)"),
        ("Goodwill",
         r"Goodwill\s+([\d,]+)\s+([\d,]+)"),
        ("Restricted cash",
         r"Restricted cash\s+([\d,]+)\s+([\d,]+)"),
        ("Deferred tax assets, net",
         r"Deferred tax assets, net\s+([\d,]+)\s+([\d,]+)"),
        ("Other assets",
         r"Other assets\s+([\d,]+)\s+([\d,]+)"),
        ("Total assets",
         r"Total assets\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)"),

        ("Deferred revenues",
         r"Deferred revenues\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)"),
        ("Accounts payable",
         r"Accounts payable\s+([\d,]+)\s+([\d,]+)"),
        ("Income tax payable",
         r"Income tax payable\s+([\d,]+)\s+([\d,]+)"),
        ("Accrued expenses and other current liabilities",
         r"Accrued expenses and other current liabilities\s+([\d,]+)\s+([\d,]+)"),
        ("Total current liabilities",
         r"Total current liabilities\s+([\d,]+)\s+([\d,]+)"),
        ("Long-term obligation under operating leases",
         r"Long-term obligation under operating leases\s+([\d,]+)\s+([\d,]+)"),
        ("Deferred tax liabilities, net",
         r"Deferred tax liabilities, net\s+([\d,]+)\s+—"),
        ("Total liabilities",
         r"Total liabilities\s+([\d,]+)\s+([\d,]+)"),
        ("Additional paid-in capital",
         r"Additional paid-in capital\s+([\d,]+)\s+([\d,]+)"),
        ("Accumulated deficit",
         r"Accumulated deficit\s+\(([\d,]+)\)\s+\(([\d,]+)\)"),
        ("Total stockholders' equity",
         r"Total stockholders[’'] equity\s+([\d,]+)\s+([\d,]+)"),
        ("Total liabilities and stockholders' equity",
         r"Total liabilities and stockholders[’'] equity\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)")
    ]

    for label, pattern in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        values = match.groups()

        row = {
            "label": label,
            "2024": parse_number(values[0]),
            "2023": parse_number(values[1]) if len(values) > 1 else None
        }

        # Special case for parenthesized accumulated deficit.
        if label == "Accumulated deficit":
            row["2024"] = -abs(row["2024"])
            row["2023"] = -abs(row["2023"])

        table["rows"].append(row)

    return table


# ------------------------------------------------------------
# Pages 87, 89, 90 — 3-year financial tables
# ------------------------------------------------------------

def parse_three_year_table(page, title, document_page, table_id):
    text = clean_text(page["text"])

    table = {
        "table_id": table_id,
        "pdf_page": page["page"],
        "document_page": document_page,
        "title": title,
        "units": "Amounts in thousands",
        "columns": ["2024", "2023", "2022"],
        "rows": []
    }

    # Matches:
    #
    # Revenue
    # 748,024
    # 531,109
    # 369,495
    #
    # after whitespace normalization.
    #
    # We use a known list of rows rather than attempting
    # to interpret arbitrary numbers.

    known_labels = {
        87: [
            "Revenues",
            "Cost of revenues",
            "Gross profit",
            "Research and development",
            "Sales and marketing",
            "General and administrative",
            "Total operating expenses",
            "Income (loss) from operations",
            "Other income",
            "Other expense",
            "Other (expense) income, net",
            "Income (loss) before interest income and income taxes",
            "Interest income",
            "Income (loss) before income taxes",
            "Provision for income taxes",
            "Net income (loss) and comprehensive income (loss)",
            "Net income (loss) per share attributable to Class A and Class B common stockholders, basic",
            "Net income (loss) per share attributable to Class A and Class B common stockholders, diluted",
        ],
        89: [
            "Net income (loss)",
            "Depreciation and amortization",
            "Stock-based compensation expense",
            "Accretion on marketable securities, net",
            "Gain on sale of capitalized software",
            "Loss on disposal of leasehold improvements",
            "Impairment of capitalized software",
            "Deferred revenue",
            "Accounts receivable",
            "Deferred cost of revenues",
            "Prepaid expenses and other current assets",
            "Accounts payable",
            "Accrued expenses and other current liabilities",
            "Noncurrent assets and liabilities",
            "Net cash provided by operating activities",
            "Purchases of investments",
            "Maturities of investments",
            "Capitalized software expense and purchases of intangible assets",
            "Purchase of property and equipment",
            "Proceeds from sale of capitalized software",
            "Acquisitions of companies, net of $5 cash acquired",
            "Net cash used for investing activities",
            "Proceeds from exercise of stock options",
            "Taxes paid related to net-share settlement of share-based compensation awards",
            "Net cash (used for) provided by financing activities",
            "Net increase in cash, cash equivalents and restricted cash",
            "Cash, cash equivalents and restricted cash - Beginning of period",
            "Cash, cash equivalents and restricted cash - End of period",
        ],
        90: [
            "Cash paid for income taxes",
            "Capitalized software and purchases of intangible assets included in Current liabilities",
            "Property and equipment included in Current liabilities",
            "Landlord incentive included in Prepaid expenses and other current assets",
            "Right of use assets obtained in exchange for new operating lease liabilities",
            "Right of use assets disposed or adjusted modifying operating leases liabilities",
        ]
    }

    labels = known_labels.get(page["page"], [])

    for label in labels:
        # Find the label and capture up to three financial values.
        pattern = re.escape(label) + r"\s+(?:\$\s*)?" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"(?:\$\s*)?" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"(?:\$\s*)?" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)"

        match = re.search(pattern, text)

        if not match:
            continue

        values = match.groups()

        row = {
            "label": label,
            "2024": parse_number(values[0]),
            "2023": parse_number(values[1]),
            "2022": parse_number(values[2])
        }

        table["rows"].append(row)

    return table


# ------------------------------------------------------------
# Page 88 — Stockholders' Equity
# ------------------------------------------------------------

def parse_stockholders_equity(page):
    table = {
        "table_id": "stockholders_equity",
        "pdf_page": page["page"],
        "document_page": 87,
        "title": "Consolidated Statements of Stockholders’ Equity",
        "units": "Amounts in thousands",
        "columns": [
            "Common Stock Shares",
            "Common Stock Amount",
            "Additional Paid-In Capital",
            "Accumulated Deficit",
            "Total"
        ],
        "rows": []
    }

    text = clean_text(page["text"])

    # Each balance/activity row has five values.
    labels = [
        "BALANCE—January 1, 2022",
        "Stock-based compensation expense",
        "Stock options exercised",
        "Release of restricted stock units",
        "Net loss",
        "BALANCE—December 31, 2022",
        "BALANCE—January 1, 2023",
        "Release of performance stock units",
        "Taxes paid related to net-share settlement of share-based compensation awards",
        "BALANCE—December 31, 2023",
        "BALANCE—January 1, 2024",
        "BALANCE—December 31, 2024",
        "Net income",
        "Release of restricted stock units",
    ]

    # Special handling:
    # the extracted values may contain "$" between columns.
    for label in labels:
        pattern = re.escape(label) + r"\s+" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"\$?\s*" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"\$?\s*" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"\$?\s*" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)\s+" \
                  r"\$?\s*" \
                  r"(\(?[\d,]+(?:\.\d+)?\)?|—)"

        match = re.search(pattern, text)

        if not match:
            continue

        values = match.groups()

        row = {
            "label": label,
            "common_stock_shares": parse_number(values[0]),
            "common_stock_amount": parse_number(values[1]),
            "additional_paid_in_capital": parse_number(values[2]),
            "accumulated_deficit": parse_number(values[3]),
            "total": parse_number(values[4])
        }

        table["rows"].append(row)

    return table


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    pages = load_pages()

    page_map = {page["page"]: page for page in pages}

    tables = []

    # Page 86
    if 86 in page_map:
        tables.append(parse_balance_sheet(page_map[86]))

    # Page 87
    if 87 in page_map:
        tables.append(
            parse_three_year_table(
                page_map[87],
                "Consolidated Statements of Operations and Comprehensive Income (Loss)",
                86,
                "income_statement"
            )
        )

    # Page 88
    if 88 in page_map:
        tables.append(parse_stockholders_equity(page_map[88]))

    # Page 89
    if 89 in page_map:
        tables.append(
            parse_three_year_table(
                page_map[89],
                "Consolidated Statements of Cash Flows",
                88,
                "cash_flow_statement"
            )
        )

    # Page 90
    if 90 in page_map:
        tables.append(
            parse_three_year_table(
                page_map[90],
                "Supplemental Disclosure of Cash Flow Information",
                89,
                "cash_flow_supplemental"
            )
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    print(f"Created: {OUTPUT_PATH}")

    for table in tables:
        print(
            f"{table['table_id']}: "
            f"{len(table['rows'])} rows"
        )


if __name__ == "__main__":
    main()