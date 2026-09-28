"""
Step 1: Data Pipeline — Parse knowledge base documents into chunks.

Each document is split by its natural headers (one plan/recharge = one chunk).
Metadata is attached: product_family, plan_name, category, language, source_file.
Output: data/chunks_en.json and data/chunks_ar.json

Usage:
    python step1_parse.py              # parse and save chunks
    python step1_parse.py --preview    # preview raw text (for tuning)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

import os
import re
import json
import sys
import docx        # python-docx
import pdfplumber


# ---------------------------------------------------------------------------
# Config: file-to-metadata mapping
# ---------------------------------------------------------------------------

# Maps each filename (without extension) to its product_family and category
FILE_META = {
    "PrepaidSystems": {
        "product_family": "Aqwa Card / Prepaid Systems",
        "category": "prepaid"
    },
    "DataLine": {
        "product_family": "DataLine",
        "category": "data"
    },
    "Emerald (2)": {
        "product_family": "Emerald",
        "category": "emerald"
    },
    "HekayaInternet": {
        "product_family": "Hekaya Internet",
        "category": "internet"
    },
    "HekayaMixat": {
        "product_family": "Hekaya Mixat",
        "category": "mix"
    },
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.join(BASE_DIR, "e& Knowledge base - with English version")
EN_DIR = os.path.join(KB_ROOT, "English")
AR_DIR = os.path.join(KB_ROOT, "e& Knowledge base")
OUT_DIR = os.path.join(BASE_DIR, "data")


# ---------------------------------------------------------------------------
# Header detection patterns
# ---------------------------------------------------------------------------

# English headers — lines that start a new plan/recharge section
# Examples: "Recharge: Aqwa Card 19", "Plan: DataLine 100", "Emerald 50"
EN_HEADER_PATTERNS = [
    r"^Recharge\s*:\s*.+",        # "Recharge: Aqwa Card 19"
    r"^Plan\s*:\s*.+",            # "Plan: DataLine 100"
    r"^Package\s*:\s*.+",         # "Package: ..."
    r"^Bundle\s*:\s*.+",          # "Bundle: ..."
    r"^\d+\s*[-\u2013]\s*.+",     # "19 - Aqwa Card" style
    r"^\d+\.\s+[A-Z].{3,80}",     # "14. Frequently Asked Questions...", "1. Overview..."
    r"^(Emerald|Data Line|Aqwa Card|Hekaya)\s*\d+",  # Specific plan names
]

# English subheaders — short section titles inside a plan chunk
# E.g. "Key Features:", "Terms and Conditions:", "FAQs:", "Add-ons:"
EN_SUBHEADER_PATTERNS = [
    r"^[A-Z][A-Za-z &/()-]{2,50}:\s*$",   # Title Case line ending with colon
    r"^[A-Z][A-Z &/()-]{4,50}$",           # ALL CAPS short line (section title)
    r"^\d+\.\s+[A-Z].{5,60}$",            # Numbered section like "1. Overview"
]

# Arabic headers — lines that start a new plan section
# The documents use the definite article ال prefix (e.g., الشحنة not شحنة)
AR_HEADER_PATTERNS = [
    r"^الشحنة\s*:\s*.+",          # "الشحنة:   أقوى كارت 5"
    r"^الباقة\s*:\s*.+",          # "الباقة: حكاية ميكسات 69"
    r"^شحنة\s*:\s*.+",           # "شحنة: ..." (without article)
    r"^باقة\s*:\s*.+",           # "باقة: ..."
    r"^باكدج\s*:\s*.+",          # "باكدج: ..."
    r"^نظام\s*:\s*.+",           # "نظام: ..."
    r"^خط\s*:\s*.+",             # "خط: ..."
]

# Arabic subheaders — short section titles inside an Arabic plan chunk
AR_SUBHEADER_PATTERNS = [
    r"^[\u0600-\u06FF\s]{3,40}:\s*$",     # Short Arabic line ending with colon
    r"^(المزايا|الشروط|الأسعار|الإضافات|الميزات|الاشتراك|التفاصيل|الأسئلة|ملاحظات|ملاحظة).*",
]

# Minimum body length (chars) for a chunk to be saved as its own entry.
# Subheader chunks shorter than this are merged into the previous chunk.
MIN_CHUNK_BODY = 80


def is_header_line(line: str, language: str) -> bool:
    """Check if a line matches any MAIN header pattern for the given language."""
    line = line.strip()
    if not line:
        return False

    patterns = EN_HEADER_PATTERNS if language == "en" else AR_HEADER_PATTERNS
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def is_subheader_line(line: str, language: str) -> bool:
    """Check if a line matches any SUBHEADER pattern for the given language."""
    line = line.strip()
    if not line or len(line) > 80:  # subheaders must be short
        return False

    patterns = EN_SUBHEADER_PATTERNS if language == "en" else AR_SUBHEADER_PATTERNS
    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def extract_plan_name(header_line: str) -> str:
    """Extract a clean plan name from the header line."""
    name = header_line.strip()
    # Remove numbering prefixes like "1. ", "2. " from Heading 1 sections
    name = re.sub(r"^\d+\.\s*", "", name)
    # Remove common prefixes like "Recharge:", "الشحنة:", etc.
    name = re.sub(
        r"^(Recharge|Plan|Package|Bundle|الشحنة|الباقة|شحنة|باقة|باكدج|نظام|خط)\s*:\s*",
        "", name, flags=re.IGNORECASE
    )
    name = name.strip()
    
    # If PDF parser merged a paragraph into the header, truncate it to just the plan name.
    # Strategy 1: Detect a repeated plan-name prefix (e.g. "Data Line 8 GB The Data Line 8 GB...")
    #   — find the first word boundary after a known short name pattern and cut there.
    # Strategy 2: Cut at common English prose verbs that signal a sentence has started.
    # Strategy 3: Cut at punctuation (period/comma) if present early.
    # Strategy 4: Fall back to first 60 chars.
    if len(name) > 60:
        # Strategy 1: repeated token — plan name keywords followed by a space+uppercase (new sentence)
        # e.g. "Data Line 8 GB The Data Line..." → cut before second "The"
        repeat_match = re.search(
            r'((?:Aqwa Card|Data Line|Hekaya|Emerald|Mix|Mixat)\s+[\d.]+\s*(?:GB|MB)?)\s+[A-Z]',
            name, re.IGNORECASE
        )
        if repeat_match:
            name = repeat_match.group(1).strip()
        else:
            # Strategy 2: prose verbs / articles that signal a sentence has started
            sentence_match = re.search(
                r'\b(is|are|was|were|with|granting|which|provides|offers|details|cost|does|has|the|and|of|for|in|at|to|a )\b',
                name[15:], re.IGNORECASE  # skip first 15 chars (part of the name itself)
            )
            if sentence_match:
                cut = 15 + sentence_match.start()
                name = name[:cut].strip().rstrip(',')
            else:
                # Strategy 3: cut at first period or comma
                punct_match = re.search(r'[.,]', name)
                if punct_match:
                    name = name[:punct_match.start()].strip()
                else:
                    # Strategy 4: hard truncate
                    name = name[:60].strip()

    return name


# ---------------------------------------------------------------------------
# Semantic Chunker Setup
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from langchain_text_splitters import SemanticChunker  # new home (langchain >= 0.3)
    except ImportError:
        from langchain_experimental.text_splitter import SemanticChunker  # legacy fallback

load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", AZURE_ENDPOINT)
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", AZURE_API_KEY)
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Initialize Azure OpenAI Embeddings for Semantic Chunking
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_key=AZURE_EMBEDDING_API_KEY,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_API_VERSION,
)

# Initialize Semantic Chunker
semantic_chunker = SemanticChunker(embeddings)


# ---------------------------------------------------------------------------
# PDF parsing (English)
# ---------------------------------------------------------------------------

def extract_pages_from_pdf(filepath: str) -> list[dict]:
    """Extract text per page from a PDF file using pdfplumber."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page_num": num, "text": text})
    return pages


def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    pages = extract_pages_from_pdf(filepath)
    return "\n".join(p["text"] for p in pages if p["text"])


def split_text_semantically(text: str, language: str) -> list[dict]:
    """
    Split text first by main headers, then apply Semantic Chunking on the body.
    """
    lines = text.split("\n")
    structural_chunks = []
    current_header = None
    current_body_lines = []

    for line in lines:
        stripped = line.strip()
        if is_header_line(stripped, language):
            if current_header is not None:
                structural_chunks.append({"header": current_header, "body": "\n".join(current_body_lines).strip()})
            current_header = stripped
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_header is not None:
        structural_chunks.append({"header": current_header, "body": "\n".join(current_body_lines).strip()})

    # Now apply semantic chunking on each structural chunk's body
    semantic_chunks = []
    for s_chunk in structural_chunks:
        body_text = s_chunk["body"]
        if not body_text:
            continue
            
        try:
            # Semantic chunking relies on embeddings to group semantically similar sentences
            sub_docs = semantic_chunker.create_documents([body_text])
            for sub_doc in sub_docs:
                if len(sub_doc.page_content.strip()) > 30: # ignore tiny fragments
                    semantic_chunks.append({
                        "header": s_chunk["header"],
                        "parent": s_chunk["header"],
                        "body": sub_doc.page_content.strip()
                    })
        except Exception as e:
            # Fallback to structural chunk if semantic fails
            semantic_chunks.append({"header": s_chunk["header"], "parent": s_chunk["header"], "body": body_text})

    return semantic_chunks


def parse_pdf(filepath: str) -> list[dict]:
    """Parse a PDF file into chunks with exact page coordinates and clean raw spans."""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    meta_info = FILE_META.get(filename, {})

    pages = extract_pages_from_pdf(filepath)
    full_text = "\n".join(p["text"] for p in pages if p["text"])
    raw_chunks = split_text_semantically(full_text, language="en")

    if not raw_chunks:
        print(f"  ⚠ No headers detected in {filename}.pdf — using full doc as one chunk")
        raw_chunks = [{"header": filename, "body": full_text.strip(), "parent": filename}]

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        full_text_chunk = f"{chunk['header']}\n{chunk['body']}"
        plan_name = extract_plan_name(chunk.get("parent") or chunk["header"])
        body_text = chunk.get("body", "").strip()

        # Coordinate detection: determine which PDF page this chunk belongs to
        page_num = 1
        char_start = 0
        char_end = len(body_text)
        raw_span = body_text

        # Search across pages using a clean probe from body_text
        clean_probe = " ".join(body_text.split())[:60]
        if clean_probe:
            for p in pages:
                p_text = p["text"]
                p_norm = " ".join(p_text.split())
                if clean_probe in p_norm:
                    page_num = p["page_num"]
                    # Try exact substring offset on this page
                    first_line = body_text.split("\n")[0].strip()[:40]
                    idx = p_text.find(first_line)
                    if idx >= 0:
                        char_start = idx
                        char_end = char_start + len(body_text)
                    break

        chunks.append({
            "text": full_text_chunk,
            "metadata": {
                "product_family": meta_info.get("product_family", filename),
                "plan_name": plan_name,
                "category": meta_info.get("category", "unknown"),
                "language": "en",
                "source_file": os.path.basename(filepath),
                "chunk_index": i,
                "page_num": page_num,
                "char_start": char_start,
                "char_end": char_end,
                "raw_span": raw_span,
                "para_idx": 0,
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# DOCX parsing (Arabic)
# ---------------------------------------------------------------------------

def extract_paragraphs_from_docx(filepath: str) -> list[dict]:
    """
    Extract paragraphs from a DOCX file, keeping track of style names and paragraph index.
    Returns: [{"para_idx": 0, "text": "...", "style": "Heading 1"}, ...]
    """
    doc = docx.Document(filepath)
    paragraphs = []
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            paragraphs.append({
                "para_idx": idx,
                "text": text,
                "style": para.style.name if para.style else "Normal",
            })
    return paragraphs


def is_heading_style(style_name: str) -> bool:
    """Check if a paragraph style is a heading."""
    return style_name.lower().startswith("heading")


def split_docx_semantically(paragraphs: list[dict], language: str) -> list[dict]:
    """
    Split DOCX paragraphs first by Heading 1, then apply Semantic Chunking on the body.
    """
    structural_chunks = []
    current_header = None
    current_body_lines = []

    for para in paragraphs:
        text = para["text"]
        style = para["style"].lower()

        is_h1 = ("heading 1" in style) or is_header_line(text, language)

        if is_h1:
            if current_header is not None:
                structural_chunks.append({"header": current_header, "body": "\n".join(current_body_lines).strip()})
            current_header = text
            current_body_lines = []
        else:
            current_body_lines.append(text)

    if current_header is not None:
        structural_chunks.append({"header": current_header, "body": "\n".join(current_body_lines).strip()})

    # Now apply semantic chunking on each structural chunk's body
    semantic_chunks = []
    for s_chunk in structural_chunks:
        body_text = s_chunk["body"]
        if not body_text:
            continue
            
        try:
            sub_docs = semantic_chunker.create_documents([body_text])
            for sub_doc in sub_docs:
                if len(sub_doc.page_content.strip()) > 30:
                    semantic_chunks.append({
                        "header": s_chunk["header"],
                        "parent": s_chunk["header"],
                        "body": sub_doc.page_content.strip()
                    })
        except Exception as e:
            semantic_chunks.append({"header": s_chunk["header"], "parent": s_chunk["header"], "body": body_text})

    return semantic_chunks


def parse_docx(filepath: str) -> list[dict]:
    """Parse a DOCX file into chunks with paragraph coordinates and clean raw spans."""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    meta_info = FILE_META.get(filename, {})

    paragraphs = extract_paragraphs_from_docx(filepath)
    raw_chunks = split_docx_semantically(paragraphs, language="ar")

    # Fallback: if no headers found, treat the whole doc as one chunk
    if not raw_chunks:
        print(f"  ⚠ No headers detected in {filename}.docx — using full doc as one chunk")
        all_text = "\n".join(p["text"] for p in paragraphs)
        raw_chunks = [{"header": filename, "body": all_text}]

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        full_text_chunk = f"{chunk['header']}\n{chunk['body']}"
        plan_name = extract_plan_name(chunk.get("parent") or chunk["header"])
        body_text = chunk.get("body", "").strip()

        para_idx = 0
        raw_span = body_text
        clean_probe = " ".join(body_text.split())[:50]
        if clean_probe:
            for p in paragraphs:
                p_norm = " ".join(p["text"].split())
                if clean_probe in p_norm:
                    para_idx = p["para_idx"]
                    break

        chunks.append({
            "text": full_text_chunk,
            "metadata": {
                "product_family": meta_info.get("product_family", filename),
                "plan_name": plan_name,
                "category": meta_info.get("category", "unknown"),
                "language": "ar",
                "source_file": os.path.basename(filepath),
                "chunk_index": i,
                "page_num": 1,
                "char_start": 0,
                "char_end": len(body_text),
                "raw_span": raw_span,
                "para_idx": para_idx,
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# Preview mode — print raw extracted text for tuning
# ---------------------------------------------------------------------------

def preview_raw_text():
    """Print raw extracted text from all files so you can see the structure."""
    print("=" * 70)
    print("PREVIEW MODE — showing raw text from each document")
    print("=" * 70)

    # English PDFs
    if os.path.isdir(EN_DIR):
        for fname in sorted(os.listdir(EN_DIR)):
            if fname.endswith(".pdf"):
                filepath = os.path.join(EN_DIR, fname)
                print(f"\n{'─' * 60}")
                print(f"📄 [EN] {fname}")
                print(f"{'─' * 60}")
                text = extract_text_from_pdf(filepath)
                # Print first 2000 chars to get a sense of structure
                print(text[:2000])
                if len(text) > 2000:
                    print(f"\n... ({len(text) - 2000} more characters)")

    # Arabic DOCX
    if os.path.isdir(AR_DIR):
        for fname in sorted(os.listdir(AR_DIR)):
            if fname.endswith(".docx"):
                filepath = os.path.join(AR_DIR, fname)
                print(f"\n{'─' * 60}")
                print(f"📄 [AR] {fname}")
                print(f"{'─' * 60}")
                paragraphs = extract_paragraphs_from_docx(filepath)
                for j, p in enumerate(paragraphs[:40]):
                    print(f"  [{j:3d}] style={p['style']:20s} | {p['text'][:100]}")
                if len(paragraphs) > 40:
                    print(f"  ... ({len(paragraphs) - 40} more paragraphs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for preview mode
    if "--preview" in sys.argv:
        preview_raw_text()
        return

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse English PDFs
    all_en_chunks = []
    print("📂 Parsing English PDFs...")
    if os.path.isdir(EN_DIR):
        for fname in sorted(os.listdir(EN_DIR)):
            if fname.endswith(".pdf"):
                filepath = os.path.join(EN_DIR, fname)
                chunks = parse_pdf(filepath)
                print(f"  ✅ {fname} → {len(chunks)} chunks")
                all_en_chunks.extend(chunks)
    else:
        print(f"  ❌ English directory not found: {EN_DIR}")

    # Parse Arabic DOCX
    all_ar_chunks = []
    print("\n📂 Parsing Arabic DOCX files...")
    if os.path.isdir(AR_DIR):
        for fname in sorted(os.listdir(AR_DIR)):
            if fname.endswith(".docx"):
                filepath = os.path.join(AR_DIR, fname)
                chunks = parse_docx(filepath)
                print(f"  ✅ {fname} → {len(chunks)} chunks")
                all_ar_chunks.extend(chunks)
    else:
        print(f"  ❌ Arabic directory not found: {AR_DIR}")

    # Save to JSON
    en_path = os.path.join(OUT_DIR, "chunks_en.json")
    ar_path = os.path.join(OUT_DIR, "chunks_ar.json")

    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(all_en_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved {len(all_en_chunks)} English chunks → {en_path}")

    with open(ar_path, "w", encoding="utf-8") as f:
        json.dump(all_ar_chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(all_ar_chunks)} Arabic chunks → {ar_path}")

    # Print a sample chunk for verification
    if all_en_chunks:
        print("\n📋 Sample English chunk:")
        sample = all_en_chunks[0]
        print(f"   Plan: {sample['metadata']['plan_name']}")
        print(f"   Family: {sample['metadata']['product_family']}")
        print(f"   Text preview: {sample['text'][:150]}...")

    if all_ar_chunks:
        print("\n📋 Sample Arabic chunk:")
        sample = all_ar_chunks[0]
        print(f"   Plan: {sample['metadata']['plan_name']}")
        print(f"   Family: {sample['metadata']['product_family']}")
        print(f"   Text preview: {sample['text'][:150]}...")

    print("\n✅ Step 1 complete! Check the data/ folder for the JSON files.")


if __name__ == "__main__":
    main()
