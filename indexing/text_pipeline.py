"""
indexing/text_pipeline.py

Extracts text from a PDF/doc and splits it into chunks. Phase 1 scope:
plain text extraction + simple fixed-size chunking — no embeddings yet
(that's Phase 2, embedder.py).
"""

from pathlib import Path
from pypdf import PdfReader

import re


def clean_extracted_text(text):
    """
    Collapse pypdf's occasional one-word-per-line extraction artifact
    (seen on some PDF layouts) back into normal spacing, without
    damaging text that already extracted cleanly. Multiple consecutive
    newlines get collapsed to a single space; a genuine paragraph break
    (blank line followed by a clear new sentence) is a real loss here,
    but this project prioritizes clean embeddings over perfect paragraph
    structure, since search quality depends on the former.
    """
    # Collapse 2+ consecutive newlines/whitespace into a single space
    cleaned = re.sub(r'\s*\n\s*', ' ', text)
    # Collapse any resulting multiple spaces into one
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()

def extract_text_from_txt(file_path):
    """Read a plain text file directly."""
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()
def extract_text_from_docx(file_path):
    """Extract all paragraph text from a Word document."""
    from docx import Document
    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_csv(file_path):
    """
    Read a CSV file and turn it into readable text -- each row rendered
    as 'column: value, column: value, ...' so a search query about the
    data's content has meaningful text to match against, rather than raw
    comma-separated values.
    """
    import csv
    rows_text = []
    with open(file_path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_text = ", ".join(f"{key}: {value}" for key, value in row.items())
            rows_text.append(row_text)
    return "\n".join(rows_text)


def extract_text_from_md(file_path):
    """Read a Markdown file as plain text (markdown syntax doesn't need
    special stripping for embedding purposes -- headers, lists, etc.
    still carry useful semantic content as-is)."""
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file as one string, cleaned up."""
    reader = PdfReader(str(file_path))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    raw_text = "\n".join(text_parts)
    return clean_extracted_text(raw_text)

def chunk_text(text, chunk_size=250, overlap=40):
    """
    Split text into chunks using a recursive strategy: try splitting on
    the biggest natural boundary first (paragraph breaks), and only fall
    back to smaller boundaries (lines, then sentences, then a hard
    character cut) when a piece is still too large. This avoids cutting
    mid-sentence or mid-paragraph whenever a natural boundary is
    available, while still guaranteeing every chunk respects chunk_size.

    Based on the widely-used recursive text splitting approach (similar
    to LangChain's RecursiveCharacterTextSplitter) -- chosen over pure
    fixed-size chunking after Task 3.x testing showed fixed-size cuts
    sometimes split coherent content (e.g. numbered sections) awkwardly.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", ""]  # "" = hard character-level fallback
    return _recursive_split(text, chunk_size, overlap, separators)


def _recursive_split(text, chunk_size, overlap, separators):
    """Core recursive splitting logic, tries each separator in order."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        # Last resort: hard character-count split with overlap, same as
        # the original fixed-size approach.
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    pieces = text.split(separator)
    chunks = []
    current_chunk = ""

    for piece in pieces:
        candidate = current_chunk + separator + piece if current_chunk else piece

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(piece) > chunk_size:
                # This single piece is still too big -- recurse with the
                # next, smaller separator.
                if remaining_separators:
                    chunks.extend(_recursive_split(piece, chunk_size, overlap, remaining_separators))
                else:
                    chunks.append(piece)
                current_chunk = ""
            else:
                current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]

def process_document(file_path, chunk_size=250, overlap=40):
    """
    Full pipeline for one document: extract text, then chunk it.
    Supports .pdf, .txt, .docx, .csv, and .md.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext == ".csv":
        text = extract_text_from_csv(file_path)
    elif ext == ".md":
        text = extract_text_from_md(file_path)
    else:
        raise ValueError(f"Unsupported file type for text_pipeline: {ext}")

    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)