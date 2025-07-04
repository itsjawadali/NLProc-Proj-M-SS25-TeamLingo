#!/usr/bin/env python3
import os
import re
import sys
import pdfplumber
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from every page of the given PDF.
    """
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=1) or ""
            text_pages.append(t)
    return "\n".join(text_pages)


def normalize_whitespace(text: str) -> str:
    """
    Collapse spurious line-breaks from PDF, but keep paragraph splits.
    """
    # Mark real paragraph breaks
    text = re.sub(r'\n\s*\n', '<PARA>', text)
    # Collapse all other newlines/spaces into a single space
    text = re.sub(r'\s*\n\s*', ' ', text)
    # Restore paragraph breaks
    return text.replace('<PARA>', '\n\n').strip()


def detect_headings(text: str) -> List[Dict[str, int]]:
    """
    Finds headings in the text. Matches either numbered headings or ALL-CAPS lines.
    Returns a list of dicts with 'start' index and 'title'.
    """
    pattern = re.compile(
        r'^(?P<title>(?:\d+(?:\.\d+)*\s+.+|[A-Z][A-Z0-9 ]{3,}))$',
        re.MULTILINE
    )
    return [{"start": m.start(), "title": m.group("title").strip()}
            for m in pattern.finditer(text)]


def sliding_window_chunk(words: List[str], win: int, stride: int) -> List[str]:
    """
    Breaks a list of words into overlapping windows.
    """
    chunks = []
    for i in range(0, len(words), stride):
        window = words[i:i + win]
        if not window:
            break
        chunks.append(" ".join(window))
        if i + win >= len(words):
            break
    return chunks


def chunk_by_headings(text: str, win: int, stride: int) -> List[Dict[str, str]]:
    """
    Splits text into sections based on headings, then applies sliding-window
    chunking within each section (or entire doc if no headings).
    """
    headings = detect_headings(text)
    raw_chunks = []

    if not headings:
        # Fallback: chunk entire document
        words = text.split()
        for sub in sliding_window_chunk(words, win, stride):
            raw_chunks.append({"section": "FULL_DOCUMENT", "text": sub})
    else:
        # Add an artificial "end" marker
        headings.append({"start": len(text), "title": None})
        for idx in range(len(headings) - 1):
            sec = headings[idx]
            start, end = sec["start"], headings[idx + 1]["start"]
            section_text = text[start:end].strip()
            paras = [p.strip() for p in section_text.split("\n\n") if p.strip()]
            for para in paras:
                words = para.split()
                if len(words) <= win:
                    raw_chunks.append({"section": sec["title"], "text": para})
                else:
                    for sub in sliding_window_chunk(words, win, stride):
                        raw_chunks.append({"section": sec["title"], "text": sub})
    return raw_chunks


def load_and_chunk_pdf(
    pdf_path: str,
    win: int = 100,
    stride: int = 50
) -> List[Dict]:
    """
    Full pipeline: extract → normalize → detect headings → chunk → wrap records.
    """
    raw = extract_text_from_pdf(pdf_path)
    norm = normalize_whitespace(raw)
    chunks = chunk_by_headings(norm, win, stride)

    records = []
    doc_id = os.path.basename(pdf_path)
    for idx, c in enumerate(chunks):
        records.append({
            "doc_id":   doc_id,
            "section":  c["section"],
            "chunk_id": idx,
            "text":     c["text"]
        })
    return records


def load_and_chunk_environmental_data(
    data_dir: str,
    win: int = 100,
    stride: int = 50
) -> List[Dict]:
    """
    Convenience wrapper to chunk the climate-change PDF located in `data_dir`.
    Example:
        env_chunks = load_and_chunk_environmental_data("specialization/data")
    """
    pdf_name = "The_Reality_of_Climate_Change_Evidence_Impacts_and.pdf"
    pdf_path = os.path.join(data_dir, pdf_name)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Could not find {pdf_name} in {data_dir}")
    return load_and_chunk_pdf(pdf_path, win, stride)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pdf_path_or_data_dir>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    # If it's a directory, use the environmental wrapper
    if os.path.isdir(arg):
        recs = load_and_chunk_environmental_data(arg)
    else:
        recs = load_and_chunk_pdf(arg)

    for r in recs:
        print(r)
