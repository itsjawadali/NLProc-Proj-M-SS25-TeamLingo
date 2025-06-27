#!/usr/bin/env python3
import sys, os

# ───── Add project root so we can import specialization.specialization ─────
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
# ────────────────────────────────────────────────────────────────────────────

#!/usr/bin/env python3
import os
import sys
import json
from specialization.specialization import (
    load_and_chunk_pdf,
    normalize_whitespace,
    chunk_by_headings
)


def load_and_chunk_txt(txt_path: str, win: int = 100, stride: int = 50):
    """
    Load a plain-text file, normalize and chunk via the same logic
    as PDFs.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    norm = normalize_whitespace(raw)
    chunks = chunk_by_headings(norm, win, stride)

    records = []
    doc_id = os.path.basename(txt_path)
    for idx, c in enumerate(chunks):
        records.append({
            "doc_id":   doc_id,
            "section":  c["section"],
            "chunk_id": idx,
            "text":     c["text"]
        })
    return records


def main(input_dir: str, output_path: str):
    all_records = []

    for fname in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            continue
        lower = fname.lower()
        if lower.endswith(".pdf"):
            recs = load_and_chunk_pdf(path)
        elif lower.endswith(".txt"):
            recs = load_and_chunk_txt(path)
        else:
            # skip other file types
            continue

        print(f"[+] {fname}: {len(recs)} chunks")
        all_records.extend(recs)

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for r in all_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[✓] Wrote {len(all_records)} total chunks to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_corpus.py <input_dir> <output_jsonl>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
