#!/usr/bin/env python3
import os
import sys
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss

def main(input_jsonl: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    # 1. Load chunks
    texts = []
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            records.append(rec)

    print(f"[+] Loaded {len(texts)} chunks from {input_jsonl}")

    # 2. Encode with SentenceTransformer
    print(f"[+] Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    print(f"[+] Computed embeddings: shape={embeddings.shape}")

    # 3. Build FAISS index (cosine similarity via inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[+] Built FAISS index with {index.ntotal} vectors (dim={dim})")

    # 4. Save index and metadata
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index.idx")
    records_path = os.path.join(index_dir, "corpus_records.pkl")

    faiss.write_index(index, index_path)
    print(f"[✓] FAISS index saved to {index_path}")

    with open(records_path, "wb") as f:
        pickle.dump(records, f)
    print(f"[✓] Records metadata saved to {records_path}")

if __name__ == "__main__":
    if len(sys.argv) not in (3,4):
        print("Usage: python create_indexes.py <input_jsonl> <output_index_dir> [<model_name>]", file=sys.stderr)
        sys.exit(1)
    input_jsonl = sys.argv[1]
    index_dir = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) == 4 else "all-MiniLM-L6-v2"
    main(input_jsonl, index_dir, model_name)
