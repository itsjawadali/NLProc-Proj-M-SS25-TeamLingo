import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from generator.generator import Generator
from generator.utils import (
    build_prompt,
    classify_qtype,
    build_explanation_prompt
)
# from utils.logger import log_query
from utils.logger import log_query

def _get_text(r):
    if isinstance(r, dict):
        return r.get("text", "").strip()
    elif hasattr(r, "text"):
        return r.text.strip()
    else:
        return str(r).strip()

class Retriever:
    def __init__(self, index_path, records_path, threshold=0.2):
        self.index     = faiss.read_index(index_path)
        with open(records_path, 'rb') as f:
            self.records = pickle.load(f)
        self.embedder  = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

    def embed(self, q):
        emb = self.embedder.encode([q], convert_to_numpy=True)
        return emb.astype('float32')

    def get_top_k(self, q, k=10):
        dists, idxs = self.index.search(self.embed(q), k)
        out = []
        for dist, idx in zip(dists[0], idxs[0]):
            score = 1.0 / (1.0 + dist)
            if score >= self.threshold:
                out.append((self.records[idx], score))
        return out

def answer_question(
    question: str,
    index_path:   str = "models/faiss_index.idx",
    records_path: str = "models/corpus_records.pkl",
    threshold:    float = 0.2
) -> str:
    # 1) Determine question type
    qtype = classify_qtype(question)

    # 2) Retrieve just 2 contexts for explanations, else 3
    retriever = Retriever(index_path, records_path, threshold=threshold)
    desired_k = 2 if qtype == "explanation" else 3
    candidates = retriever.get_top_k(question, k=10)[:desired_k]
    contexts   = [_get_text(r) for r,_ in candidates]

    # 3) Build prompt
    if qtype == "explanation":
        prompt = build_explanation_prompt(question, contexts)
    else:
        prompt = build_prompt(question, contexts)
        if qtype == "list":
            prompt += "\n\nPlease list *all* the items asked for, exactly as they appear above."

    # 4) Generate (give explanations more room)
    generator = Generator()
    if qtype == "explanation":
        raw_ans = generator.generate(prompt, max_length=200)
    else:
        raw_ans = generator.generate(prompt)

    # 5) Log and return
    log_query(
        question,
        list(zip(contexts, [float(s) for _,s in candidates])),
        prompt,
        raw_ans
    )
    return raw_ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q",         "--question", required=True, help="Your query")
    parser.add_argument("--threshold", type=float, default=0.2,                  help="Min FAISS similarity to keep a chunk")
    args = parser.parse_args()

    answer = answer_question(
        question=args.question,
        threshold=args.threshold
    )
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
