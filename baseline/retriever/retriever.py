import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(
        self,
        index_path: str,
        records_path: str,
        bm25=None,
        threshold: float = 0.0,
        use_reranker: bool = False
    ):
        self.index = faiss.read_index(index_path)
        with open(records_path, 'rb') as f:
            self.records = pickle.load(f)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.bm25 = bm25
        self.threshold = threshold
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.cross_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def embed(self, query: str):
        emb = self.embedder.encode([query], convert_to_numpy=True)
        return emb.astype('float32')

    def get_top_k(self, query: str, k: int = 10):
        emb = self.embed(query)
        distances, indices = self.index.search(emb, k * 2)
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            score = 1.0 / (1.0 + dist)
            if score >= self.threshold:
                candidates.append({'record': self.records[idx], 'score': score})
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:k]

        records = [c['record'] for c in candidates]
        if self.use_reranker and records:
            records = self.rerank(query, records)

        scores = [c['score'] for c in candidates]
        return list(zip(records, scores))

    def rerank(self, query: str, records: list):
        pairs = [(query, r.text if hasattr(r, 'text') else r.get('text', '')) for r in records]
        rerank_scores = self.cross_ranker.predict(pairs)
        combined = list(zip(records, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [r for r,_ in combined]
