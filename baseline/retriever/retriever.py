# baseline/retriever/retriever.py

import pickle
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Encapsulates FAISS‐based retrieval of text chunks.
    """
    def __init__(self, index_path: str, records_path: str, threshold: float = 0.2):
        # load FAISS index
        self.index = faiss.read_index(index_path)
        # load the serialized records list
        with open(records_path, 'rb') as f:
            self.records = pickle.load(f)
        # embedder for queries
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # score cutoff
        self.threshold = threshold

    def embed(self, query: str):
        """
        Embed a single string query into a normalized vector.
        """
        vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vec.astype('float32')

    def get_top_k(self, query: str, k: int = 10):
        """
        Returns up to k records whose FAISS score >= threshold,
        as a list of (record, score) tuples.
        """
        qv = self.embed(query)
        dists, idxs = self.index.search(qv, k)
        out = []
        for dist, idx in zip(dists[0], idxs[0]):
            score = 1.0 / (1.0 + dist)
            if score >= self.threshold:
                out.append((self.records[idx], score))
        return out
