# baseline/retriever/retriever.py

import os
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    A document retriever that uses SentenceTransformers for embeddings and FAISS for indexing.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=200):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        self.chunk_size = chunk_size

    def _chunk_text(self, text):
        """
        Splits the text into chunks based on punctuation and the target chunk_size.
        """
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) < self.chunk_size:
                current += sentence + " "
            else:
                chunks.append(current.strip())
                current = sentence + " "
        if current:
            chunks.append(current.strip())
        return chunks

    def add_documents(self, texts):
        """
        Adds a list of documents to the retriever. Documents are chunked and embedded.
        """
        all_chunks = []
        for text in texts:
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)
            all_chunks.extend(chunks)
        self.embeddings = self.model.encode(all_chunks).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, question, top_k=3):
        """
        Queries the retriever and returns the top_k matching chunks along with distances.
        """
        query_vec = self.model.encode([question]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        # Return list of tuples: (chunk text, distance)
        return [(self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save(self, folder="retriever_store"):
        """
        Saves the FAISS index and documents list to disk.
        """
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, folder="retriever_store"):
        """
        Loads the FAISS index and documents list from disk.
        """
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
