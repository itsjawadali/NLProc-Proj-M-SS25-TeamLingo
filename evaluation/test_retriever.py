# evaluation/test_retriever.py

import unittest
from baseline.retriever.retriever import Retriever
from specialization.specialization import load_and_chunk_environmental_data

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Initialize Retriever with a stricter similarity threshold
        self.retriever = Retriever(similarity_threshold=0.5)
        # Add a simple baseline chunk
        baseline_chunks = ["This is a sample baseline chunk about environment and ecosystems."]
        self.retriever.add_documents(baseline_chunks)
        # Load and add all environmental PDF chunks
        env_chunks = load_and_chunk_environmental_data("specialization/data")
        self.retriever.add_documents(env_chunks)

    def test_environmental_query_returns_something(self):
        """
        For an environmental query, ensure we get at least one retrieved chunk.
        """
        results = self.retriever.query("climate change", top_k=30, final_k=5)
        self.assertTrue(
            len(results) > 0,
            "Retriever returned no chunks for an environmental query."
        )

if __name__ == '__main__':
    unittest.main()
