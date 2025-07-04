# evaluation/test_generator.py

import unittest
from baseline.generator.generator import Generator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = Generator()

    def test_build_and_generate(self):
        question = "Why is renewable energy important?"
        # Simulated retrieved chunks:
        chunks = [
            "Renewable energy reduces greenhouse gas emissions.",
            "It helps mitigate climate change by replacing fossil fuels."
        ]
        prompt = self.gen.build_prompt(question, chunks)
        ans = self.gen.generate_answer(prompt)
        self.assertTrue(len(ans) > 0)
        self.assertIn("energy", ans.lower())

if __name__ == '__main__':
    unittest.main()
