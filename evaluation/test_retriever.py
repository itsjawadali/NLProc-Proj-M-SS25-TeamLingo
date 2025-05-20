# evaluation/test_retriever.py

from baseline.retriever.retriever import Retriever
from utils.file_loader import load_text_file, load_pdf
import os

def test_retriever():
    retriever = Retriever()
    docs = []
    if os.path.exists("baseline/data/sample.txt"):
        docs.append(load_text_file("baseline/data/sample.txt"))
    if os.path.exists("baseline/data/sample.pdf"):
        docs.append(load_pdf("baseline/data/sample.pdf"))
    retriever.add_documents(docs)
    
    # Example query
    question = "What are the main causes of global warming?"
    results = retriever.query(question)
    print("Retriever Test Results:")
    for chunk, distance in results:
        print(f"Chunk: {chunk}\nDistance: {distance}\n")

if __name__ == "__main__":
    test_retriever()
