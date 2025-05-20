# evaluation/test_batch.py

import json
import datetime
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from utils.file_loader import load_text_file, load_pdf
from utils.logger import log_query
import os

# Initialize retriever and generator
retriever = Retriever()
generator = Generator()

# Load documents
docs = []
if os.path.exists("baseline/data/sample.txt"):
    docs.append(load_text_file("baseline/data/sample.txt"))
if os.path.exists("baseline/data/sample.pdf"):
    docs.append(load_pdf("baseline/data/sample.pdf"))
retriever.add_documents(docs)

# Load test cases from test_inputs.json
with open("evaluation/test_inputs.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

for case in test_cases:
    question = case["question"]
    expected_keywords = case.get("expected_keywords", [])
    group_id = case.get("group_id", "default")
    
    # Retrieve context chunks
    results = retriever.query(question, top_k=3)
    context_chunks = [chunk for chunk, _ in results]
    
    # Build prompt and generate answer
    prompt = generator.build_prompt(question, context_chunks)
    answer = generator.generate_answer(prompt)
    
    # Check for grounding (a simple keyword search)
    grounded = any(keyword.lower() in answer.lower() for keyword in expected_keywords)
    
    # Log details
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "retrieved_chunks": context_chunks,
        "prompt": prompt,
        "generated_answer": answer,
        "group_id": group_id,
        "is_grounded": grounded
    }
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print("Grounded ✅" if grounded else "Not grounded ❌")
    
    log_query(question, context_chunks, prompt, answer, group_id)
