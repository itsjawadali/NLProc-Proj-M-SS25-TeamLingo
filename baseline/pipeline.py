# baseline/pipeline.py

from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from utils.file_loader import load_text_file, load_pdf
from utils.logger import log_query

def main():
    # Initialize retriever and generator
    retriever = Retriever()
    generator = Generator()
    
    # Load documents (from baseline/data folder)
    docs = []
    docs.append(load_text_file("baseline/data/sample.txt"))
    docs.append(load_pdf("baseline/data/sample.pdf"))
    
    # Add documents to retriever
    retriever.add_documents(docs)
    
    # Define a sample question
    question = "How can renewable energy help the environment?"
    
    # Retrieve context chunks
    retrieved = retriever.query(question, top_k=3)
    context_chunks = [chunk for chunk, _ in retrieved]
    
    # Build a prompt and generate an answer
    prompt = generator.build_prompt(question, context_chunks)
    answer = generator.generate_answer(prompt)
    
    # Log the query details
    log_query(question, context_chunks, prompt, answer, group_id="pipeline_test")
    
    # Output results
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
