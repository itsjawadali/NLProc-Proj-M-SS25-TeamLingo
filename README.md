# RAG Project – Summer Semester 2025

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. we have started with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code

## Getting Started
1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`


# Document-Retriever and Generator Pipeline

This project implements a full pipeline for document retrieval and question answering using vector similarity search and a lightweight language model. It includes tools for loading documents, retrieving relevant content using FAISS + SentenceTransformers, generating answers using FLAN-T5, and evaluating the system.

---

## Project Structure

```
NLProc-Proj-M-SS25-main/
├── baseline/
│   ├── pipeline.py               # Main entry point combining retriever + generator
│   ├── generator/
│   │   ├── generator.py          # Generator class for building prompts and generating answers
│   ├── retriever/
│   │   ├── retriever.py          # Retriever class using FAISS and SentenceTransformers
│   ├── data/
│   │   ├── sample.txt            # Sample text document for testing
│   │   ├── sample.pdf            # Sample PDF document for testing
├── utils/
│   ├── file_loader.py            # Functions to load .txt and .pdf files
│   ├── logger.py                 # Logs pipeline queries and outputs to JSONL
├── evaluation/
│   ├── test_generator.py         # Test script for evaluating generation
│   ├── test_retriever.py         # Test script for retrieval functionality
│   ├── test_inputs.json          # Known Q&A pairs for testing
├── specialization/
│   ├── specialization.py         # Any topic-specialized models or retrieval extensions
├── requirements.txt              # Python dependencies
├── README.md                     # This file
```

---

## Functionality Overview

### 1. Retriever (`retriever.py`)

* **Class:** `Retriever`
* **Main Methods:**

  * `add_documents(docs)`: Accepts list of text documents, chunks and embeds them.
  * `query(question)`: Returns top-K similar text chunks using FAISS and cosine similarity.
  * `save(path)`: Save FAISS index and metadata.
  * `load(path)`: Load saved FAISS index.
* **Internals:**

  * Uses `sentence-transformers/all-MiniLM-L6-v2` for sentence embeddings.
  * Handles document chunking into 100–150 word sections.

### 2. Generator (`generator.py`)

* **Class:** `Generator`
* **Main Methods:**

  * `build_prompt(context, question)`: Formats a natural language prompt using retrieved chunks.
  * `generate_answer(prompt)`: Uses HuggingFace's `flan-t5-base` to generate an answer.
* **Model:** `google/flan-t5-base` (runs on CPU)

### 3. File Loader (`file_loader.py`)

* Supports `.txt` and `.pdf` document formats.
* Uses PyMuPDF (`fitz`) for PDF parsing.

### 4. Logger (`logger.py`)

* Logs each query as a JSON object with:

  * question
  * retrieved\_chunks
  * prompt
  * generated\_answer
  * timestamp
  * group\_id
* Output file: `logs.jsonl`

### 5. Test Scripts

* `test_generator.py`: Validates generation from sample documents.
* `test_retriever.py`: Tests retrieval functionality.
* `test_inputs.json`: Stores known questions with expected answers.

---

## Running the Project

### Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run Pipeline**

```bash
python -m baseline.pipeline
```

This will:

* Load documents from `baseline/data/sample.txt` and `sample.pdf`
* Add them to the FAISS index
* Run a sample query
* Retrieve matching chunks
* Build a prompt
* Generate an answer using FLAN-T5
* Log the output to `logs.jsonl`

### Example Output

```
Question: How is AI used in healthcare?
Answer: predict patient outcomes, optimize treatments, and accelerate drug discovery.
```

### Example Usage in Code (in pipeline.py)

```python
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from utils.file_loader import load_text_file, load_pdf
from utils.logger import log_query

r = Retriever()
g = Generator()

# Load documents
docs = [load_text_file("baseline/data/sample.txt"),
        load_pdf("baseline/data/sample.pdf")]

r.add_documents(docs)
query = "How is AI used in healthcare?"
chunks = r.query(query)
prompt = g.build_prompt([c[0] for c in chunks], query)
answer = g.generate_answer(prompt)
log_query(query, chunks, prompt, answer)
```

---

## Requirements

* Python >= 3.9
* Required libraries:

  * sentence-transformers
  * faiss-cpu
  * transformers
  * PyMuPDF (fitz)
  * scikit-learn
  * tqdm

Install using:

```bash
pip install -r requirements.txt
```

---

## Future Work

* Improve grounding verification using named entity checks
* Add web interface for QA
* Integrate long-context models for better summarization
* Add feedback-based retraining

---


## Authors
TeamLingo
