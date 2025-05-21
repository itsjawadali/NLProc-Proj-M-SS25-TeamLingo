# Document-Retriever and Generator Pipeline

This project implements a complete question-answering pipeline by combining document retrieval and language generation. It allows a user to input natural language queries and receive relevant, context-grounded answers based on information from local documents (.txt and .pdf). The system uses sentence embeddings with FAISS for similarity search and FLAN-T5 for answer generation. This is a lightweight, CPU-friendly prototype suitable for local NLP experimentation.

---

## 🌍 Overall Project Goal

To build a pipeline that:

* Accepts unstructured documents (.txt, .pdf)
* Chunks and indexes them using FAISS and Sentence Transformers
* Retrieves relevant content given a question
* Generates a natural language answer using a pre-trained language model
* Logs the process for reproducibility and evaluation

---

## 📁 Folder and File Overview

```
NLProc-Proj-M-SS25-main/
├── baseline/
│   ├── pipeline.py               # Orchestrates the entire pipeline
│   ├── generator/
│   │   ├── generator.py          # Generator class using FLAN-T5
│   ├── retriever/
│   │   ├── retriever.py          # Retriever class using FAISS + embeddings
│   ├── data/
│   │   ├── sample.txt            # Sample text file for input
│   │   ├── sample.pdf            # Sample PDF file for input
├── utils/
│   ├── file_loader.py            # File loaders for .txt and .pdf
│   ├── logger.py                 # Logs the Q&A process to a .jsonl file
├── evaluation/
│   ├── test_generator.py         # Tests for the generator
│   ├── test_retriever.py         # Tests for the retriever
│   ├── test_inputs.json          # Example Q&A pairs for batch testing
├── specialization/
│   ├── specialization.py         # Place for future topic-specific customization
├── requirements.txt              # Python dependency list
├── README.md                     # This file
```

---

## 🧠 Key Functionalities

### `baseline/pipeline.py`

Coordinates all components of the pipeline:

* Loads and indexes documents
* Handles query input
* Retrieves relevant document chunks
* Builds prompt and generates answer
* Logs everything

### `retriever.py`

Implements the `Retriever` class:

* `add_documents()` - loads and chunks documents
* `query()` - retrieves relevant chunks using FAISS + Sentence Transformers
* `save()` / `load()` - save/load FAISS index

### `generator.py`

Implements the `Generator` class:

* Uses `google/flan-t5-base`
* `build_prompt(chunks, question)` - formats query + context into a prompt
* `generate_answer()` - runs the model and returns output

### `file_loader.py`

Loads document content:

* `load_text_file()` - reads plain text
* `load_pdf()` - parses PDFs using `fitz`

### `logger.py`

Logs each query and answer as JSONL with:

* timestamp
* question
* chunks
* prompt
* answer
* group\_id (optional)

## 🧪 Evaluation (evaluation/)

test_retriever.py

Checks chunk relevance + FAISS distance

test_generator.py

Tests generation quality on fixed prompt

test_batch.py

Loads test_inputs.json, runs full pipeline

Checks whether answer is grounded in context

Prints ✅ or ❌

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone [https://github.com/itsjawadali/NLProc-Proj-M-SS25-TeamLingo.git]
cd NLProc-Proj-M-SS25-main
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full Pipeline

```bash
python -m baseline.pipeline
```

This will:

* Load sample text and PDF files
* Embed and index them
* Query with a sample question
* Generate and display an answer
* Log the entire process

---

## 🧪 Example Output

```bash
Question: How is AI used in healthcare?
Answer: predict patient outcomes, optimize treatments, and accelerate drug discovery.
```

