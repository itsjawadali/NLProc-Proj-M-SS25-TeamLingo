# 🌍 Environment and Climate Change Question-Answering System (Team Lingo)

This project implements a comprehensive **Retrieval-Augmented Generation (RAG)** pipeline tailored specifically for answering natural language questions about **Environment and climate change**, grounded in authoritative scientific literature.

The system supports input in plain English and produces precise, context-supported answers by leveraging structured PDF documents, Sentence Transformers, FAISS, and llama3.1 (before flan-T5-base) for high-quality generation.

---

## 🌟 Overall Project Goal

To develop an end-to-end RAG pipeline that:

- Accepts scientific PDF documents
- Extracts, cleans, and chunks content
- Indexes chunks using **FAISS** and **Sentence Transformers**
- Retrieves contextually relevant content using similarity-based retrieval
- Generates natural-language answers using **llama3.1**
- Logs each interaction for transparency and evaluation

---

## 📂 Folder and File Overview

```
NLProc-Proj-M-SS25-TeamLingo-main/
│
├── baseline/
│   ├── pipeline.py                  # Main orchestrator script
│   ├── generator/
│   │   ├── generator.py             # FLAN-T5-base answer generation
│   │   └── utils.py                 # Prompt building & query classification
│   └── retriever/
│       └── retriever.py             # Retrieval using FAISS and embeddings
│
├── evaluation/
│   ├── logs/
│   │   └── log.jsonl                # Logs of questions and generated answers
│   └── test_batch.py                # Testing Dataset
    └── evaluation_metrics.py        # Implemeted evaluation metrics
    └── test_inputs.py               # Include dataset in json format containing questions and expected keywords for evaluation.
│
├── specialization/
│   ├── data/                       # Data files
│   └── specialization.py           # PDF chunking and preprocessing
│
├── scripts/
│   ├── build_corpus.py             # Chunk corpus creation from PDF
│   └── create_indexes.py           # Embedding + FAISS indexing
│
├── utils/
│   ├── file_loader.py              # PDF/text loading (pdfplumber)
│   ├── logger.py                   # Logs query-answer interactions
│   └── utils.py                    # General utilities
│
├── corpus/
│   └── chunks.jsonl                # Extracted and cleaned document chunks
│
├── models/
│   ├── corpus_records.pkl          # Metadata for document chunks
│   └── faiss_index.idx             # FAISS index built on chunk embeddings
│
├── README.md                       # Project documentation (you are here)
└── requirements.txt                # Python dependencies
```

---

## 🧠 Key Functionalities

### 🔹 `pipeline.py`
- Coordinates the full RAG pipeline
- Classifies question types (numeric, list, definition, explanation)
- Selects optimal context and constructs prompts dynamically

### 🔹 `retriever.py`
- Embeds and retrieves text using Sentence Transformers (all-MiniLM-L6-v2)
- Retrieves top chunks with FAISS
- Optional reranking using cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`)

### 🔹 `generator.py`
- Uses `google/flan-t5-base` for controlled, accurate answer generation

### 🔹 `app.py`
- UI of the model for Q&A and test-cases evaluation

### 🔹 `file_loader.py`
- Loads and parses PDF/text files into readable text using `pdfplumber`

### 🔹 `logger.py`
- Logs question, context, prompt, and generated answer to `log.jsonl`

---

## 🧪 Evaluation Methods

### 🔸 `test_batch.py` or 🔸 `app.py`
- Loads test questions from `test_inputs.json`
- Evaluation metrics:
  - **Numeric/List/Definition/Explanation**: Keyword precision/recall
  - **Explanations**: ROUGE-L Recall
- Outputs detailed results and aggregate metrics

---

## 📈 Improvements made after mid-term presentation
- Tested hybrid-retrival( FAISS + BM25 ) but no improvements were observed in better recall.
- Swapped LLM model from Flan-T5-BASE to llama3.1 for better answer generation and accuracy
- Developed a UI for our model testing with streamLit

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone <https://github.com/itsjawadali/NLProc-Proj-M-SS25-TeamLingo.git>
or download the zip file
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the Corpus
```bash
python scripts/build_corpus.py specialization/data corpus/chunks.jsonl
```

### 5. Create the FAISS Index
```bash
python scripts/create_indexes.py corpus/chunks.jsonl models
```

### 6. Run the Pipeline (Single Query)
```bash
python -m baseline.pipeline --question "How much ice did Greenland lose annually?"
```

### 7. Run Batch Evaluation
```bash
python -m evaluation.test_batch
```
## 🧪 Evaluation Methods
### 8. Launch Streamlit UI
```bash
streamlit run baseline/app.py
---



## 📗 Data Sources

- **Primary Document**:
  - *The Reality of Climate Change: Evidence, Impacts and Choices* (PDF)
  - *Are we adapting to climate change* (PDF)
  - *TFinancial climate risk a review of recent advances* (PDF)

---

## 👥 Team Members

- Jawad Ali  
- Saad Abdullah  
- Bilal Ahmad

---
