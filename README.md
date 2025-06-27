🌍 Environment and Climate Change Question-Answering System (Team Lingo)
This project implements a comprehensive Retrieval-Augmented Generation (RAG) pipeline tailored specifically for answering natural language questions about climate change, grounded in authoritative scientific literature. It allows users to input queries in plain English and receive precise, contextually-supported answers from structured documents (PDF format).

The system uses Sentence Transformers for embedding, FAISS for fast similarity-based retrieval, and FLAN-T5 for high-quality answer generation, making it suitable for effective and efficient NLP experimentation and real-world deployment.

🌟 Overall Project Goal
To develop an end-to-end RAG pipeline that:

Accepts scientific PDF documents.

Extracts, cleans, and chunks the content.

Indexes these chunks using FAISS and Sentence Transformers.

Retrieves contextually relevant information based on natural language queries.

Generates concise natural-language answers using a pre-trained FLAN-T5-base model.

Logs each query-answer interaction for transparency and evaluation.

📂 Folder and File Overview
graphql
Copy
Edit
NLProc-Proj-M-SS25-TeamLingo-main/
│
├── 📂 baseline/
│   ├── pipeline.py                   # Main orchestrator script
│   ├── extractor.py                  # Post-processing (deduplication)
│   ├── 📂 generator/
│   │   ├── generator.py              # FLAN-T5-base answer generation
│   │   └── utils.py                  # Prompt building and question classification
│   └── 📂 retriever/
│       └── retriever.py              # Retrieval using FAISS and embeddings
│
├── 📂 evaluation/
│   ├── logs/
│   │   └── log.jsonl                 # Logs of queries and answers
│   └── test_batch.py                 # Batch evaluation harness
│
├── 📂 specialization/
│   ├── data/
│   │   └── The_Reality_of_Climate_Change_Evidence_Impacts_and.pdf # Source document
│   └── specialization.py             # PDF chunking and loading logic
│
├── 📂 scripts/
│   ├── build_corpus.py               # Corpus creation from PDF
│   └── create_indexes.py             # Embedding and indexing script
│
├── 📂 utils/
│   ├── file_loader.py                # PDF and text file loading utility
│   ├── logger.py                     # Query logging utility
│   └── utils.py                      # General utilities (fallback)
│
├── 📂 corpus/
│   └── chunks.jsonl                  # JSONL corpus output (chunks)
│
├── 📂 models/
│   ├── corpus_records.pkl            # Chunk metadata
│   └── faiss_index.idx               # FAISS index of chunk embeddings
│
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
🧠 Key Functionalities
🔹 pipeline.py
Coordinates end-to-end retrieval and generation.

Dynamically selects appropriate context size based on question type (numeric, list, definition, explanation).

Performs query classification (generator/utils.py) and builds tailored prompts.

🔹 retriever.py
Class Retriever: Loads chunks, creates embeddings with Sentence Transformer (all-MiniLM-L6-v2).

Retrieves relevant document chunks via FAISS index.

Optional cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-12-v2) for improved retrieval precision.

🔹 generator.py
Class Generator: Uses FLAN-T5 (google/flan-t5-base) for answer generation.

Generates answers based purely on the retrieved chunks context.

🔹 extractor.py
Cleans and deduplicates generated answers (particularly lists).

🔹 file_loader.py
Loads and processes text and PDF documents (pdfplumber).

🔹 logger.py
Logs detailed information (question, retrieved chunks, prompt, and generated answer) to JSONL for analysis.

🧪 Evaluation Methods (evaluation/)
🔸 test_batch.py
Loads a predefined test set from test_inputs.json.

Evaluates numeric, list, and definition queries using keyword matching.

Evaluates explanation queries using ROUGE-L Recall metric.

Prints detailed output and aggregated evaluation metrics.

🚀 How to Run the Project
1. Clone the Repository

2. Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

5. Build the Corpus
python scripts/build_corpus.py specialization/data corpus/chunks.jsonl

5. Create the Index
python scripts/create_indexes.py corpus/chunks.jsonl models

7. Run the Pipeline (Single Query)
python -m baseline.pipeline --question "How much ice did Greenland lose annually?"
8. Perform Batch Evaluation
python -m evaluation.test_batch

📊 Current Evaluation Metrics
Question Type	Precision	Recall	F1-Score	ROUGE-L Recall
Numeric/List/Definition	1.00	0.67	0.80	-
Explanation (Descriptive)	-	-	-	~0.15 (avg)

Precision: Perfect, meaning no hallucinations.

Recall: Moderate, showing areas for improvement, especially in multi-item and descriptive questions.

🚧 Limitations & Next Steps
Current Limitations:
Token overflow in explanation contexts.

Moderate recall in complex multi-item queries.

Future Improvements:
Enhanced Context Management: Optimal truncation and context sizing.

Hybrid Retrieval: Dense embeddings + Cross-encoder reranking.

Domain-Specific Model Tuning: Fine-tuning FLAN-T5 on climate-focused Q&A pairs.

Evaluation Enhancements: Semantic similarity metrics beyond keyword matching.

🛠️ Key Commands Quick Reference

# Build corpus
python scripts/build_corpus.py specialization/data corpus/chunks.jsonl

# Create FAISS index
python scripts/create_indexes.py corpus/chunks.jsonl models

# Run a single query
python -m baseline.pipeline --question "<Your question here>"

# Evaluate entire test batch
python -m evaluation.test_batch
📗 Data Sources
Primary Document:
The Reality of Climate Change: Evidence, Impacts and Choices

👥 Team Members
Jawad Ali

Saad Abdullah

Bilal Ahmad

📅 Last Updated
June 26, 2025
