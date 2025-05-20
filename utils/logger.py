# utils/logger.py

import os
import json
from datetime import datetime

LOG_PATH = "evaluation/logs/log.jsonl"

def log_query(question, retrieved_chunks, prompt, generated_answer, group_id="default"):
    """
    Logs the query details into a JSONL file.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": generated_answer,
        "group_id": group_id
    }
    with open(LOG_PATH, "a", encoding="utf-8") as logfile:
        logfile.write(json.dumps(log_entry) + "\n")
