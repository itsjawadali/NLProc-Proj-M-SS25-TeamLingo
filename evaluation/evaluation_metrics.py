import json
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(test_inputs_path, logs_path):
    # Load test questions + expected keywords
    with open(test_inputs_path, 'r') as f:
        tests = json.load(f)

    # Load pipeline run logs
    with open(logs_path, 'r') as f:
        logs = [json.loads(line) for line in f]

    # Map question → answer
    answers = { entry["question"]: entry["answer"].lower() for entry in logs }

    # Build true/pred arrays
    y_true, y_pred = [], []
    for test in tests:
        q = test["question"]
        expected = test["expected_keywords"]
        ans = answers.get(q, "")
        for kw in expected:
            y_true.append(1)
            y_pred.append(1 if kw.lower() in ans else 0)

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")
    return {"precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    compute_metrics(
      "evaluation/test_inputs.json",
      "evaluation/logs/log.jsonl"
    )
