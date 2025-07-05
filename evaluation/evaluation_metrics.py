import json
from baseline.pipeline import answer_question
from baseline.generator.utils import classify_qtype
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

def evaluate(test_inputs_path):
    """
    Runs every question in test_inputs_path through answer_question(),
    classifies by qtype, and computes:
      - ROUGE-L recall for 'explanation' questions
      - Precision/Recall/F1 over keyword hits for others
    Returns a dict with:
      - 'results': list of per-test dicts
      - 'metrics': dict with aggregate precision, recall, f1 (non-explanation only)
    """
    # Load test definitions
    with open(test_inputs_path, 'r') as f:
        tests = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    y_true, y_pred = [], []
    all_results = []

    for test in tests:
        q        = test["question"]
        expected = test["expected_keywords"]
        qtype    = classify_qtype(q)
        ans      = answer_question(q)

        result = {
            "question": q,
            "answer": ans,
            "expected": expected,
            "qtype": qtype,
            "hits": [],
            "rougeL": None
        }

        if qtype == "explanation":
            # compute ROUGE-L recall
            ref_scores = scorer.score(" ".join(expected), ans)
            result["rougeL"] = ref_scores['rougeL'].recall
        else:
            # compute keyword-hit metrics
            ans_lower = ans.lower()
            for kw in expected:
                hit = int(kw.lower() in ans_lower)
                y_true.append(1)
                y_pred.append(hit)
                result["hits"].append((kw, bool(hit)))

        all_results.append(result)

    # Aggregate non-explanation metrics
    if y_true:
        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
    else:
        precision = recall = f1 = 0.0

    return {
        "results": all_results,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }

if __name__ == "__main__":
    out = evaluate("evaluation/test_inputs.json")
    print("Aggregate metrics (non-explanation):")
    print(f"  Precision: {out['metrics']['precision']:.2f}")
    print(f"  Recall:    {out['metrics']['recall']:.2f}")
    print(f"  F1 Score:  {out['metrics']['f1']:.2f}")
