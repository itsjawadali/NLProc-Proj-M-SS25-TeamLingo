import json
from baseline.pipeline import answer_question
from baseline.generator.utils import classify_qtype
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

def run_tests():
    # 1) Load test definitions
    with open("evaluation/test_inputs.json", "r") as f:
        tests = json.load(f)

    # Metrics accumulators for non-explanation questions
    y_true, y_pred = [], []

    # Rouge-L scorer for explanations
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 2) Run each test case
    for test in tests:
        q = test["question"]
        expected = test["expected_keywords"]
        qtype = classify_qtype(q)

        print("\n---")
        print("Q:", q)
        ans = answer_question(q)
        print("A:", ans)

        if qtype == "explanation":
            # Join expected into a single reference string
            ref = " ".join(expected)
            rouge_scores = scorer.score(ref, ans)
            recall = rouge_scores['rougeL'].recall
            print(f"ROUGE-L Recall: {recall:.2f}")
        else:
            # 3) For each expected keyword, mark hit/miss
            for kw in expected:
                hit = 1 if kw.lower() in ans.lower() else 0
                y_true.append(1)      # every expected kw is a positive instance
                y_pred.append(hit)
                status = "FOUND" if hit else "MISSING"
                print(f"{kw}: {status}")

    # 4) Compute overall metrics for non-explanation questions
    if y_true:
        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        print("\n=== Metrics for numeric/list/definition questions ===")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1-Score:  {f1:.2f}")

if __name__ == "__main__":
    print("Device set to use cpu")
    run_tests()
