import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
from baseline.pipeline import answer_question
from baseline.generator.utils import classify_qtype
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

st.set_page_config(page_title="Semantic QA System", layout="wide")
st.title("🧠 Environment and Climate change Q&A System")

# Tabs for switching between modes
mode = st.radio("Select Mode:", ["🔍 Ask a Question", "🧪 Evaluate on Test Set"])

if mode == "🔍 Ask a Question":
    st.header("Ask a Question")
    question = st.text_area("Enter your question:", height=100)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, step=0.01)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(question=question, threshold=threshold)
                    st.success("Answer:")
                    st.markdown(f"**{answer}**")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif mode == "🧪 Evaluate on Test Set":
    st.header("Evaluate Model on Test Cases")

    if st.button("Run Evaluation"):
        with st.spinner("Running test cases..."):
            with open("evaluation/test_inputs.json", "r") as f:
                tests = json.load(f)

            y_true, y_pred = [], []
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            all_results = []

            for i, test in enumerate(tests):
                q = test["question"]
                expected = test["expected_keywords"]
                qtype = classify_qtype(q)
                ans = answer_question(q)

                test_result = {
                    "question": q,
                    "answer": ans,
                    "expected": expected,
                    "qtype": qtype,
                    "hits": [],
                    "rougeL": None
                }

                if qtype == "explanation":
                    ref = " ".join(expected)
                    rouge_scores = scorer.score(ref, ans)
                    recall = rouge_scores['rougeL'].recall
                    test_result["rougeL"] = recall
                else:
                    for kw in expected:
                        hit = int(kw.lower() in ans.lower())
                        y_true.append(1)
                        y_pred.append(hit)
                        test_result["hits"].append((kw, bool(hit)))

                all_results.append(test_result)

        # Display detailed results
        for i, r in enumerate(all_results):
            with st.expander(f"❓ Q{i+1}: {r['question']}"):
                st.markdown(f"**Answer:** {r['answer']}")
                st.markdown(f"**Expected Keywords:** {', '.join(r['expected'])}")
                st.markdown(f"**Type:** `{r['qtype']}`")

                if r["qtype"] == "explanation":
                    st.markdown(f"**ROUGE-L Recall:** `{r['rougeL']:.2f}`")
                else:
                    hit_str = [
                        f"- ✅ **{kw}**" if hit else f"- ❌ **{kw}**"
                        for kw, hit in r["hits"]
                    ]
                    st.markdown("**Keyword Hits:**")
                    st.markdown("\n".join(hit_str))

        # Summary metrics
        if y_true:
            precision = precision_score(y_true, y_pred)
            recall    = recall_score(y_true, y_pred)
            f1        = f1_score(y_true, y_pred)

            st.subheader("📊 Aggregate Metrics (non-explanation)")
            st.metric("Precision", f"{precision:.2f}")
            st.metric("Recall", f"{recall:.2f}")
            st.metric("F1 Score", f"{f1:.2f}")
