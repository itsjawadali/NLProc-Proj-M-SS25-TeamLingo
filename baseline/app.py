import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from baseline.pipeline import answer_question
from evaluation.evaluation_metrics import evaluate

# — Apply our “environment” theme via CSS —
st.markdown(
    """
    <style>
      /* Main background */
      [data-testid="stAppViewContainer"] {
        background-color: #e0f7da;
      }
      /* Sidebar background & wider width */
      [data-testid="stSidebar"] {
        background-color: #b8e6c8;
      }
      /* Increase sidebar max-width */
      section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        max-width: 300px !important;
      }
      /* Wrap long filenames */
      section[data-testid="stSidebar"] p, 
      section[data-testid="stSidebar"] li {
        white-space: normal !important;
        overflow-wrap: break-word !important;
      }

      /* Transparent header & toolbar tweaks */
      [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: rgba(0,0,0,0);
      }

      /* Headings in dark green */
      h1, h2, h3, h4, h5, h6 {
        color: #2e7d32 !important;
      }

      /* Buttons */
      .stButton>button {
        background-color: #4caf50 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
      }
      .stButton>button:hover {
        background-color: #43a047 !important;
      }

      /* Text area */
      .stTextArea>div>div>textarea {
        background-color: #ffffff !important;
        color: #2e7d32 !important;
        border: 2px solid #4caf50 !important;
        border-radius: 5px !important;
      }

      /* Slider & radio labels */
      .stSlider, .stRadio {
        color: #2e7d32 !important;
      }

      /* Metrics widget labels */
      .stMetricLabel {
        color: #1b5e20 !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# — Layout —
st.set_page_config(page_title="TeamLingo RAG Pipeline", layout="wide")

# Sidebar for mode selection
st.sidebar.title("🔧 Configuration")
mode = st.sidebar.radio("Select Mode:", ["🧪 Evaluate on Test Set", "🔍 Ask a Question", ])

# — Add some top padding to the sidebar section —
st.sidebar.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)

# — Show loaded articles in specialization/data —
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "specialization", "data"))

st.sidebar.markdown("### 📚 Articles")
try:
    files = sorted(os.listdir(data_dir))
    if files:
        # build a UL with green dot bullets and small text
        html = "<ul style='list-style-position: outside; list-style-type: circle'>"
        for fname in files:
            html += (
                "<li>"
                f"<small>{fname}</small>"
                "</li>"
            )
        html += "</ul>"
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<small>_(no files found)_</small>", unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.markdown("<small>_data directory not found_</small>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f"<small>_Error loading articles: {e}_</small>", unsafe_allow_html=True)

# Main title
st.title("🍃 Environment & Climate-Change Q&A 🍃")

if mode == "🔍 Ask a Question":
    st.header("💬 Ask a Question")
    question = st.text_area("Enter your question:", height=120)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, step=0.01)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("❗ Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(question=question, threshold=threshold)
                    st.success("✅ Answer:")
                    st.markdown(f"<div style='font-size:1.1em; color:#1b5e20;'>{answer}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


elif mode == "🧪 Evaluate on Test Set":
    st.header("🧪 Evaluate on Test Cases")

    if st.button("Run Evaluation"):
        with st.spinner("Running test cases..."):
            out = evaluate("evaluation/test_inputs.json")
            results = out["results"]
            metrics = out["metrics"]

        # Detailed per-test results
        for i, r in enumerate(results):
            with st.expander(f"❓ Q{i+1}: {r['question']}"):
                st.markdown(f"**Answer:** {r['answer']}")
                st.markdown(f"**Expected Keywords:** {', '.join(r['expected'])}")
                st.markdown(f"**Type:** `{r['qtype']}`")

                if r["qtype"] == "explanation":
                    st.markdown(f"**ROUGE-L Recall:** `{r['rougeL']:.2f}`")
                else:
                    hits_md = "\n".join(
                        f"- {'✅' if hit else '❌'} **{kw}**"
                        for kw, hit in r["hits"]
                    )
                    st.markdown("**Keyword Hits:**")
                    st.markdown(hits_md)

        # Evaluation metrics
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics['precision']:.2f}")
        col2.metric("Recall",    f"{metrics['recall']:.2f}")
        col3.metric("F1 Score",  f"{metrics['f1']:.2f}")
