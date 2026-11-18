# ============================================================
# FAKE NEWS CLASSIFICATION SYSTEM (Streamlit App)
# Label mapping verified from the dataset: 0 = REAL, 1 = FAKE
# ============================================================

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_OK = True
except Exception:
    LANGDETECT_OK = False

st.set_page_config(page_title="Fake News Classification System", page_icon="📰")

LABEL_MAP = {0: "REAL", 1: "FAKE"}

# ------------------------------------------------------------
# Load model, vectorizer, and (optional) metrics
# ------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as f:
            return json.load(f)
    except Exception:
        return None


model, vectorizer = load_artifacts()
metrics = load_metrics()

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def is_english(text):
    if not LANGDETECT_OK:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return False


def top_words(model, vectorizer, vec, n=8):
    """Words in the article that pushed the prediction toward FAKE or REAL."""
    if hasattr(model, "coef_"):
        weights = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return [], []

    names = vectorizer.get_feature_names_out()
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return [], []

    values = vec[0, idx].toarray().ravel()
    contrib = values * weights[idx]
    order = np.argsort(contrib)

    # Positive contribution -> FAKE (class 1), negative -> REAL (class 0)
    fake = [(names[idx[i]], contrib[i]) for i in order[::-1][:n] if contrib[i] > 0]
    real = [(names[idx[i]], contrib[i]) for i in order[:n] if contrib[i] < 0]
    return fake, real


# ------------------------------------------------------------
# Example articles (made-up samples for demo purposes)
# ------------------------------------------------------------

EXAMPLES = {
    "Example 1 (formal news style)": (
        "Senate committee approves infrastructure funding bill",
        "WASHINGTON - A Senate committee on Tuesday approved a funding bill that "
        "would allocate money for road and bridge repairs across several states, "
        "officials said. The measure passed after two days of hearings, during which "
        "lawmakers from both parties discussed how the money would be distributed. "
        "The bill now heads to the full Senate, where a vote is expected next month, "
        "according to a spokesman for the committee chairman.",
    ),
    "Example 2 (sensational style)": (
        "UNBELIEVABLE!!! THEY ARE HIDING THE TRUTH FROM YOU [VIDEO]",
        "WAKE UP AMERICA! The mainstream media will NEVER show you this shocking "
        "video that exposes the elites and their secret plan. Share this before it "
        "gets DELETED! They do not want you to know what is really happening. "
        "Patriots everywhere are furious and demanding answers right now!!!",
    ),
}


def set_example(title, article):
    st.session_state["title_input"] = title
    st.session_state["article_input"] = article


def clear_inputs():
    st.session_state["title_input"] = ""
    st.session_state["article_input"] = ""


# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------

st.title("📰 Fake News Classification System")
st.caption(
    "A Machine Learning and Natural Language Processing application "
    "that classifies news articles as REAL-style or FAKE-style."
)

st.info(
    "Note: This model detects writing patterns associated with fake news. "
    "It does not verify the absolute factual truth of the content. "
    "It supports English-language news only."
)

st.divider()

# ------------------------------------------------------------
# Input section
# ------------------------------------------------------------

st.header("Enter News Article")

st.write("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
for col, (name, (ex_title, ex_text)) in zip(ex_cols, EXAMPLES.items()):
    col.button(
        name,
        on_click=set_example,
        args=(ex_title, ex_text),
        use_container_width=True,
    )

title = st.text_input(
    "News Title",
    key="title_input",
    placeholder="Enter the news headline here...",
)
article = st.text_area(
    "News Article",
    key="article_input",
    height=250,
    placeholder="Paste the news article text here...",
)

btn_col1, btn_col2 = st.columns([3, 1])
classify_clicked = btn_col1.button(
    "Classify News", type="primary", use_container_width=True
)
btn_col2.button("Clear", on_click=clear_inputs, use_container_width=True)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if classify_clicked:
    if not article.strip():
        st.warning("Please enter the news article text first.")
    else:
        # Must match the training format: title + " " + text
        content = f"{title.strip()} {article.strip()}"

        if not is_english(content):
            st.warning(
                "This system supports English-language news only. "
                "Results for other languages may be unreliable."
            )

        if len(content.split()) < 30:
            st.warning(
                "The text is very short. Predictions on short text are less reliable."
            )

        vec = vectorizer.transform([content])
        pred = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]

        label = LABEL_MAP[pred]
        confidence = float(proba[pred]) * 100

        st.divider()
        st.subheader("Result")

        if label == "FAKE":
            st.error(f"Likely FAKE-style article ({confidence:.1f}% confidence)")
        else:
            st.success(f"Likely REAL-style article ({confidence:.1f}% confidence)")

        st.write("Probability breakdown")
        st.write(f"REAL: {proba[0] * 100:.1f}%")
        st.progress(float(proba[0]))
        st.write(f"FAKE: {proba[1] * 100:.1f}%")
        st.progress(float(proba[1]))

        if confidence < 70:
            st.warning(
                "Low confidence. Please verify this news with reliable sources."
            )

        st.info(
            "This is an automated prediction based on writing patterns. "
            "Please verify with reliable news sources before accepting "
            "or sharing this article."
        )

        # Explanation
        fake_words, real_words = top_words(model, vectorizer, vec)
        if fake_words or real_words:
            with st.expander("Why this prediction? (top influencing words)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Words pointing to FAKE**")
                    for w, _ in fake_words:
                        st.write(f"- {w}")
                    if not fake_words:
                        st.write("None found")
                with c2:
                    st.markdown("**Words pointing to REAL**")
                    for w, _ in real_words:
                        st.write(f"- {w}")
                    if not real_words:
                        st.write("None found")
                st.caption(
                    "These words show what the model learned from its training "
                    "data. They are writing patterns, not proof of truth or falsehood."
                )

# ------------------------------------------------------------
# Model performance section
# ------------------------------------------------------------

st.divider()

with st.expander("Model Performance and Limitations"):
    if metrics:
        st.write(f"Final model: **{metrics['best_model']}**")
        st.write(
            f"Training records: {metrics['train_size']:,} | "
            f"Testing records: {metrics['test_size']:,}"
        )
        st.markdown("**Model comparison** (precision, recall, and F1 are for the FAKE class)")
        st.dataframe(pd.DataFrame(metrics["comparison"]), hide_index=True)

        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual REAL", "Actual FAKE"],
            columns=["Predicted REAL", "Predicted FAKE"],
        )
        st.markdown("**Confusion matrix** (test set)")
        st.dataframe(cm)
    else:
        st.write("Metrics file not found. Run train_model.py to generate metrics.json.")

    st.markdown(
        "**Limitations**\n"
        "- English-language news only.\n"
        "- Trained on the WELFake dataset, which is mostly US political news "
        "from around 2016-2017. Performance may drop on newer or different topics.\n"
        "- The model learns writing style, not facts. A well-written false article "
        "may be classified as REAL-style, and a sensational true article as FAKE-style.\n"
        "- Results are estimates that require fact-checking."
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.caption(
    "Disclaimer: This application provides automated machine learning predictions "
    "based on patterns in its training dataset. Always verify news through "
    "reliable sources before accepting it as factual."
)