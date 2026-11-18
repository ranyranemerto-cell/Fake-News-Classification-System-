import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Fake News Classification System",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

try:
    model, tfidf = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

st.title("📰 Fake News Classification System")

st.write(
    "A Machine Learning and Natural Language Processing "
    "application that classifies news articles as REAL or FAKE."
)

st.info(
    "ℹ️ Note: This model detects writing patterns associated with fake news. "
    "It does not verify the absolute factual truth of the content."
)

st.divider()

st.subheader("Enter News Article")

news_title = st.text_input(
    "News Title",
    placeholder="Enter the news headline here..."
)

news_text = st.text_area(
    "News Article",
    placeholder="Paste or type the news article here...",
    height=250
)

col1, col2 = st.columns([3, 1])

with col1:
    classify_button = st.button("Classify News", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.rerun()

if classify_button:
    if not news_title.strip() and not news_text.strip():
        st.warning("Please enter a news title or article.")
    else:
        combined_text = news_title + " " + news_text
        input_tfidf = tfidf.transform([combined_text])
        prediction = model.predict(input_tfidf)[0]

        probabilities = model.predict_proba(input_tfidf)[0]
        class_labels = model.classes_
        proba_dict = dict(zip(class_labels, probabilities))

        st.divider()
        st.subheader("Classification Result")

        if prediction == 0:
            st.error("🚨 FAKE NEWS")
            st.write("The model classified this article as **FAKE**.")
        else:
            st.success("📰 REAL NEWS")
            st.write("The model classified this article as **REAL**.")

        st.write("### Probability Breakdown")

        proba_df = pd.DataFrame({
            "Classification": ["FAKE", "REAL"],
            "Probability (%)": [proba_dict.get(0, 0) * 100, proba_dict.get(1, 0) * 100]
        })

        st.write(f"**FAKE:** {proba_dict.get(0, 0) * 100:.2f}%")
        st.progress(float(proba_dict.get(0, 0)))

        st.write(f"**REAL:** {proba_dict.get(1, 0) * 100:.2f}%")
        st.progress(float(proba_dict.get(1, 0)))

        st.caption(
            "Confidence represents the model's estimated "
            "probability for its predicted class. It does not "
            "guarantee that the article is factually true or false."
        )

st.divider()

st.caption(
    "Disclaimer: This application provides automated "
    "machine learning predictions based on patterns in its "
    "training dataset. Always verify news through reliable "
    "sources before accepting it as factual."
)