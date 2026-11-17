
# ============================================================
# FAKE NEWS CLASSIFICATION SYSTEM
# Streamlit Web Application
# ============================================================

import streamlit as st
import joblib

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Classification System",
    page_icon="📰",
    layout="centered"
)

# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL AND VECTORIZER
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

try:
    model, tfidf = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run train_model.py first."
    )
    st.stop()

# ------------------------------------------------------------
# 3. APPLICATION HEADER
# ------------------------------------------------------------

st.title("📰 Fake News Classification System")

st.write(
    "A Machine Learning and Natural Language Processing "
    "application that classifies news articles as REAL or FAKE."
)

st.divider()

# ------------------------------------------------------------
# 4. USER INPUT
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 5. PREDICTION
# ------------------------------------------------------------

if st.button("Classify News", type="primary", use_container_width=True):

    if not news_title.strip() and not news_text.strip():
        st.warning("Please enter a news title or article.")

    else:
        # Combine title and article text.
        # This follows the same format used during training.
        combined_text = news_title + " " + news_text

        # Transform input using the saved TF-IDF vectorizer
        input_tfidf = tfidf.transform([combined_text])

        # Generate prediction
        prediction = model.predict(input_tfidf)[0]

        # Get prediction confidence
        probabilities = model.predict_proba(input_tfidf)[0]
        predicted_index = list(model.classes_).index(prediction)
        confidence = probabilities[predicted_index] * 100

        # Display result
        st.divider()
        st.subheader("Classification Result")

        if prediction == 0:
            st.error("🚨 FAKE NEWS")
            st.write(
                "The model classified this article as FAKE."
            )

        else:
            st.success("📰 REAL NEWS")
            st.write(
                "The model classified this article as REAL."
            )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(confidence / 100))

        st.caption(
            "Confidence represents the model's estimated "
            "probability for its predicted class. It does not "
            "guarantee that the article is factually true or false."
        )

# ------------------------------------------------------------
# 6. DISCLAIMER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Disclaimer: This application provides automated "
    "machine learning predictions based on patterns in its "
    "training dataset. Always verify news through reliable "
    "sources before accepting it as factual."
)
