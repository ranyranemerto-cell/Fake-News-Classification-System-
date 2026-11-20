# Fake News Classification System

A machine learning and natural language processing (NLP) web application that classifies **English-language** news articles as **REAL-style** or **FAKE-style**, built with Python, scikit-learn, and Streamlit.

**Live demo:** https://verify-fact-fake-news-system.streamlit.app

> **Important:** This system detects *writing patterns* associated with fake news. It does **not** verify whether the content is factually true. Predictions are automated estimates that require fact-checking.

---

## Problem

Fake news spreads quickly through social media and other online platforms, and many readers find it difficult to tell whether an article is trustworthy. This project helps students and online readers become more aware of potentially misleading articles by flagging text that resembles known fake news.

## Features

- Paste a news title and article text, then get a **REAL-style / FAKE-style** prediction
- **Confidence score** and probability breakdown
- **"Why this prediction?"** panel showing the words that pushed the result toward FAKE or REAL
- **Model performance** section (model comparison and confusion matrix)
- **English-only check**, with a warning for other languages
- Built-in example articles for quick demos
- Clear disclaimers that results require fact-checking

## Dataset

| Item | Details |
|---|---|
| Name | WELFake Dataset |
| Source | https://zenodo.org/records/4561253 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Original records | 72,134 labeled news articles |
| Input features | News title and article text |
| Target | `label` (**0 = REAL, 1 = FAKE**) |

**Label verification:** the dataset description and the actual data disagree on the meaning of the labels. We checked sample headlines from each label and found that label 0 contains normal, formal news headlines and label 1 contains sensational, clickbait-style headlines. The project therefore uses **0 = REAL** and **1 = FAKE**.

The dataset files are **not included** in this repository because of their size. Download them from the link above.

**Citation:** Verma, P. K., Agrawal, P., & Prodan, R. (2021). *WELFake dataset for fake news detection in text data* (Version 0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4561253

Related paper: Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. "WELFake: Word Embedding Over Linguistic Features for Fake News Detection," *IEEE Transactions on Computational Social Systems*. https://doi.org/10.1109/TCSS.2021.3068519

### Data cleaning (`check_dataset.py`)

| Step | Records |
|---|---|
| Original dataset | 72,134 |
| After removing missing or empty title/text | 70,793 |
| After removing 8,607 duplicate texts | **62,186** |

No texts with conflicting labels were found. The cleaned file is saved as `WELFake_Cleaned.csv`.

## Method (`train_model.py`)

1. Combine title and text into one input.
2. Split the data 80/20 with a **stratified** split (`random_state=42`): 49,748 training records and 12,438 testing records. Duplicates were removed **before** the split to avoid data leakage.
3. Convert text to features using **TF-IDF** (English stop words removed, up to 50,000 features, unigrams and bigrams, `min_df=2`, sublinear TF).
4. Train two models and compare them:
   - **Multinomial Naive Bayes** (baseline)
   - **Logistic Regression** (alternative model)
5. Select the model with the best F1-score and save it as `fake_news_model.joblib`, together with `tfidf_vectorizer.joblib` and `metrics.json`.

## Results (test set, 12,438 articles)

Precision, recall, and F1 are for the **FAKE** class.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 90.08% | 87.56% | 90.48% | 88.99% |
| **Logistic Regression (final)** | **96.18%** | **96.27%** | **95.07%** | **95.67%** |

Confusion matrix of the final model:

| | Predicted REAL | Predicted FAKE |
|---|---|---|
| **Actual REAL** | 6,721 | 203 |
| **Actual FAKE** | 272 | 5,242 |

**Near-duplicate check:** in a random sample of 2,000 test articles, 2.80% had a similarity of 0.90 or higher to a training article, so the risk of leakage from near-duplicates is small.

## Limitations

- **English-language news only.**
- The dataset is mostly US political news from around 2016-2017, so performance may drop on newer or different topics.
- The model learns **writing style, not facts**. A well-written false article may be classified as REAL-style, and a sensational true article as FAKE-style.
- Formal news-agency wording (for example "said", "Washington", "spokesman") tends to push predictions toward REAL, which shows the model partly learns source style.
- Results are estimates and always need fact-checking with reliable sources.

## Tech stack and libraries

- Python
- [scikit-learn](https://scikit-learn.org): TF-IDF, Multinomial Naive Bayes, Logistic Regression, evaluation metrics
- pandas, NumPy, SciPy: data handling
- joblib: saving and loading the model
- [Streamlit](https://streamlit.io): web application
- langdetect: English-language check

## Run locally

```bash
git clone https://github.com/ranyranemerto-cell/Fake-News-Classification-System-.git
cd Fake-News-Classification-System-
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, download WELFake, place `WELFake_Dataset.csv` in the project folder, then run:

```bash
python check_dataset.py   # cleans the data and creates WELFake_Cleaned.csv
python train_model.py     # trains the models and saves the model files and metrics.json
```

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `check_dataset.py` | Dataset verification and cleaning |
| `train_model.py` | Model training, comparison, and evaluation |
| `fake_news_model.joblib` | Trained Logistic Regression model |
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `metrics.json` | Saved evaluation results shown in the app |
| `requirements.txt` | Python dependencies |

## Disclaimer

This application provides automated machine learning predictions based on patterns in its training dataset. Always verify news through reliable sources before accepting it as factual.
