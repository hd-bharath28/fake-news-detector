"""
Train a simple TF-IDF + LogisticRegression pipeline and save to models/fake_news_detector.joblib

Usage:
    python train_model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from utils import clean_text
import config
import nltk

# download necessary nltk data
nltk.download('stopwords')
nltk.download('punkt')

DATA_DIR = "data"
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")

def load_and_prepare():
    # Expect True.csv and Fake.csv (Kaggle dataset format)
    if not (os.path.exists(TRUE_PATH) and os.path.exists(FAKE_PATH)):
        raise FileNotFoundError(
            "Place True.csv and Fake.csv in the data/ directory. See README for dataset link."
        )

    df_true = pd.read_csv(TRUE_PATH)
    df_fake = pd.read_csv(FAKE_PATH)

    # Some files have columns: title, text
    df_true["label"] = 0  # real
    df_fake["label"] = 1  # fake

    # Compose content
    df_true["content"] = (df_true.get("title", "").fillna("") + " " + df_true.get("text", "").fillna("")).str.strip()
    df_fake["content"] = (df_fake.get("title", "").fillna("") + " " + df_fake.get("text", "").fillna("")).str.strip()

    df = pd.concat([df_true[["content", "label"]], df_fake[["content", "label"]]], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["content"] = df["content"].astype(str).apply(clean_text)
    return df

def train():
    df = load_and_prepare()
    X = df["content"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=3)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"Saved trained pipeline to {config.MODEL_PATH}")

if __name__ == "__main__":
    train()