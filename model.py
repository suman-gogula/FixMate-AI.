"""
model.py - Machine Learning part of FixMate AI

Pipeline:  cleaned text --> TF-IDF vectors --> Logistic Regression --> category

Also contains analyze_complaint(), which runs the COMPLETE FixMate pipeline:
    preprocess -> ML category -> rule-based urgency -> department -> action
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess, preprocess_steps
from rules import (CATEGORIES, assign_department, detect_urgency,
                   recommend_action, safety_note)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
LOW_CONFIDENCE = 0.40


def load_dataset() -> pd.DataFrame:
    """Read dataset.csv and add a 'clean' column with preprocessed text."""
    df = pd.read_csv(DATASET_PATH)
    df["clean"] = df["complaint"].apply(preprocess)
    return df


def train_model(test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Train and evaluate the classifier.
    Accuracy and the confusion matrix are computed on the held-out TEST set,
    which the model never saw during training.
    """
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["category"],
        test_size=test_size, random_state=seed, stratify=df["category"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),   # words + word pairs
        ("clf", LogisticRegression(C=10, max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=CATEGORIES,
                                   output_dict=True, zero_division=0)
    report_df = (pd.DataFrame(report).T.round(2)
                 .rename(columns={"f1-score": "f1", "support": "samples"}))

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=CATEGORIES),
        "report_df": report_df,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_counts": df["category"].value_counts().reindex(CATEGORIES),
    }


def predict_category(bundle: dict, clean_text: str):
    """Return (category, confidence, {category: probability})."""
    pipe = bundle["pipeline"]
    probs = pipe.predict_proba([clean_text])[0]
    idx = int(np.argmax(probs))
    label = pipe.classes_[idx]
    return label, float(probs[idx]), {c: float(p) for c, p in zip(pipe.classes_, probs)}


def analyze_complaint(bundle: dict, text: str) -> dict:
    """Full FixMate pipeline for ONE complaint. Returns everything the UI shows."""
    steps = preprocess_steps(text)
    category, confidence, probs = predict_category(bundle, steps["clean_text"])
    urgency = detect_urgency(text, category)
    return {
        "original": text.strip(),
        "steps": steps,
        "clean": steps["clean_text"],
        "category": category,
        "confidence": confidence,
        "probabilities": probs,
        "low_confidence": confidence < LOW_CONFIDENCE,
        "urgency": urgency["level"],
        "urgency_reason": urgency["reason"],
        "department": assign_department(category),
        "action": recommend_action(category, urgency["level"], text),
        "safety": safety_note(category, urgency["level"]),
    }


if __name__ == "__main__":
    b = train_model()
    print(f"Dataset: {b['n_total']} complaints  (train {b['n_train']}, test {b['n_test']})")
    print(f"Test accuracy: {b['accuracy']:.2%}")
    print(b["report_df"])
    for s in ["Water is leaking continuously from my bathroom ceiling.",
              "The light bulb in my bedroom is not working."]:
        r = analyze_complaint(b, s)
        print("\n", s)
        for k in ["clean", "category", "confidence", "urgency", "urgency_reason", "department", "action"]:
            print(f"   {k:15}: {r[k]}")
