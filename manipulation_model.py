from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

DATASET_COLUMNS = (
    "text",
    "label",
    "intent_label",
    "intent",
    "manipulation_type",
    "domain",
    "severity",
)
CLASSIFIER_TARGETS = ("label", "intent_label", "manipulation_type", "domain", "severity")
TRIVIAL_GREETING_PATTERNS = [
    re.compile(r"^(hi|hey|hello|hii+|heyy+)(\s+(guys|everyone|all|there))?[!.?]*$", re.IGNORECASE),
    re.compile(r"^(hi|hey|hello).*(who are you|who'?s this)[!.?]*$", re.IGNORECASE),
    re.compile(r"^(who are you|who'?s this|what is this)[!.?]*$", re.IGNORECASE),
]
SHORT_BENIGN_REQUEST_PATTERNS = [
    re.compile(r"^(can|could|would)\s+you\b", re.IGNORECASE),
    re.compile(r"^(please|pls)\b", re.IGNORECASE),
    re.compile(r"^(let me know|tell me)\b", re.IGNORECASE),
]
MANIPULATIVE_CUE_PATTERNS = [
    re.compile(r"\bif you (really|actually) cared\b", re.IGNORECASE),
    re.compile(r"\byou owe me\b", re.IGNORECASE),
    re.compile(r"\beveryone else\b", re.IGNORECASE),
    re.compile(r"\blast chance\b", re.IGNORECASE),
    re.compile(r"\bprove yourself\b", re.IGNORECASE),
    re.compile(r"\bdo not make me\b", re.IGNORECASE),
    re.compile(r"\bif you do not\b", re.IGNORECASE),
    re.compile(r"\bor else\b", re.IGNORECASE),
    re.compile(r"\bi will escalate\b", re.IGNORECASE),
    re.compile(r"\bcomfortable being blamed\b", re.IGNORECASE),
    re.compile(r"\bnot on our side\b", re.IGNORECASE),
    re.compile(r"\bthe least you can do\b", re.IGNORECASE),
]
LOW_CONFIDENCE_MANIPULATIVE_THRESHOLD = 0.85


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _has_manipulative_cues(text: str) -> bool:
    return any(pattern.search(text) for pattern in MANIPULATIVE_CUE_PATTERNS)


def _is_trivial_greeting(text: str) -> bool:
    normalized = text.strip()
    tokens = _tokenize(normalized)
    if len(tokens) <= 7 and any(pattern.match(normalized) for pattern in TRIVIAL_GREETING_PATTERNS):
        return True
    return False


def _is_short_benign_request(text: str) -> bool:
    normalized = text.strip()
    tokens = _tokenize(normalized)
    if len(tokens) > 8:
        return False
    if _has_manipulative_cues(normalized):
        return False
    return any(pattern.match(normalized) for pattern in SHORT_BENIGN_REQUEST_PATTERNS)


def _trivial_message_response() -> dict[str, Any]:
    return {
        "label": "not_manipulative",
        "is_manipulative": False,
        "intent_label": "small_talk",
        "intent": "Casual greeting or trivial message",
        "manipulation_type": "none",
        "domain": "unknown",
        "severity": "none",
        "confidence": None,
        "intent_confidence": None,
    }


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    missing = [column for column in DATASET_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataset = dataset[list(DATASET_COLUMNS)].dropna().reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Dataset is empty after dropping missing rows")

    intent_map = dataset.groupby("intent_label")["intent"].nunique()
    if (intent_map > 1).any():
        raise ValueError("Each intent_label must map to exactly one canonical intent")
    return dataset


def train_artifact(dataset: pd.DataFrame) -> dict[str, Any]:
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    train_matrix = vectorizer.fit_transform(dataset["text"])

    classifiers: dict[str, LogisticRegression] = {}
    for target in CLASSIFIER_TARGETS:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(train_matrix, dataset[target])
        classifiers[target] = classifier

    return {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "intent_map": dict(
            dataset[["intent_label", "intent"]].drop_duplicates().itertuples(index=False, name=None)
        ),
    }


def save_artifact(artifact: dict[str, Any], output_path: str | Path) -> None:
    joblib.dump(artifact, output_path)


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    return joblib.load(model_path)


def _prediction_confidence(classifier: LogisticRegression, vector, predicted_label: str) -> float | None:
    if not hasattr(classifier, "predict_proba"):
        return None

    probabilities = classifier.predict_proba(vector)[0]
    classes = list(classifier.classes_)
    if predicted_label not in classes:
        return None
    return float(probabilities[classes.index(predicted_label)])


def predict_text(text: str, artifact: dict[str, Any]) -> dict[str, Any]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty")

    if _is_trivial_greeting(cleaned_text):
        return _trivial_message_response()

    vectorizer = artifact["vectorizer"]
    vector = vectorizer.transform([cleaned_text])

    predictions: dict[str, str] = {}
    confidences: dict[str, float | None] = {}
    for target, classifier in artifact["classifiers"].items():
        predicted_value = str(classifier.predict(vector)[0])
        predictions[target] = predicted_value
        confidences[target] = _prediction_confidence(classifier, vector, predicted_value)

    if predictions["label"] == "manipulative":
        if predictions["manipulation_type"] == "none" or predictions["severity"] == "none":
            predictions["label"] = "not_manipulative"
        elif (
            confidences["label"] is not None
            and confidences["label"] < LOW_CONFIDENCE_MANIPULATIVE_THRESHOLD
            and not _has_manipulative_cues(cleaned_text)
        ):
            predictions["label"] = "not_manipulative"
        elif _is_short_benign_request(cleaned_text) and not _has_manipulative_cues(cleaned_text):
            predictions["label"] = "not_manipulative"

    if predictions["label"] == "not_manipulative":
        predictions["manipulation_type"] = "none"
        predictions["severity"] = "none"

    intent_label = predictions["intent_label"]
    intent = artifact["intent_map"][intent_label]

    return {
        "label": predictions["label"],
        "is_manipulative": predictions["label"] == "manipulative",
        "intent_label": intent_label,
        "intent": intent,
        "manipulation_type": predictions["manipulation_type"],
        "domain": predictions["domain"],
        "severity": predictions["severity"],
        "confidence": confidences["label"],
        "intent_confidence": confidences["intent_label"],
    }
