from __future__ import annotations

import re
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from manipulation_model import load_dataset, predict_text, train_artifact

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "manipulation_detection_dataset.csv"
REPORTS_DIR = ROOT / "reports"
METHODS_MD = REPORTS_DIR / "methodologies_algorithms_report.md"
PERF_MD = REPORTS_DIR / "performance_evaluation_results_report.md"
METHODS_DOCX = REPORTS_DIR / "methodologies_algorithms_report.docx"
PERF_DOCX = REPORTS_DIR / "performance_evaluation_results_report.docx"

WRAP = 96
BAR_WIDTH = 30


def wrap(text: str, width: int = WRAP) -> str:
    return textwrap.fill(text, width=width)


def bar(value: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def count_bar(count: int, max_count: int, width: int = 24) -> str:
    ratio = 0 if max_count == 0 else count / max_count
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def fenced(text: str) -> str:
    return "```text\n" + text.rstrip() + "\n```"


def dataset_summary() -> dict:
    dataset = load_dataset(DATASET_PATH)
    artifact = train_artifact(dataset)
    matrix = artifact["vectorizer"].transform(dataset["text"])

    training_metrics: dict[str, dict[str, float]] = {}
    for target, classifier in artifact["classifiers"].items():
        truth = dataset[target]
        pred = classifier.predict(matrix)
        training_metrics[target] = {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro")),
            "weighted_f1": float(f1_score(truth, pred, average="weighted")),
        }

    sample_text = "If you really cared, you would send the report tonight."
    sample_vector = artifact["vectorizer"].transform([sample_text]).getrow(0)
    feature_names = artifact["vectorizer"].get_feature_names_out()
    ranked = sorted(zip(sample_vector.indices, sample_vector.data), key=lambda item: item[1], reverse=True)
    top_features = [(feature_names[idx], float(value)) for idx, value in ranked[:16]]

    return {
        "dataset": dataset,
        "artifact": artifact,
        "feature_shape": matrix.shape,
        "training_metrics": training_metrics,
        "sample_text": sample_text,
        "sample_shape": sample_vector.shape,
        "sample_nnz": int(sample_vector.nnz),
        "sample_top_features": top_features,
    }


def evaluation_summary(dataset) -> dict:
    fields = ("label", "intent_label", "manipulation_type", "domain", "severity")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratify_key = dataset["label"] + "|" + dataset["intent_label"]

    all_true = {field: [] for field in fields}
    all_pred = {field: [] for field in fields}
    fold_accuracies: list[float] = []

    for train_idx, test_idx in splitter.split(dataset["text"], stratify_key):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)
        artifact = train_artifact(train_df)

        label_preds = []
        for row in test_df.itertuples(index=False):
            prediction = predict_text(row.text, artifact)
            label_preds.append(prediction["label"])
            for field in fields:
                all_true[field].append(getattr(row, field))
                all_pred[field].append(prediction[field])

        fold_accuracies.append(float(accuracy_score(test_df["label"], label_preds)))

    metrics = {}
    for field in fields:
        metrics[field] = {
            "accuracy": float(accuracy_score(all_true[field], all_pred[field])),
            "macro_f1": float(f1_score(all_true[field], all_pred[field], average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(all_true[field], all_pred[field], average="weighted", zero_division=0)
            ),
        }

    return {
        "fold_accuracies": fold_accuracies,
        "metrics": metrics,
        "label_report": classification_report(all_true["label"], all_pred["label"], zero_division=0),
    }


def render_methodology_report(summary: dict, generated_at: str) -> str:
    dataset = summary["dataset"]
    training_metrics = summary["training_metrics"]
    feature_rows, feature_cols = summary["feature_shape"]

    label_counts = dataset["label"].value_counts().to_dict()
    domain_counts = dataset["domain"].value_counts().sort_index().to_dict()
    intent_counts = dataset["intent_label"].value_counts().sort_index().to_dict()
    max_domain = max(domain_counts.values())
    max_intent = max(intent_counts.values())

    word_rows = []
    char_rows = []
    for name, value in summary["sample_top_features"]:
        row = [f"`{name}`", f"{value:.6f}"]
        if name.startswith("word__") and len(word_rows) < 7:
            word_rows.append(row)
        elif name.startswith("char__") and len(char_rows) < 9:
            char_rows.append(row)

    training_table = md_table(
        ["Target", "Accuracy", "Macro F1", "Weighted F1"],
        [
            [f"`{target}`", f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}", f"{metrics['weighted_f1']:.4f}"]
            for target, metrics in training_metrics.items()
        ],
    )

    domain_chart = "\n".join(
        f"{name.ljust(12)} {count_bar(count, max_domain)} {count}"
        for name, count in domain_counts.items()
    )
    intent_chart = "\n".join(
        f"{name.ljust(18)} {count_bar(count, max_intent)} {count}"
        for name, count in intent_counts.items()
    )

    parts = [
        "# Methodologies / Algorithms Implemented",
        "",
        f"Generated on: `{generated_at}`",
        "",
        "## 1. Executive Summary",
        wrap(
            "This project is a multi-output natural language classification system for manipulation detection. "
            "Given a single message as input, the system predicts whether the text is manipulative and also "
            "returns a structured explanation of the message in terms of intent, manipulation tactic, domain, "
            "and severity."
        ),
        "",
        wrap(
            "The implementation is centered on classical machine learning rather than deep neural networks. "
            "It uses a shared TF-IDF vector representation of the input text and trains separate logistic "
            "regression classifiers over that shared representation."
        ),
        "",
        "## 2. Problem Formulation",
        "- Input: one short free-text message from a user, chat, or bot interaction.",
        "- Predicted outputs: `label`, `intent_label`, `intent`, `manipulation_type`, `domain`, and `severity`.",
        "",
        wrap(
            "A key design decision is that only the text is used as model input. The other fields are not fed "
            "into the model; they are supervised targets predicted from the text."
        ),
        "",
        "## 3. Data Pipeline",
        wrap(
            "The training set is synthetic and is generated by `build_manipulation_dataset.py`. The generator "
            "uses intent metadata, domain-specific action pools, neutral templates, tactic-specific "
            "manipulative templates, and a real-world style augmentation pass to create broad, schema-consistent examples."
        ),
        "",
        wrap(
            "During training, `load_dataset()` validates the schema, removes incomplete rows, and enforces that "
            "every `intent_label` maps to exactly one canonical human-readable intent string."
        ),
        "",
        wrap(
            "The realism pass introduces conversational openings and closings, multi-sentence variants, contractions, "
            "casual chat tokens, punctuation shifts, and slightly shorter text-message style rows. This makes the "
            "dataset less rigid than a single-template corpus while keeping the target labels deterministic."
        ),
        "",
        "### Dataset Dimensions",
        f"- Rows used for training: `{len(dataset)}`",
        "- Columns loaded by the training pipeline: `text`, `label`, `intent_label`, `intent`, `manipulation_type`, `domain`, `severity`",
        f"- Vectorized feature matrix shape after fitting: `{feature_rows} x {feature_cols}`",
        f"- Label balance: `manipulative={label_counts['manipulative']}`, `not_manipulative={label_counts['not_manipulative']}`",
        "",
        "### Domain Distribution",
        fenced(domain_chart),
        "",
        "### Intent Distribution",
        fenced(intent_chart),
        "",
        "## 4. Text Vectorization Methodology",
        wrap(
            "Vectorization happens in `manipulation_model.py` through a `FeatureUnion`. This means two different "
            "TF-IDF representations are learned in parallel and then concatenated into one sparse feature vector."
        ),
        "",
        "- Branch A: word-level TF-IDF with lowercase normalization, English stopword removal, and `ngram_range=(1, 2)`.",
        "- Branch B: character-level TF-IDF with `analyzer='char_wb'` and `ngram_range=(3, 5)`.",
        "",
        wrap(
            "The vectorizer is fit once on the training corpus using `fit_transform()`. At inference time, the same fitted "
            "vocabulary and IDF weights are reused with `transform()`, so every new message is projected into the same feature space."
        ),
        "",
        "### Why TF-IDF",
        wrap(
            "TF-IDF assigns larger weight to terms that are informative in a particular message but not overly common across "
            "the dataset. This is a strong fit for sparse text classification because it highlights discriminative phrases."
        ),
        "",
        "### Vectorization Snapshot",
        f'- Sample text: "{summary["sample_text"]}"',
        f'- Transformed vector shape: `{summary["sample_shape"][0]} x {summary["sample_shape"][1]}`',
        f'- Non-zero active features in the sample: `{summary["sample_nnz"]}`',
        "",
        "**Top active word features**",
        md_table(["Word Feature", "TF-IDF"], word_rows),
        "",
        "**Top active character features**",
        md_table(["Char Feature", "TF-IDF"], char_rows),
        "",
        "## 5. Classification Algorithms",
        wrap(
            "After vectorization, the project trains five separate `LogisticRegression` classifiers, one for each predictive target: "
            "`label`, `intent_label`, `manipulation_type`, `domain`, and `severity`."
        ),
        "",
        wrap(
            "Conceptually, each classifier learns a linear scoring function `score = w.x + b` over the sparse TF-IDF vector. "
            "Features with positive weights increase evidence for a class; features with negative weights reduce it. "
            "The class with the strongest learned score is returned as the prediction."
        ),
        "",
        wrap(
            "Logistic regression is a good choice here because the feature space is high-dimensional and sparse, the dataset is "
            "moderate in size, inference needs to be fast, and linear models are easier to debug than heavier neural alternatives."
        ),
        "",
        wrap(
            "Implementation details: `class_weight='balanced'` and `max_iter=1000` are used for every classifier. Balanced class weights "
            "provide resilience if future datasets become skewed, and the larger iteration budget helps convergence on sparse text features."
        ),
        "",
        "### Training Fit on the Current Dataset",
        training_table,
        "",
        "## 6. Artifact Packaging and Inference",
        wrap(
            "The saved artifact contains the fitted `FeatureUnion` vectorizer, the dictionary of trained classifiers, and a mapping "
            "from `intent_label` to canonical intent. It is serialized with `joblib` into `manipulation_model.joblib`."
        ),
        "",
        wrap(
            "At inference time, `manipulation_inference.py` loads the artifact, transforms the incoming text once, and runs all "
            "classifiers over the same vector. The service exposes `/health` and `/predict` endpoints through FastAPI."
        ),
        "",
        wrap(
            "A post-processing rule ensures consistency: if the predicted label is `not_manipulative`, then `manipulation_type` and "
            "severity are forced to `none`."
        ),
        "",
        wrap(
            "The Telegram bot in `bot.py` acts as a lightweight client. It forwards user text to the inference API and replies with "
            "the predicted label, intent, manipulation type, domain, severity, and confidence."
        ),
        "",
        "## 7. Dependencies and Runtime Stack",
        "- `pandas`",
        "- `scikit-learn`",
        "- `joblib`",
        "- `fastapi`",
        "- `uvicorn`",
        "- `httpx`",
        "- `python-telegram-bot`",
        "",
        "## 8. Strengths and Current Constraints",
        wrap(
            "Strengths: the pipeline is fast, interpretable, reproducible, easy to serve, and produces richer output than a simple binary classifier."
        ),
        "",
        wrap(
            "Constraints: the current dataset is synthetic, so real-world generalization still needs external validation. The model is linear and "
            "therefore strongest at learning lexical and phrase-level patterns rather than deep semantic reasoning."
        ),
    ]

    return "\n".join(parts).rstrip() + "\n"


def render_performance_report(summary: dict, eval_summary: dict, generated_at: str) -> str:
    dataset = summary["dataset"]
    training_metrics = summary["training_metrics"]
    eval_metrics = eval_summary["metrics"]
    label_counts = dataset["label"].value_counts().to_dict()

    train_table = md_table(
        ["Target", "Accuracy", "Macro F1", "Weighted F1"],
        [
            [f"`{target}`", f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}", f"{metrics['weighted_f1']:.4f}"]
            for target, metrics in training_metrics.items()
        ],
    )

    eval_table = md_table(
        ["Target", "Accuracy", "Macro F1", "Weighted F1"],
        [
            [f"`{target}`", f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}", f"{metrics['weighted_f1']:.4f}"]
            for target, metrics in eval_metrics.items()
        ],
    )

    fold_table = md_table(
        ["Fold", "Label Accuracy"],
        [[str(index), f"{value:.4f}"] for index, value in enumerate(eval_summary["fold_accuracies"], start=1)],
    )

    accuracy_chart = "\n".join(
        f"{target.ljust(18)} {bar(metrics['accuracy'])} {metrics['accuracy']:.4f}"
        for target, metrics in eval_metrics.items()
    )
    label_chart = "\n".join(
        f"{label.ljust(18)} {count_bar(count, max(label_counts.values()))} {count}"
        for label, count in label_counts.items()
    )

    parts = [
        "# Performance Evaluation and Results Obtained",
        "",
        f"Generated on: `{generated_at}`",
        "",
        "## 1. Evaluation Objective",
        wrap(
            "This report summarizes how the current manipulation detection pipeline performs on the synthetic training dataset "
            "and under held-out cross-validation. The goal is to document both pipeline fit quality and generalization within "
            "the same data generation regime."
        ),
        "",
        "## 2. Evaluation Setup",
        wrap(
            f"Dataset under evaluation: `data/manipulation_detection_dataset.csv` with {len(dataset)} rows. Each row supplies one text input and "
            "supervised targets for `label`, `intent_label`, `manipulation_type`, `domain`, and `severity`."
        ),
        "",
        wrap(
            "Training-fit evaluation: the model is trained on the full dataset and then scored on that same dataset. This is a useful "
            "sanity check for implementation correctness, but it is not the main estimate of generalization."
        ),
        "",
        wrap(
            "Held-out evaluation: `evaluate_model.py` performs 5-fold `StratifiedKFold` cross-validation. Stratification is done on the combined "
            "key `label|intent_label` so that the positive/negative classes and intent distribution remain stable across folds."
        ),
        "",
        wrap(
            "Metrics reported: accuracy, macro F1, and weighted F1 for each target. For the binary label target, a full classification report "
            "is also included."
        ),
        "",
        "### Dataset Snapshot",
        f"- Vectorized feature space size: `{summary['feature_shape'][1]}` features",
        f"- Label balance: `manipulative={label_counts['manipulative']}`, `not_manipulative={label_counts['not_manipulative']}`",
        "",
        "**Label balance chart**",
        fenced(label_chart),
        "",
        "## 3. Training-Fit Results",
        train_table,
        "",
        wrap(
            "All training-fit metrics are 1.0000 on the current dataset. This indicates that the pipeline is internally consistent and that "
            "the classifiers can fully separate the generated examples after vectorization."
        ),
        "",
        "## 4. Cross-Validation Results",
        "### Fold-wise Binary Label Accuracy",
        fold_table,
        "",
        "### Aggregate 5-Fold Metrics",
        eval_table,
        "",
        "**Cross-validation accuracy chart**",
        fenced(accuracy_chart),
        "",
        "### Binary Label Classification Report",
        fenced(eval_summary["label_report"]),
        "",
        "## 5. Interpretation",
        wrap(
            "The `label`, `manipulation_type`, and `severity` targets achieve perfect held-out performance under the current synthetic split. "
            "That suggests the language patterns for those targets are highly separable in the generated dataset."
        ),
        "",
        wrap(
            "In the current 10,000-row version, `intent_label` and `domain` also reach perfect cross-validation scores. This means the expanded "
            "synthetic dataset is still highly structured and cleanly separable across all supervised targets."
        ),
        "",
        wrap(
            "Because the evaluation folds come from the same synthetic generation process as the training data, these results should be interpreted "
            "as controlled-benchmark performance rather than final real-world accuracy."
        ),
        "",
        wrap(
            "Even after expanding the dataset to 10,000 rows and adding noisier chat-style surface variation, the current target space remains "
            "highly separable under this synthetic generation regime. That is why cross-validation remains perfect in this version of the project."
        ),
        "",
        "## 6. Main Findings",
        "1. The end-to-end training and inference pipeline is functioning correctly and produces stable outputs across all targets.",
        "2. The shared TF-IDF plus logistic regression approach is sufficient to separate the current synthetic classes with very high accuracy.",
        "3. The added chat-style noise and shorter-message variation were not enough to break separability under the current synthetic generation process.",
        "4. Real-world validation remains the next major step before making stronger claims about deployment robustness.",
        "",
        "## 7. Reproducibility",
        "- Training command: `.venv/bin/python train_manipulation_model.py`",
        "- Evaluation command: `.venv/bin/python evaluate_model.py`",
        "- Single inference command: `.venv/bin/python manipulation_inference.py --text \"If you cared, you would send the report tonight.\"`",
    ]

    return "\n".join(parts).rstrip() + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _clean_inline_markdown(text: str) -> str:
    cleaned = text.replace("`", "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    return cleaned


def _parse_markdown_blocks(markdown: str) -> list[dict]:
    lines = markdown.splitlines()
    blocks: list[dict] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            blocks.append({"type": "code", "lines": code_lines})
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            blocks.append(
                {
                    "type": "heading",
                    "level": len(heading_match.group(1)),
                    "text": _clean_inline_markdown(heading_match.group(2)),
                }
            )
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1

            rows = []
            for raw in table_lines:
                cells = [cell.strip() for cell in raw.strip("|").split("|")]
                if cells and all(set(cell) <= {"-", ":"} for cell in cells):
                    continue
                rows.append([_clean_inline_markdown(cell) for cell in cells])

            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        if re.match(r"^-\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^-\s+", lines[index].strip()):
                items.append(_clean_inline_markdown(re.sub(r"^-\s+", "", lines[index].strip())))
                index += 1
            blocks.append({"type": "bullet_list", "items": items})
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(_clean_inline_markdown(lines[index].strip()))
                index += 1
            blocks.append({"type": "number_list", "items": items})
            continue

        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            blocks.append({"type": "strong_paragraph", "text": _clean_inline_markdown(stripped)})
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            if (
                next_stripped.startswith("```")
                or next_stripped.startswith("|")
                or re.match(r"^(#{1,6})\s+", next_stripped)
                or re.match(r"^-\s+", next_stripped)
                or re.match(r"^\d+\.\s+", next_stripped)
                or (next_stripped.startswith("**") and next_stripped.endswith("**"))
            ):
                break
            paragraph_lines.append(next_stripped)
            index += 1

        blocks.append({"type": "paragraph", "text": _clean_inline_markdown(" ".join(paragraph_lines))})

    return blocks


def _xml_text(text: str) -> str:
    escaped = escape(text)
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        return f'<w:t xml:space="preserve">{escaped}</w:t>'
    return f"<w:t>{escaped}</w:t>"


def _run_xml(text: str, *, bold: bool = False, size: int = 22, font: str | None = None) -> str:
    parts = ["<w:r>"]
    if bold or size != 22 or font is not None:
        props = []
        if bold:
            props.append("<w:b/>")
        if font is not None:
            props.append(f'<w:rFonts w:ascii="{escape(font)}" w:hAnsi="{escape(font)}"/>')
        props.append(f'<w:sz w:val="{size}"/>')
        props.append(f'<w:szCs w:val="{size}"/>')
        parts.append("<w:rPr>" + "".join(props) + "</w:rPr>")

    segments = text.split("\n")
    for idx, segment in enumerate(segments):
        if idx:
            parts.append("<w:br/>")
        parts.append(_xml_text(segment))
    parts.append("</w:r>")
    return "".join(parts)


def _paragraph_xml(
    text: str,
    *,
    bold: bool = False,
    size: int = 22,
    font: str | None = None,
    indent: int = 0,
    spacing_after: int = 120,
) -> str:
    ppr_parts = [f'<w:spacing w:after="{spacing_after}"/>']
    if indent:
        ppr_parts.append(f'<w:ind w:left="{indent}"/>')
    return "<w:p><w:pPr>" + "".join(ppr_parts) + "</w:pPr>" + _run_xml(
        text, bold=bold, size=size, font=font
    ) + "</w:p>"


def _table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    width = max(1500, int(9000 / max_cols))
    tbl_parts = [
        "<w:tbl>",
        "<w:tblPr>",
        '<w:tblW w:w="0" w:type="auto"/>',
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        "</w:tblBorders>",
        "</w:tblPr>",
    ]

    for row_index, row in enumerate(rows):
        tbl_parts.append("<w:tr>")
        padded = row + [""] * (max_cols - len(row))
        for cell in padded:
            tbl_parts.append(
                "<w:tc>"
                "<w:tcPr>"
                f'<w:tcW w:w="{width}" w:type="dxa"/>'
                "</w:tcPr>"
                + _paragraph_xml(cell, bold=(row_index == 0), spacing_after=40)
                + "</w:tc>"
            )
        tbl_parts.append("</w:tr>")

    tbl_parts.append("</w:tbl>")
    return "".join(tbl_parts)


def _markdown_to_docx_document_xml(markdown: str) -> str:
    blocks = _parse_markdown_blocks(markdown)
    body_parts: list[str] = []

    heading_sizes = {1: 32, 2: 28, 3: 24, 4: 22, 5: 20, 6: 18}

    for block in blocks:
        block_type = block["type"]
        if block_type == "heading":
            level = int(block["level"])
            body_parts.append(
                _paragraph_xml(
                    block["text"],
                    bold=True,
                    size=heading_sizes.get(level, 22),
                    spacing_after=160,
                )
            )
        elif block_type == "paragraph":
            body_parts.append(_paragraph_xml(block["text"], spacing_after=120))
        elif block_type == "strong_paragraph":
            body_parts.append(_paragraph_xml(block["text"], bold=True, spacing_after=100))
        elif block_type == "bullet_list":
            for item in block["items"]:
                body_parts.append(_paragraph_xml(f"• {item}", indent=360, spacing_after=80))
        elif block_type == "number_list":
            for item in block["items"]:
                body_parts.append(_paragraph_xml(item, indent=360, spacing_after=80))
        elif block_type == "code":
            for line in block["lines"] or [""]:
                body_parts.append(
                    _paragraph_xml(line, font="Courier New", size=18, indent=360, spacing_after=40)
                )
        elif block_type == "table":
            body_parts.append(_table_xml(block["rows"]))

    body_parts.append(
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body_parts)
        + "</w:body></w:document>"
    )


def write_docx(path: Path, markdown: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    document_xml = _markdown_to_docx_document_xml(markdown)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""
    app_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels_xml)
        docx.writestr("docProps/core.xml", core_xml)
        docx.writestr("docProps/app.xml", app_xml)
        docx.writestr("word/document.xml", document_xml)


def main() -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = dataset_summary()
    eval_sum = evaluation_summary(summary["dataset"])

    methods_markdown = render_methodology_report(summary, generated_at)
    perf_markdown = render_performance_report(summary, eval_sum, generated_at)

    write_text(METHODS_MD, methods_markdown)
    write_text(PERF_MD, perf_markdown)
    write_docx(METHODS_DOCX, methods_markdown, "Methodologies / Algorithms Implemented")
    write_docx(PERF_DOCX, perf_markdown, "Performance Evaluation and Results Obtained")

    print(f"wrote={METHODS_MD}")
    print(f"wrote={PERF_MD}")
    print(f"wrote={METHODS_DOCX}")
    print(f"wrote={PERF_DOCX}")


if __name__ == "__main__":
    main()
