import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from manipulation_model import load_dataset, predict_text, train_artifact

EVAL_FIELDS = ("label", "intent_label", "manipulation_type", "domain", "severity")


def evaluate(dataset_path: Path, folds: int, random_state: int) -> None:
    dataset = load_dataset(dataset_path)
    stratify_key = dataset["label"] + "|" + dataset["intent_label"]

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    all_true = {field: [] for field in EVAL_FIELDS}
    all_pred = {field: [] for field in EVAL_FIELDS}

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(dataset["text"], stratify_key), start=1):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)
        artifact = train_artifact(train_df)

        for row in test_df.itertuples(index=False):
            prediction = predict_text(row.text, artifact)
            for field in EVAL_FIELDS:
                all_true[field].append(getattr(row, field))
                all_pred[field].append(prediction[field])

        fold_label_accuracy = accuracy_score(
            test_df["label"],
            [predict_text(text, artifact)["label"] for text in test_df["text"]],
        )
        print(f"fold_{fold_index}_label_accuracy={fold_label_accuracy:.4f}")

    print(f"dataset_path={dataset_path}")
    print(f"rows={len(dataset)}")
    print(f"folds={folds}")

    for field in EVAL_FIELDS:
        accuracy = accuracy_score(all_true[field], all_pred[field])
        macro_f1 = f1_score(all_true[field], all_pred[field], average="macro", zero_division=0)
        weighted_f1 = f1_score(all_true[field], all_pred[field], average="weighted", zero_division=0)
        print(f"{field}_accuracy={accuracy:.4f}")
        print(f"{field}_macro_f1={macro_f1:.4f}")
        print(f"{field}_weighted_f1={weighted_f1:.4f}")

    print("label_report_start")
    print(classification_report(all_true["label"], all_pred["label"], zero_division=0))
    print("label_report_end")
    print("note=intent is now evaluated through the closed-set intent_label target.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/manipulation_detection_dataset.csv",
        help="Path to the manipulation dataset CSV.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold shuffling.",
    )
    args = parser.parse_args()

    evaluate(Path(args.dataset), args.folds, args.seed)


if __name__ == "__main__":
    main()
