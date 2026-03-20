import os
from pathlib import Path

from manipulation_model import load_dataset, save_artifact, train_artifact

DATASET_PATH = Path(os.getenv("MANIPULATION_DATASET", "data/manipulation_detection_dataset.csv"))
MODEL_PATH = Path(os.getenv("MANIPULATION_MODEL_PATH", "manipulation_model.joblib"))


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    artifact = train_artifact(dataset)
    save_artifact(artifact, MODEL_PATH)

    print(f"trained_rows={len(dataset)}")
    print(f"dataset_path={DATASET_PATH}")
    print(f"model_path={MODEL_PATH}")


if __name__ == "__main__":
    main()
