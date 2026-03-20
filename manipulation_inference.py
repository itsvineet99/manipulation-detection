import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from manipulation_model import load_artifact, predict_text

MODEL_PATH = Path(os.getenv("MANIPULATION_MODEL_PATH", "manipulation_model.joblib"))
model_artifact = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    is_manipulative: bool
    intent_label: str
    intent: str
    manipulation_type: str
    domain: str
    severity: str
    confidence: Optional[float] = None
    intent_confidence: Optional[float] = None


def get_model():
    global model_artifact
    if model_artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run train_manipulation_model.py first."
            )
        model_artifact = load_artifact(MODEL_PATH)
    return model_artifact


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    get_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        prediction = predict_text(req.text, get_model())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(**prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Run a single local prediction and print JSON output.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    if args.text:
        print(json.dumps(predict_text(args.text, get_model()), indent=2))
        return

    import uvicorn

    uvicorn.run("manipulation_inference:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
