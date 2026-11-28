import json
import math
import torch

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DEPLOYMENT = (
    ROOT / "TrustFact_Model" / "TrustFact_Model"
)

MODEL_PATH = DEPLOYMENT / "model"

CONFIG_PATH = DEPLOYMENT / "final_model_config.json"

FRONTEND_PATH = (
    Path(__file__).resolve().parent
    / "trustfact_frontend"
    / "TrustFact_Frontend.html"
)

# --------------------------------------------------
# 2. Load frozen model configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

LABELS = config["labels"]
TEMPERATURE = config["temperature"]
THRESHOLD = config["confidence_threshold"]
MAX_LENGTH = config["max_length"]

DEVICE = torch.device("cpu")

# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

print("Loading TrustFact model...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.to(DEVICE)
model.eval()

assert model.config.num_labels == len(LABELS)
assert [
    model.config.id2label[i]
    for i in range(len(LABELS))
] == LABELS

assert TEMPERATURE > 0
assert 0 <= THRESHOLD <= 1

print("TrustFact model loaded successfully.")
print("Temperature:", TEMPERATURE)
print("Confidence threshold:", THRESHOLD)

# --------------------------------------------------
# 4. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="TrustFact API",
    description="Experimental calibrated misinformation classifier",
    version="1.0.0"
)

class CheckRequest(BaseModel):
    text: str | None = None
    url: str | None = None

# --------------------------------------------------
# 5. Prediction function
# --------------------------------------------------

def predict_claim(claim):

    inputs = tokenizer(
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.inference_mode():

        logits = model(**inputs).logits

        probabilities = torch.softmax(
            logits / TEMPERATURE,
            dim=1
        )[0]

    probabilities = probabilities.cpu().tolist()

    predicted_id = max(
        range(len(probabilities)),
        key=lambda i: probabilities[i]
    )

    predicted_label = LABELS[predicted_id]

    confidence = probabilities[predicted_id]

    sorted_probs = sorted(
        probabilities,
        reverse=True
    )

    margin = sorted_probs[0] - sorted_probs[1]

    entropy = -sum(
        p * math.log(max(p, 1e-12))
        for p in probabilities
    ) / math.log(len(LABELS))

    accepted = confidence >= THRESHOLD

    return {
        "claim": claim,
        "model_name": config["model"],
        "predicted_label": predicted_label,
        "calibrated_confidence": confidence,
        "confidence_threshold": THRESHOLD,
        "temperature": TEMPERATURE,
        "accepted": accepted,

        "decision": (
            predicted_label
            if accepted
            else "Insufficient confidence"
        ),

        "probabilities": {
            label: probability
            for label, probability
            in zip(LABELS, probabilities)
        },

        "uncertainty": {
            "normalized_entropy": entropy,
            "prediction_margin": margin,
            "uncertainty_score": 1 - confidence
        },

        "evidence_verified": False,

        "warning": (
            "Experimental model prediction. "
            "No independent factual evidence was retrieved."
        )
    }

# --------------------------------------------------
# 6. API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": config["model"],
        "device": str(DEVICE),
        "temperature": TEMPERATURE,
        "confidence_threshold": THRESHOLD,
        "model_loaded": True
    }


@app.post("/check")
def check_claim(request: CheckRequest):

    claim = (request.text or "").strip()

    if not claim:

        if request.url:
            raise HTTPException(
                status_code=400,
                detail="URL verification is not implemented."
            )

        raise HTTPException(
            status_code=400,
            detail="Please enter a statement."
        )

    if len(claim) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Statement is too long."
        )

    return predict_claim(claim)
# --------------------------------------------------
# 7. Live SHAP and LIME explainability
# --------------------------------------------------

import numpy as np
import threading

EXPLANATION_LOCK = threading.Lock()


def predict_calibrated_batch(texts):

    texts = [str(text) for text in texts]

    results = []

    for start in range(0, len(texts), 16):

        batch = texts[start:start + 16]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.inference_mode():

            logits = model(**inputs).logits

            probabilities = torch.softmax(
                logits / TEMPERATURE,
                dim=1
            )

        results.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(results, axis=0)


class ExplainRequest(BaseModel):
    text: str
    method: str = "lime"


@app.post("/explain")
def explain_claim(request: ExplainRequest):

    claim = request.text.strip()
    method = request.method.lower().strip()

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Please enter a statement."
        )

    if method not in ["shap", "lime"]:
        raise HTTPException(
            status_code=400,
            detail="Method must be shap or lime."
        )

    token_count = len(
        tokenizer(claim)["input_ids"]
    )

    if token_count > MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Statement exceeds explanation token limit."
        )

    if not EXPLANATION_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another explanation is running."
        )

    try:

        prediction = predict_claim(claim)

        predicted_label = prediction["predicted_label"]

        predicted_id = LABELS.index(predicted_label)

        features = []

        # ----------------------------
        # LIME explanation
        # ----------------------------

        if method == "lime":

            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(
                class_names=LABELS,
                random_state=42
            )

            explanation = explainer.explain_instance(
                claim,
                predict_calibrated_batch,
                labels=[predicted_id],
                num_features=10,
                num_samples=500
            )

            for word, weight in explanation.as_list(
                label=predicted_id
            ):

                features.append({
                    "token": word,
                    "weight": float(weight)
                })

        # ----------------------------
        # SHAP explanation
        # ----------------------------

        else:

            import shap

            masker = shap.maskers.Text(tokenizer)

            explainer = shap.Explainer(
                predict_calibrated_batch,
                masker,
                output_names=LABELS,
                algorithm="partition"
            )

            explanation = explainer(
                [claim],
                max_evals=500,
                batch_size=16
            )

            tokens = explanation.data[0]

            values = explanation.values[
                0, :, predicted_id
            ]

            for token, value in zip(tokens, values):

                token = str(token).strip()

                if token:

                    features.append({
                        "token": token,
                        "weight": float(value)
                    })

        features = sorted(
            features,
            key=lambda item: abs(item["weight"]),
            reverse=True
        )[:10]

        for feature in features:

            weight = feature["weight"]

            feature["direction"] = (
                "supports" if weight > 0
                else "opposes" if weight < 0
                else "neutral"
            )

        return {
            "claim": claim,
            "method": method.upper(),
            "predicted_label": predicted_label,
            "calibrated_confidence":
                prediction["calibrated_confidence"],
            "accepted": prediction["accepted"],
            "features": features,
            "warning": (
                "Explanations describe model behavior, "
                "not factual evidence."
            )
        }

    finally:

        EXPLANATION_LOCK.release()