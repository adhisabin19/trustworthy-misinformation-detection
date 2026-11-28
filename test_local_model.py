import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parent

DEPLOYMENT = ROOT / "TrustFact_Model" / "TrustFact_Model"

MODEL_PATH = DEPLOYMENT / "model"
CONFIG_PATH = DEPLOYMENT / "final_model_config.json"

assert MODEL_PATH.exists(), "Model folder missing"
assert CONFIG_PATH.exists(), "Configuration missing"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LABELS = config["labels"]
TEMPERATURE = config["temperature"]
THRESHOLD = config["confidence_threshold"]
MAX_LENGTH = config["max_length"]

print("Loading TrustFact model...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")
print("Temperature:", TEMPERATURE)
print("Confidence threshold:", THRESHOLD)

claim = "The United States has 50 states."

inputs = tokenizer(
    claim,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH
)

with torch.inference_mode():
    logits = model(**inputs).logits

    probabilities = torch.softmax(
        logits / TEMPERATURE,
        dim=1
    )[0]

confidence, predicted_id = torch.max(
    probabilities,
    dim=0
)

confidence = confidence.item()
predicted_label = LABELS[predicted_id.item()]

print("\n" + "=" * 50)
print("TRUSTFACT LOCAL PREDICTION")
print("=" * 50)

print("Statement:", claim)
print("Predicted label:", predicted_label)
print(f"Calibrated confidence: {confidence:.2%}")

if confidence >= THRESHOLD:
    print("Decision:", predicted_label)
else:
    print("Decision: Insufficient confidence")

print("\nCLASS PROBABILITIES")

for label, probability in zip(LABELS, probabilities):
    print(f"{label:15s}: {probability.item():.2%}")

print("\nLocal inference completed successfully.")