"""
STEP D — Predict AI vs Human on New Resumes
=============================================
Loads fine-tuned BanglaBERT from bangla_ai_detector/,
runs prediction on all .txt files in prediction_resumes/,
saves results to ai_human_predictions.csv

Folder structure:
    your_project/
    ├── bangla_ai_detector/        ← downloaded from Drive
    ├── prediction_resumes/        ← new resumes to classify
    └── stepD_predict_ai_human.py
"""

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR    = "bangla_ai_detector"
PRED_FOLDER  = "prediction_resumes"
OUTPUT_CSV   = "ai_human_predictions.csv"
MAX_LENGTH   = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading BanglaBERT classifier from '{MODEL_DIR}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.\n")


def extract_text(file_path: str) -> str:
    """Strip PII block and annotation tags, return clean resume text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\[Info_Start\].*?\[Info_End\]", "", content, flags=re.DOTALL)
    content = re.sub(r"\[section\]", "", content)
    content = re.sub(r"\[(Objective|Skill|Experience|Expericence|Education)\]", "", content)
    content = re.sub(r"\s+", " ", content).strip()

    return content


def predict(text: str) -> dict:
    """Run model inference, return label and confidence."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1)[0]
    pred_id    = torch.argmax(probs).item()
    label      = model.config.id2label[pred_id]
    confidence = round(probs[pred_id].item() * 100, 2)

    return {
        "Label":            label,
        "Confidence (%)":   confidence,
        "P(Human) (%)":     round(probs[0].item() * 100, 2),
        "P(AI) (%)":        round(probs[1].item() * 100, 2),
    }


# ─── EXECUTION ────────────────────────────────────────────────────────────────
if not os.path.exists(PRED_FOLDER):
    print(f"Error: '{PRED_FOLDER}' folder not found.")
    exit(1)

files = sorted([f for f in os.listdir(PRED_FOLDER) if f.endswith(".txt")])
print(f"Predicting {len(files)} resumes...\n")

rows = []
for filename in files:
    try:
        text   = extract_text(os.path.join(PRED_FOLDER, filename))
        result = predict(text)
        rows.append({"Filename": filename, **result})
        print(f"  {filename:30s} → {result['Label']:5s}  ({result['Confidence (%)']:.1f}%)")
    except Exception as e:
        print(f"  ✗ {filename}: {e}")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n{'─'*50}")
print(f"Results saved → '{OUTPUT_CSV}'")
print(f"\nSummary:")
print(df["Label"].value_counts().to_string())
