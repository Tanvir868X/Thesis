"""
Generate Abstractive Summaries using Fine-tuned BanglaT5
===================================================================
Loads bangla_resume_summarizer_banglat5/ and runs inference on all resumes.
Output: BanglaT5b_Abstractive_Summaries.csv

Folder structure:
    your_project_folder/
    ├── bangla_resume_summarizer_banglat5/
    ├── raw_resumes/
    └── abstractive_summarize_banglat5.py
"""

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "bangla_resume_summarizer_banglat5"
RAW_FOLDER  = "raw_resumes"
OUTPUT_PATH = "BanglaT5_Abstractive_Summaries.csv"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading fine-tuned BanglaT5 from '{MODEL_DIR}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.\n")


def extract_sections(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\[Info_Start\].*?\[Info_End\]", "", content, flags=re.DOTALL)
    content = re.sub(r"\[section\]", "", content)

    pattern = r"([^\[\n।]+)[\s।]*\[(Objective|Skill|Experience|Expericence|Education)\]"
    matches = re.findall(pattern, content)

    data = {"Objective": [], "Experience": [], "Education": [], "Skill": []}
    for text, label in matches:
        category = "Experience" if "Exp" in label else label
        if category in data:
            clean = text.strip().replace("।", "").strip()
            if len(clean) > 3:
                data[category].append(clean)

    return data


def generate_summary(sentences: list) -> str:
    if not sentences:
        return "তথ্য নেই।"

    # BanglaT5 — no language token needed
    input_text = " । ".join(sentences[:4])

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            min_length=15,
            num_beams=4,
            length_penalty=1.2,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_resume(file_path: str) -> str:
    sections = extract_sections(file_path)

    summary_parts = []
    for cat in ["Objective", "Experience", "Education", "Skill"]:
        sentences = sections.get(cat, [])
        generated = generate_summary(sentences) if sentences else "তথ্য নেই।"
        summary_parts.append(f"{cat.upper()}: {generated}")

    return "\n\n".join(summary_parts)


# ─── EXECUTION ────────────────────────────────────────────────────────────────
if not os.path.exists(RAW_FOLDER):
    print(f"Error: '{RAW_FOLDER}' not found.")
    exit(1)

if not os.path.exists(MODEL_DIR):
    print(f"Error: '{MODEL_DIR}' not found. Run Step 2B first.")
    exit(1)

files = sorted([f for f in os.listdir(RAW_FOLDER) if f.endswith(".txt")])
print(f"Generating summaries for {len(files)} resumes...\n")

final_data = []
for filename in files:
    try:
        summary = process_resume(os.path.join(RAW_FOLDER, filename))
        final_data.append({"Filename": filename, "Abstractive_Summary": summary})
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename}: {e}")

pd.DataFrame(final_data).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nDone: {len(final_data)} summaries → '{OUTPUT_PATH}'")
