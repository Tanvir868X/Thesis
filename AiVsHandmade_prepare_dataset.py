"""
Prepare AI vs Human Dataset
======================================
Reads from:
    ai_resumes/          → label 1 (AI-written)
    humanwritten_resumes/ → label 0 (Human-written)

Outputs:
    ai_human_dataset.csv  (Filename, Text, Label, Split)
    Split: 70% train / 15% val / 15% test
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── CONFIG ──────────────────────────────────────────────────────────────────
AI_FOLDER    = "ai_resumes"
HUMAN_FOLDER = "humanwritten_resumes"
OUTPUT_CSV   = "ai_human_dataset.csv"
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────────────────────────────


def extract_text(file_path: str) -> str:
    """Strip PII block and section markers, return clean resume text."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\[Info_Start\].*?\[Info_End\]", "", content, flags=re.DOTALL)
    content = re.sub(r"\[section\]", "", content)
    content = re.sub(r"\[(Objective|Skill|Experience|Expericence|Education)\]", "", content)
    content = re.sub(r"\s+", " ", content).strip()

    return content


def load_folder(folder: str, label: int) -> list:
    rows = []
    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    for filename in files:
        text = extract_text(os.path.join(folder, filename))
        if len(text) > 50:     # skip near-empty files
            rows.append({"Filename": filename, "Text": text, "Label": label})
    return rows


# ─── Load both classes ────────────────────────────────────────────────────────
ai_rows    = load_folder(AI_FOLDER,    label=1)
human_rows = load_folder(HUMAN_FOLDER, label=0)

print(f"AI resumes    : {len(ai_rows)}")
print(f"Human resumes : {len(human_rows)}")

df = pd.DataFrame(ai_rows + human_rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ─── 70 / 15 / 15 split ───────────────────────────────────────────────────────
train_df, temp_df = train_test_split(df,      test_size=0.30, stratify=df["Label"], random_state=RANDOM_SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["Label"], random_state=RANDOM_SEED)

train_df = train_df.copy(); train_df["Split"] = "train"
val_df   = val_df.copy();   val_df["Split"]   = "val"
test_df  = test_df.copy();  test_df["Split"]  = "test"

final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nDataset saved: {len(final_df)} samples → '{OUTPUT_CSV}'")
print(f"Train : {len(train_df)}  |  Val : {len(val_df)}  |  Test : {len(test_df)}")
print(f"\nLabel distribution:\n{final_df.groupby(['Split','Label']).size().to_string()}")
