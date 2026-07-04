"""
STEP 1 — Generate Training Data via OpenRouter (gpt-oss-120b)
=============================================================
Reads all resumes from raw_resumes/, extracts 4 sections per resume,
sends each section to gpt-oss-120b for abstractive summary,
saves as resume_summarization_dataset.csv

Output CSV columns:
    Filename | Category | Input_Text | Reference_Summary

Run this ONCE to build the training dataset.
~100 resumes × 4 sections = ~400 rows
"""

import os
import re
import time
import requests
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
MODEL              = "openai/gpt-oss-120b:free"   # free tier
RAW_FOLDER         = "raw_resumes"
OUTPUT_CSV         = "resume_summarization_dataset.csv"
DELAY              = 2.0    # seconds between calls
MAX_RETRIES        = 5      # retry on failure
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "https://bup.edu.bd",   # optional but good practice
}

SYSTEM_PROMPT = (
    "আপনি একজন বাংলা ভাষার বিশেষজ্ঞ। "
    "আপনার কাজ হলো বাংলা সিভির নির্দিষ্ট অংশ পড়ে "
    "২-৩টি সংক্ষিপ্ত, সুসংহত এবং অর্থবহ বাংলা বাক্যে সারসংক্ষেপ তৈরি করা। "
    "শুধুমাত্র সারসংক্ষেপটি লিখুন — কোনো ভূমিকা, শিরোনাম বা ইংরেজি শব্দ যোগ করবেন না। "
    "বাক্য অসম্পূর্ণ রাখবেন না — প্রতিটি বাক্য সম্পূর্ণ করুন।"
)

CATEGORY_LABELS = {
    "Objective": "লক্ষ্য (Career Objective)",
    "Experience": "পেশাগত অভিজ্ঞতা (Work Experience)",
    "Education":  "শিক্ষাগত যোগ্যতা (Education)",
    "Skill":      "দক্ষতা (Skills)",
}


def extract_sections(file_path: str) -> dict:
    """Parse resume file and return dict of {category: [sentences]}."""
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


def call_openrouter(category: str, sentences: list) -> str:
    """Send section sentences to gpt-oss-120b, return Bengali summary."""
    context = " । ".join(sentences)
    label   = CATEGORY_LABELS[category]

    user_prompt = (
        f"নিচে একটি বাংলা সিভির '{label}' অংশের বাক্যগুলো দেওয়া হলো:\n\n"
        f"{context}\n\n"
        f"এই তথ্যের উপর ভিত্তি করে ২-৩ বাক্যে সংক্ষিপ্ত বাংলা সারসংক্ষেপ লিখুন। "
        f"প্রতিটি বাক্য সম্পূর্ণ করুন।"
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 300,      # high enough to never truncate Bengali sentences
        "temperature": 0.3,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()

            # Check for API-level error in response body
            if "error" in result:
                raise ValueError(result["error"].get("message", "Unknown API error"))

            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)     # exponential backoff
            else:
                return "সারসংক্ষেপ তৈরি করা সম্ভব হয়নি।"


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if not os.path.exists(RAW_FOLDER):
    print(f"Error: '{RAW_FOLDER}' folder not found.")
    exit(1)

files = sorted([f for f in os.listdir(RAW_FOLDER) if f.endswith(".txt")])
print(f"Processing {len(files)} resumes with gpt-oss-120b...\n")

rows = []
for i, filename in enumerate(files, 1):
    path     = os.path.join(RAW_FOLDER, filename)
    sections = extract_sections(path)

    print(f"[{i}/{len(files)}] {filename}")
    for category in ["Objective", "Experience", "Education", "Skill"]:
        sentences = sections.get(category, [])
        if not sentences:
            continue

        summary = call_openrouter(category, sentences)
        rows.append({
            "Filename":          filename,
            "Category":          category,
            "Input_Text":        " । ".join(sentences),
            "Reference_Summary": summary,
        })
        print(f"    ✓ [{category}] {summary[:60]}...")
        time.sleep(DELAY)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nDataset saved: {len(df)} training pairs → '{OUTPUT_CSV}'")
print(f"\nCategory breakdown:\n{df['Category'].value_counts().to_string()}")