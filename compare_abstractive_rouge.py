import pandas as pd
import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer


# ROUGE + Semantic Similarity: BanglaT5 vs mT5


ref_path = "resume_summarization_dataset.csv"
bt5_path = "BanglaT5_Abstractive_Summaries.csv"
mt5_path = "MT5_Abstractive_Summaries.csv"

print("Loading semantic similarity model...")
sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.\n")

CATEGORIES = ["Objective", "Experience", "Education", "Skill"]


# ── Bengali-aware ROUGE ───────────────────────────────────────────────────────

def tokenize(text):
    """Keep Bengali Unicode chars + split on whitespace."""
    if not text or pd.isna(text):
        return []
    text = re.sub(r'[^\u0980-\u09FF\s]', ' ', str(text))
    return [t for t in text.split() if t.strip()]


def rouge_n(hyp, ref, n):
    hyp_tok = tokenize(hyp)
    ref_tok = tokenize(ref)
    if not hyp_tok or not ref_tok:
        return 0.0
    def ngrams(tok): return Counter(tuple(tok[i:i+n]) for i in range(len(tok)-n+1))
    hyp_ng = ngrams(hyp_tok)
    ref_ng = ngrams(ref_tok)
    overlap = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
    prec = overlap / sum(hyp_ng.values())
    rec  = overlap / sum(ref_ng.values())
    return round(2*prec*rec/(prec+rec), 4) if prec+rec else 0.0


def rouge_l(hyp, ref):
    hyp_tok = tokenize(hyp)
    ref_tok = tokenize(ref)
    if not hyp_tok or not ref_tok:
        return 0.0
    m, n = len(hyp_tok), len(ref_tok)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if hyp_tok[i-1]==ref_tok[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs  = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    return round(2*prec*rec/(prec+rec), 4) if prec+rec else 0.0


def get_rouge(hyp, ref):
    return {
        "rouge1": rouge_n(hyp, ref, 1),
        "rouge2": rouge_n(hyp, ref, 2),
        "rougeL": rouge_l(hyp, ref),
    }


def get_sem(hyp, ref):
    if not hyp or not ref: return 0.0
    emb = sem_model.encode([str(hyp), str(ref)])
    return round(float(np.dot(emb[0], emb[1]) /
                       (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))), 4)


# ── Load data ─────────────────────────────────────────────────────────────────
ref_df = pd.read_csv(ref_path, encoding="utf-8-sig")
ref_df.columns = ref_df.columns.str.strip()

abs_map = {(r["Filename"], r["Category"]): r["Abstractive_Reference_Summary"]
           for _, r in ref_df.iterrows()}
ext_map = {(r["Filename"], r["Category"]): r["Extractive_Reference_Summary"]
           for _, r in ref_df.iterrows()}

bt5_df  = pd.read_csv(bt5_path, encoding="utf-8-sig")
mt5_df  = pd.read_csv(mt5_path, encoding="utf-8-sig")

bt5_map = {(r["Filename"], r["Category"]): str(r["Summary"]) for _, r in bt5_df.iterrows()}
mt5_map = {(r["Filename"], r["Category"]): str(r["Summary"]) for _, r in mt5_df.iterrows()}

all_files = sorted(set(r[0] for r in bt5_map) | set(r[0] for r in mt5_map))
print(f"Loaded {len(ref_df)} reference entries. Processing {len(all_files)} resumes...\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
section_rows = []
resume_rows  = []

for filename in all_files:
    res_bt5 = []
    res_mt5 = []

    for category in CATEGORIES:
        abs_ref  = abs_map.get((filename, category), "")
        ext_ref  = ext_map.get((filename, category), "")
        if not abs_ref and not ext_ref:
            continue

        bt5_text = bt5_map.get((filename, category), "")
        mt5_text = mt5_map.get((filename, category), "")

        bt5_abs_r = get_rouge(bt5_text, abs_ref)
        mt5_abs_r = get_rouge(mt5_text, abs_ref)
        bt5_abs_s = get_sem(bt5_text, abs_ref)
        mt5_abs_s = get_sem(mt5_text, abs_ref)

        bt5_ext_r = get_rouge(bt5_text, ext_ref)
        mt5_ext_r = get_rouge(mt5_text, ext_ref)
        bt5_ext_s = get_sem(bt5_text, ext_ref)
        mt5_ext_s = get_sem(mt5_text, ext_ref)

        section_rows.append({
            "filename":               filename, "category": category,
            "bt5_abs_rouge1":         bt5_abs_r["rouge1"],
            "bt5_abs_rouge2":         bt5_abs_r["rouge2"],
            "bt5_abs_rougeL":         bt5_abs_r["rougeL"],
            "bt5_abs_semantic_sim":   bt5_abs_s,
            "mt5_abs_rouge1":         mt5_abs_r["rouge1"],
            "mt5_abs_rouge2":         mt5_abs_r["rouge2"],
            "mt5_abs_rougeL":         mt5_abs_r["rougeL"],
            "mt5_abs_semantic_sim":   mt5_abs_s,
            "bt5_ext_rouge1":         bt5_ext_r["rouge1"],
            "bt5_ext_rouge2":         bt5_ext_r["rouge2"],
            "bt5_ext_rougeL":         bt5_ext_r["rougeL"],
            "bt5_ext_semantic_sim":   bt5_ext_s,
            "mt5_ext_rouge1":         mt5_ext_r["rouge1"],
            "mt5_ext_rouge2":         mt5_ext_r["rouge2"],
            "mt5_ext_rougeL":         mt5_ext_r["rougeL"],
            "mt5_ext_semantic_sim":   mt5_ext_s,
            "bt5_summary":            bt5_text,
            "mt5_summary":            mt5_text,
            "abstractive_reference":  abs_ref,
            "extractive_reference":   ext_ref,
        })

        res_bt5.append({"abs_r1": bt5_abs_r["rouge1"], "abs_r2": bt5_abs_r["rouge2"],
                        "abs_rl": bt5_abs_r["rougeL"], "abs_s": bt5_abs_s,
                        "ext_r1": bt5_ext_r["rouge1"], "ext_r2": bt5_ext_r["rouge2"],
                        "ext_rl": bt5_ext_r["rougeL"], "ext_s": bt5_ext_s})
        res_mt5.append({"abs_r1": mt5_abs_r["rouge1"], "abs_r2": mt5_abs_r["rouge2"],
                        "abs_rl": mt5_abs_r["rougeL"], "abs_s": mt5_abs_s,
                        "ext_r1": mt5_ext_r["rouge1"], "ext_r2": mt5_ext_r["rouge2"],
                        "ext_rl": mt5_ext_r["rougeL"], "ext_s": mt5_ext_s})

    if res_bt5:
        def avg(lst, k): return round(sum(x[k] for x in lst)/len(lst), 4)
        resume_rows.append({
            "filename":              filename,
            "bt5_abs_rouge1":        avg(res_bt5,"abs_r1"), "bt5_abs_rouge2": avg(res_bt5,"abs_r2"),
            "bt5_abs_rougeL":        avg(res_bt5,"abs_rl"), "bt5_abs_semantic_sim": avg(res_bt5,"abs_s"),
            "bt5_ext_rouge1":        avg(res_bt5,"ext_r1"), "bt5_ext_rouge2": avg(res_bt5,"ext_r2"),
            "bt5_ext_rougeL":        avg(res_bt5,"ext_rl"), "bt5_ext_semantic_sim": avg(res_bt5,"ext_s"),
            "mt5_abs_rouge1":        avg(res_mt5,"abs_r1"), "mt5_abs_rouge2": avg(res_mt5,"abs_r2"),
            "mt5_abs_rougeL":        avg(res_mt5,"abs_rl"), "mt5_abs_semantic_sim": avg(res_mt5,"abs_s"),
            "mt5_ext_rouge1":        avg(res_mt5,"ext_r1"), "mt5_ext_rouge2": avg(res_mt5,"ext_r2"),
            "mt5_ext_rougeL":        avg(res_mt5,"ext_rl"), "mt5_ext_semantic_sim": avg(res_mt5,"ext_s"),
        })

    print(f"  ✓ {filename}")

# ── Save ──────────────────────────────────────────────────────────────────────
section_df = pd.DataFrame(section_rows)
resume_df  = pd.DataFrame(resume_rows)
section_df.to_csv("abstractive_rouge_per_section.csv", index=False, encoding="utf-8-sig")
resume_df.to_csv("abstractive_rouge_per_resume.csv",   index=False, encoding="utf-8-sig")

# ── Aggregate table ───────────────────────────────────────────────────────────
def print_table(title, metrics):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    print(f"{'Metric':<22} {'BanglaT5':>10} {'mT5':>10} {'Winner':>10}")
    print("-"*70)
    for label, bt5_col, mt5_col in metrics:
        bt5_m = resume_df[bt5_col].mean()
        mt5_m = resume_df[mt5_col].mean()
        win   = "BanglaT5" if bt5_m >= mt5_m else "mT5"
        print(f"{label:<22} {bt5_m:>10.4f} {mt5_m:>10.4f} {win:>10}")
    print("="*70)

print_table("vs ABSTRACTIVE REFERENCE", [
    ("ROUGE-1","bt5_abs_rouge1","mt5_abs_rouge1"),
    ("ROUGE-2","bt5_abs_rouge2","mt5_abs_rouge2"),
    ("ROUGE-L","bt5_abs_rougeL","mt5_abs_rougeL"),
    ("Semantic Sim","bt5_abs_semantic_sim","mt5_abs_semantic_sim"),
])
print_table("vs EXTRACTIVE REFERENCE", [
    ("ROUGE-1","bt5_ext_rouge1","mt5_ext_rouge1"),
    ("ROUGE-2","bt5_ext_rouge2","mt5_ext_rouge2"),
    ("ROUGE-L","bt5_ext_rougeL","mt5_ext_rougeL"),
    ("Semantic Sim","bt5_ext_semantic_sim","mt5_ext_semantic_sim"),
])
print(f"\nSaved: abstractive_rouge_per_section.csv ({len(section_df)} rows)")
print(f"Saved: abstractive_rouge_per_resume.csv   ({len(resume_df)} rows)")
