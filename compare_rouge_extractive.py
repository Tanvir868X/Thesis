import pandas as pd
import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# ROUGE + Semantic Similarity: TF-IDF vs TextRank



ref_path      = "resume_summarization_dataset.csv"
tfidf_path    = "extractive_summarized_cvs.csv"
textrank_path = "textrank_summarized_cvs.csv"

print("Loading semantic similarity model...")
sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.\n")

CATEGORIES = ["Objective", "Experience", "Education", "Skill"]


# ── Bengali-aware ROUGE ───────────────────────────────────────────────────────

def tokenize(text):
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

tfidf_df    = pd.read_csv(tfidf_path,    encoding="utf-8-sig")
textrank_df = pd.read_csv(textrank_path, encoding="utf-8-sig")

tfidf_map    = {(r["Filename"], r["Category"]): str(r["Summary"]) for _, r in tfidf_df.iterrows()}
textrank_map = {(r["Filename"], r["Category"]): str(r["Summary"]) for _, r in textrank_df.iterrows()}

all_files = sorted(set(r[0] for r in tfidf_map) | set(r[0] for r in textrank_map))
print(f"Loaded {len(ref_df)} reference entries. Processing {len(all_files)} resumes...\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
section_rows = []
resume_rows  = []

for filename in all_files:
    res_tf = []
    res_tr = []

    for category in CATEGORIES:
        abs_ref = abs_map.get((filename, category), "")
        ext_ref = ext_map.get((filename, category), "")
        if not abs_ref and not ext_ref:
            continue

        tf_text = tfidf_map.get((filename, category), "")
        tr_text = textrank_map.get((filename, category), "")

        tf_abs_r = get_rouge(tf_text, abs_ref)
        tr_abs_r = get_rouge(tr_text, abs_ref)
        tf_abs_s = get_sem(tf_text, abs_ref)
        tr_abs_s = get_sem(tr_text, abs_ref)

        tf_ext_r = get_rouge(tf_text, ext_ref)
        tr_ext_r = get_rouge(tr_text, ext_ref)
        tf_ext_s = get_sem(tf_text, ext_ref)
        tr_ext_s = get_sem(tr_text, ext_ref)

        section_rows.append({
            "filename":                  filename, "category": category,
            "tfidf_abs_rouge1":          tf_abs_r["rouge1"],
            "tfidf_abs_rouge2":          tf_abs_r["rouge2"],
            "tfidf_abs_rougeL":          tf_abs_r["rougeL"],
            "tfidf_abs_semantic_sim":    tf_abs_s,
            "textrank_abs_rouge1":       tr_abs_r["rouge1"],
            "textrank_abs_rouge2":       tr_abs_r["rouge2"],
            "textrank_abs_rougeL":       tr_abs_r["rougeL"],
            "textrank_abs_semantic_sim": tr_abs_s,
            "tfidf_ext_rouge1":          tf_ext_r["rouge1"],
            "tfidf_ext_rouge2":          tf_ext_r["rouge2"],
            "tfidf_ext_rougeL":          tf_ext_r["rougeL"],
            "tfidf_ext_semantic_sim":    tf_ext_s,
            "textrank_ext_rouge1":       tr_ext_r["rouge1"],
            "textrank_ext_rouge2":       tr_ext_r["rouge2"],
            "textrank_ext_rougeL":       tr_ext_r["rougeL"],
            "textrank_ext_semantic_sim": tr_ext_s,
            "tfidf_summary":             tf_text,
            "textrank_summary":          tr_text,
            "abstractive_reference":     abs_ref,
            "extractive_reference":      ext_ref,
        })

        res_tf.append({"abs_r1": tf_abs_r["rouge1"], "abs_r2": tf_abs_r["rouge2"],
                       "abs_rl": tf_abs_r["rougeL"], "abs_s": tf_abs_s,
                       "ext_r1": tf_ext_r["rouge1"], "ext_r2": tf_ext_r["rouge2"],
                       "ext_rl": tf_ext_r["rougeL"], "ext_s": tf_ext_s})
        res_tr.append({"abs_r1": tr_abs_r["rouge1"], "abs_r2": tr_abs_r["rouge2"],
                       "abs_rl": tr_abs_r["rougeL"], "abs_s": tr_abs_s,
                       "ext_r1": tr_ext_r["rouge1"], "ext_r2": tr_ext_r["rouge2"],
                       "ext_rl": tr_ext_r["rougeL"], "ext_s": tr_ext_s})

    if res_tf:
        def avg(lst, k): return round(sum(x[k] for x in lst)/len(lst), 4)
        resume_rows.append({
            "filename":                  filename,
            "tfidf_abs_rouge1":          avg(res_tf,"abs_r1"), "tfidf_abs_rouge2": avg(res_tf,"abs_r2"),
            "tfidf_abs_rougeL":          avg(res_tf,"abs_rl"), "tfidf_abs_semantic_sim": avg(res_tf,"abs_s"),
            "tfidf_ext_rouge1":          avg(res_tf,"ext_r1"), "tfidf_ext_rouge2": avg(res_tf,"ext_r2"),
            "tfidf_ext_rougeL":          avg(res_tf,"ext_rl"), "tfidf_ext_semantic_sim": avg(res_tf,"ext_s"),
            "textrank_abs_rouge1":       avg(res_tr,"abs_r1"), "textrank_abs_rouge2": avg(res_tr,"abs_r2"),
            "textrank_abs_rougeL":       avg(res_tr,"abs_rl"), "textrank_abs_semantic_sim": avg(res_tr,"abs_s"),
            "textrank_ext_rouge1":       avg(res_tr,"ext_r1"), "textrank_ext_rouge2": avg(res_tr,"ext_r2"),
            "textrank_ext_rougeL":       avg(res_tr,"ext_rl"), "textrank_ext_semantic_sim": avg(res_tr,"ext_s"),
        })

    print(f"  ✓ {filename}")

# ── Save ──────────────────────────────────────────────────────────────────────
section_df = pd.DataFrame(section_rows)
resume_df  = pd.DataFrame(resume_rows)
section_df.to_csv("extractive_rouge_per_section.csv", index=False, encoding="utf-8-sig")
resume_df.to_csv("extractive_rouge_per_resume.csv",   index=False, encoding="utf-8-sig")

# ── Aggregate table ───────────────────────────────────────────────────────────
def print_table(title, metrics):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    print(f"{'Metric':<22} {'TF-IDF':>10} {'TextRank':>10} {'Winner':>10}")
    print("-"*70)
    for label, tf_col, tr_col in metrics:
        tf_m = resume_df[tf_col].mean()
        tr_m = resume_df[tr_col].mean()
        win  = "TF-IDF" if tf_m >= tr_m else "TextRank"
        print(f"{label:<22} {tf_m:>10.4f} {tr_m:>10.4f} {win:>10}")
    print("="*70)

print_table("vs ABSTRACTIVE REFERENCE", [
    ("ROUGE-1","tfidf_abs_rouge1","textrank_abs_rouge1"),
    ("ROUGE-2","tfidf_abs_rouge2","textrank_abs_rouge2"),
    ("ROUGE-L","tfidf_abs_rougeL","textrank_abs_rougeL"),
    ("Semantic Sim","tfidf_abs_semantic_sim","textrank_abs_semantic_sim"),
])
print_table("vs EXTRACTIVE REFERENCE", [
    ("ROUGE-1","tfidf_ext_rouge1","textrank_ext_rouge1"),
    ("ROUGE-2","tfidf_ext_rouge2","textrank_ext_rouge2"),
    ("ROUGE-L","tfidf_ext_rougeL","textrank_ext_rougeL"),
    ("Semantic Sim","tfidf_ext_semantic_sim","textrank_ext_semantic_sim"),
])
print(f"\nSaved: extractive_rouge_per_section.csv ({len(section_df)} rows)")
print(f"Saved: extractive_rouge_per_resume.csv   ({len(resume_df)} rows)")
