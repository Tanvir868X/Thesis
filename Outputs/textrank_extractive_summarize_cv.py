import pandas as pd
import joblib
import os
import re
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# TextRank Extractive Summarizer — for comparison against TF-IDF scoring
#
# Approach: per-section TextRank using the same TF-IDF vectorizer already
# trained in Pipeline 1 (cv_tfidf_vectorizer.pkl). This keeps the comparison
# fair — both methods use identical text representations; only the sentence
# selection strategy differs.
#
# TF-IDF scoring (current):  ranks each sentence independently by its total
#                             TF-IDF weight → picks top-N globally per section
#
# TextRank (this script):     builds a similarity graph between all sentences
#                             in a section, runs PageRank, picks top-N by rank
#                             → prefers sentences that are central/representative
#                             rather than just high-weight individually
#
# Top-N per section (same as TF-IDF script):
#   Objective  → 1 sentence
#   Education  → 1 sentence
#   Experience → 2 sentences
#   Skill      → 8 sentences
# ---------------------------------------------------------------------------

tfidf = joblib.load("cv_tfidf_vectorizer.pkl")

TOP_N = {
    "Objective":  1,
    "Education":  1,
    "Experience": 2,
    "Skill":      8,
}


def parse_resume(file_path):
    """Extract labeled sentences from a raw resume file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\[Info_Start\].*?\[Info_End\]", "", content, flags=re.DOTALL)
    content = re.sub(r"\[section\]", "", content)

    pattern = r"([^\[\n।]+)[\s।]*\[(Objective|Skill|Experience|Expericence|Education)\]"
    matches = re.findall(pattern, content)

    data = []
    for text, label in matches:
        clean_text = text.strip().replace("।", "").replace(".", "").strip()
        clean_label = "Experience" if label == "Expericence" else label
        if clean_text:
            data.append({"text": clean_text, "label": clean_label})
    return data


def textrank_section(sentences, top_n):
    """
    Run TextRank on a list of sentences and return the top-N by PageRank score.

    Steps:
      1. Vectorize with the pre-trained TF-IDF (same as TF-IDF script)
      2. Build NxN cosine similarity matrix
      3. Threshold low similarities to 0 (keeps graph sparse, avoids noise)
      4. Build weighted undirected graph with nx
      5. Run PageRank
      6. Return top-N sentences sorted by rank score (descending)
    """
    n = len(sentences)

    # Degenerate cases — not enough sentences to build a meaningful graph
    if n == 0:
        return []
    if n == 1:
        return sentences

    vecs = tfidf.transform(sentences)
    sim_matrix = cosine_similarity(vecs)

    # Zero out self-similarity (diagonal) and very weak edges
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix[sim_matrix < 0.05] = 0

    # If the matrix is all zeros (all sentences completely dissimilar),
    # fall back to returning sentences in original order up to top_n
    if sim_matrix.sum() == 0:
        return sentences[:top_n]

    # Build graph: nodes = sentence indices, edges weighted by similarity
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, alpha=0.85, max_iter=200)

    # Sort by PageRank score descending, pick top_n
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in ranked[:top_n]]

    # Return sentences in their original document order (preserves readability)
    selected_indices_ordered = sorted(selected_indices)
    return [sentences[i] for i in selected_indices_ordered]


def create_textrank_summary(sentences_data):
    """Build a full resume summary using per-section TextRank."""
    if not sentences_data:
        return "No valid data found."

    df = pd.DataFrame(sentences_data)
    summary_parts = []

    for label in ["Objective", "Education", "Experience", "Skill"]:
        subset = df[df["label"] == label]["text"].tolist()
        top_n  = TOP_N.get(label, 1)

        if not subset:
            summary_parts.append(f"[{label}]: তথ্য নেই।")
            continue

        top_sentences = textrank_section(subset, top_n)

        if label == "Experience":
            text = " | ".join(top_sentences)
            summary_parts.append(f"[Experience Highlights]: {text}")
        elif label == "Skill":
            text = ", ".join(top_sentences)
            summary_parts.append(f"[Top Skills]: {text}")
        else:
            summary_parts.append(f"[{label}]: {top_sentences[0]}")

    return " \n".join(summary_parts)


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------
raw_folder = "raw_resumes"
results    = []

print("Starting TextRank Summarization...")

if os.path.exists(raw_folder):
    files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".txt")])
    print(f"Processing {len(files)} resumes...\n")

    for filename in files:
        try:
            file_path = os.path.join(raw_folder, filename)
            sentences = parse_resume(file_path)
            summary   = create_textrank_summary(sentences)
            results.append({"filename": filename, "summary": summary})
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")

    out_path = "textrank_summarized_cvs.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSUCCESS: {len(results)} summaries saved to '{out_path}'")

else:
    print(f"Error: folder '{raw_folder}' not found.")
