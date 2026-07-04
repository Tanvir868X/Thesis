import pandas as pd
import joblib
import os
import re

# 1. LOAD YOUR PRE-TRAINED ASSETS
model = joblib.load('cv_model_svm.pkl')
tfidf = joblib.load('cv_tfidf_vectorizer.pkl')

def extract_sentences_from_file(file_path):
    """Modified version of your process_cv logic to handle single files"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove PII and Section tags
    content = re.sub(r'\[Info_Start\].*?\[Info_End\]', '', content, flags=re.DOTALL)
    content = re.sub(r'\[section\]', '', content)
    
    # Find all text parts regardless of label to get a full view
    # We use your regex logic but keep it flexible
    pattern = r'([^\[\n।]+)[\s।]*\[(Objective|Skill|Experience|Expericence|Education)\]'
    matches = re.findall(pattern, content)
    
    data = []
    for text, label in matches:
        clean_text = text.strip().replace('।', '').replace('.', '')
        # Fix typo
        clean_label = "Experience" if label == "Expericence" else label
        if clean_text:
            data.append({"text": clean_text, "label": clean_label})
    return data

def create_summary(sentences_data):
    if not sentences_data:
        return "No valid data found."

    df_temp = pd.DataFrame(sentences_data)
    
    # 2. USE TF-IDF TO FIND IMPORTANCE
    # .A1 converts the matrix to a simple 1D array of scores
    vecs = tfidf.transform(df_temp['text'])
    df_temp['importance'] = vecs.sum(axis=1).A1 
    
    summary_parts = []

    # Strategy: 
    # Objective -> Top 1 by TF-IDF
    # Education -> Top 1 (most informative)
    # Experience -> Top 2 (highest impact sentences)
    # Skills -> Top 5-10 keywords
    
    for label in ['Objective', 'Education']:
        subset = df_temp[df_temp['label'] == label]
        if not subset.empty:
            best_idx = subset['importance'].idxmax()
            summary_parts.append(f"[{label}]: {subset.loc[best_idx, 'text']}")

    exp_subset = df_temp[df_temp['label'] == 'Experience']
    if not exp_subset.empty:
        top_exp = exp_subset.sort_values(by='importance', ascending=False).head(2)
        exp_text = " | ".join(top_exp['text'].values)
        summary_parts.append(f"[Experience Highlights]: {exp_text}")

    skill_subset = df_temp[df_temp['label'] == 'Skill']
    if not skill_subset.empty:
        skills = ", ".join(skill_subset['text'].unique()[:8])
        summary_parts.append(f"[Top Skills]: {skills}")

    return " \n".join(summary_parts)

# --- EXECUTION LOOP FOR 100 RESUMES ---
raw_folder = "raw_resumes"
results = []

print("Starting Summarization...")

if os.path.exists(raw_folder):
    for filename in os.listdir(raw_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(raw_folder, filename)
            sentences = extract_sentences_from_file(file_path)
            
            summary_text = create_summary(sentences)
            
            results.append({
                "filename": filename,
                "summary": summary_text
            })

    # Save to CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("extractive_summarized_cvs.csv", index=False, encoding="utf-8-sig")
    print(f"Successfully summarized {len(results)} resumes into 'summarized_cvs.csv'!")
else:
    print("Error: 'raw_resumes' folder not found.")
