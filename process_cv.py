import pandas as pd
import re
import os

def parse_resumes(folder_path):
    all_rows = []
    
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            
            content = re.sub(r'\[Info_Start\].*?\[Info_End\]', '', content, flags=re.DOTALL)

            
            content = re.sub(r'\[section\]', '', content)

            
            pattern = r'([^\[\n।]+)[\s।]*\[(Objective|Skill|Expericence|Education)\]'
            matches = re.findall(pattern, content)

            for text, label in matches:
                clean_text = text.strip().replace('।', '').replace('.', '')
                final_label = "Experience" if label == "Expericence" else label
                
                if clean_text: # Only add if the text isn't empty
                    all_rows.append({"filename": filename, "text": clean_text, "label": final_label})

    return pd.DataFrame(all_rows)


folder = "raw_resumes" 

if os.path.exists(folder):
    df = parse_resumes(folder)

    df = df[["filename", "text", "label"]]
    
    df.to_csv("labeled_cv_data.csv", index=False, encoding="utf-8-sig")
    
    print(f"Successfully processed {len(df)} lines from your resumes!")
    print("\nFirst 5 rows of your dataset:")
    print(df.head())
else:
    print(f"Error: Folder '{folder}' not found. Please create it and add your .txt files.")