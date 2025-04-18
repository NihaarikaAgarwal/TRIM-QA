import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# === CONFIG ===
MODEL_NAME = "all-MiniLM-L6-v2"  
TABLES_JSONL = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/tables.jsonl"
QUERY_FOLDER = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/top_chunks_top_5000"             
OUTPUT_FOLDER = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/reranked_output"        
TABLE_IDS_FILE = "table_id_5000.json"  
EMBEDDINGS_FILE = "full_embeddings.npy" 
RANGE_START = 0
RANGE_END = 149

# create ouput folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load SBERT model ===
print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME)

# === Convert table JSON to SBERT-friendly string ===
def table_to_text(table_dict, max_rows=3):
    headers = [col['text'] for col in table_dict['columns']]
    rows = table_dict['rows'][:max_rows]
    table_text = []
    for row in rows:
        cells = row['cells']
        row_text = " | ".join(
            f"{headers[i]}: {cells[i]['text']}" for i in range(min(len(headers), len(cells)))
        )
        table_text.append(row_text)
    return " || ".join(table_text)

# === Encode all 160k tables and save ===
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(TABLE_IDS_FILE):
    print(" Encoding tables 160k tables")

    table_texts = []
    table_ids = []

    with open(TABLES_JSONL, "r") as f:
        for line in tqdm(f, desc="Reading tables"):
            table = json.loads(line)
            table_texts.append(table_to_text(table))
            table_ids.append(table["tableId"])

    print("Encoding with SBERT")
    embeddings = model.encode(table_texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)

    print("Saving embeddings and table IDs")
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(TABLE_IDS_FILE, "w") as f:
        json.dump(table_ids, f)
    print("Table encoding complete.")
else:
    print("Embeddings and table IDs already exist.")

# === Rerank each query CSV ===

# Load saved table embeddings + IDs
print("Loading saved embeddings and table IDs")
embedding_matrix = np.load(EMBEDDINGS_FILE)
with open(TABLE_IDS_FILE) as f:
    all_table_ids = json.load(f)

# rerank only relevant table IDs for a query
def rerank_query(query, relevant_ids):
    query_vec = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    
    # Find index of each relevant table ID
    relevant_indices = [all_table_ids.index(tid) for tid in relevant_ids if tid in all_table_ids]
    if not relevant_indices:
        return {}, {}

    relevant_embeddings = embedding_matrix[relevant_indices]
    scores = cosine_similarity(query_vec, relevant_embeddings)[0]

    # Map: table_id n score
    tid_to_score = {all_table_ids[i]: float(score) for i, score in zip(relevant_indices, scores)}

    # Sort and assign rank
    sorted_ids = sorted(tid_to_score.items(), key=lambda x: x[1], reverse=True)
    tid_to_rank = {tid: rank + 1 for rank, (tid, _) in enumerate(sorted_ids)}

    return tid_to_score, tid_to_rank

# Process each query CSV
print(" Starting reranking on queries")

FILENAME_PATTERN = r"query_(\d+)_top_5000\.csv"  

# Process each CSV in folder
print(f"Reranking queries in range [{RANGE_START}, {RANGE_END}]")
for csv_file in sorted(os.listdir(QUERY_FOLDER)):
    if not csv_file.endswith(".csv"):
        continue
    
    match = re.match(FILENAME_PATTERN, csv_file)
    if not match:
        continue

    file_index = int(match.group(1))
    if not (RANGE_START <= file_index <= RANGE_END):
        continue  

    csv_path = os.path.join(QUERY_FOLDER, csv_file)
    df = pd.read_csv(csv_path, header=None)

    try:
        query = df.iloc[1, 0]
        target_id = str(df.iloc[1, 2])
        relevant_ids = [str(df.iloc[i, 1]) for i in range(1, len(df)) if pd.notna(df.iloc[i, 1])]
    except Exception as e:
        print(f"Skipping {csv_file} due to format error: {e}")
        continue

    print(f" Processing {csv_file}: {query[:60]}")

    tid_to_score, tid_to_rank = rerank_query(query, relevant_ids)

    rows = []
    for tid in relevant_ids:
        score = tid_to_score.get(tid, -1)
        rank = tid_to_rank.get(tid, -1)
        rows.append([query, target_id, tid, rank, score])
        
    rows = sorted(rows, key=lambda x: x[3]) 

    # Save per-query output
    base_name = os.path.splitext(csv_file)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_reranked.csv")
    out_df = pd.DataFrame(rows, columns=["query", "target_table", "table_id", "rank", "score"])
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
