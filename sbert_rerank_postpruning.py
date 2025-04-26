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
QUERY_FOLDER = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/top_chunks_top_5000"  
QUERY_TABLES_FOLDER = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/sbert_pruning_output_rows"  # Updated folder path
OUTPUT_FOLDER = "sbert_reranked_postpruning_row"        
RANGE_START = 0
RANGE_END = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load SBERT model ===
print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME)

# === Convert table JSON to SBERT-friendly string ===
# def table_to_text(table_dict, max_rows=3):
#     columns = table_dict['table']['columns']
#     rows = table_dict['table']['rows'][:max_rows]
    
#     table_text = []
#     for row in rows:
#         cells = row['cells']
#         row_text = " | ".join(
#             f"{columns[i]['text']}: {cells[i]['text']}" for i in range(min(len(columns), len(cells)))
#         )
#         table_text.append(row_text)
    
#     return " || ".join(table_text)

def table_to_text(table_dict, max_rows=3):
    columns = table_dict['table']['columns']
    rows = table_dict['table']['rows'][:max_rows]
    
    table_text = []
    table_id = table_dict["id"]  # Get table ID

    # Add table ID to the table text
    table_text.append(f"Table ID: {table_id}")
    
    for row in rows:
        cells = row['cells']
        row_text = " | ".join(
            f"{columns[i]['text']}: {cells[i]['text']}" for i in range(min(len(columns), len(cells)))
        )
        table_text.append(row_text)
    
    return " || ".join(table_text)


# === Rerank each query CSV ===
def rerank_query(query, query_number):
    tables_jsonl = os.path.join(QUERY_TABLES_FOLDER, f"query_{query_number}_top_5000_OnlyRowsPruned.jsonl")
    
    
    print(f"Looking for file: {tables_jsonl}")

    relevant_tables = []
    table_ids = []  

    try:
        with open(tables_jsonl, "r") as f:
            for line in f:
                table = json.loads(line)
                relevant_tables.append(table)
                table_ids.append(table["id"]) 
    except FileNotFoundError:
        print(f"File not found: {tables_jsonl}")
        return {}, {}, []  
    
    # Save the table IDs for this query to a JSON file
    table_ids_file = os.path.join(OUTPUT_FOLDER, f"table_id_{query_number}_POSTPRUNING.json")
    with open(table_ids_file, "w") as f:
        json.dump(table_ids, f)
    print(f"Saved table IDs for query {query_number} to {table_ids_file}")

    # Generate embeddings for the relevant tables
    table_texts = [table_to_text(table) for table in relevant_tables]
    relevant_embeddings = model.encode(table_texts, convert_to_numpy=True)

    # Generate embedding for the query
    query_vec = model.encode(query, convert_to_numpy=True).reshape(1, -1)

    # Calculate cosine similarity
    scores = cosine_similarity(query_vec, relevant_embeddings)[0]

    tid_to_score = {relevant_tables[i]["id"]: float(score) for i, score in enumerate(scores)}

    sorted_ids = sorted(tid_to_score.items(), key=lambda x: x[1], reverse=True)
    tid_to_rank = {tid: rank + 1 for rank, (tid, _) in enumerate(sorted_ids)}

    return tid_to_score, tid_to_rank, relevant_tables

print("Starting reranking on queries")

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
    except Exception as e:
        print(f"Skipping {csv_file} due to format error: {e}")
        continue

    print(f"Processing {csv_file}: {query[:60]}")

    tid_to_score, tid_to_rank, relevant_tables = rerank_query(query, file_index)

    # Process output rows
    rows = []
    for table in relevant_tables:
        tid = table["id"]
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
