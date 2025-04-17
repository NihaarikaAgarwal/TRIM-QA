import csv
import json
import numpy as np
from tqdm import tqdm
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import os
import logging
import concurrent.futures

logging.basicConfig(
    filename='my_log_file_18.log',     # File to write logs to
    filemode='a',                   # 'a' = append, 'w' = overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO              # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

#Configurations
#csv_file = "/content/sample_data/query1_test_reranking.csv"
folder_path = "/Users/inkitatewari/Documents/Semester 2/NLP/Group Project/Tapas/query csv result/top_chunks_top_18" 
jsonl_file = "tables.jsonl"
#query = "when did season 3 of orange is the new black come out"
alpha = 1  # std dev penalty weight

# Load TAPAS Model
model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

def get_query_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    query = df.iloc[1, 0]  
    return query

# Load table IDs from CSV (2nd column)
def load_relevant_table_ids(csv_file):
    table_ids = set()
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                table_ids.add(row[1].strip())
    return table_ids

# Load matching tables from JSONL
def load_matching_tables(jsonl_file, relevant_ids):
    matched = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            table = json.loads(line)
            if table.get("tableId", "").strip() in relevant_ids:
                matched.append(table)
    return matched

# Convert table format to TAPAS-compatible
def convert_to_tapas_format(table):
    headers = [col['text'] for col in table['columns']]
    data_rows = []
    for row in table['rows']:
        row_cells = [cell['text'] for cell in row['cells']]
        data_rows.append(row_cells)
    return headers, data_rows

def normalize_logits(logits):
    # Calculate mean and std for the logits
    mean = logits.mean().item()
    std = logits.std().item()
    
    if std > 0:  # Avoid division by zero
        normalized_logits = (logits - mean) / std
    else:
        normalized_logits = logits  # If std is 0, don't normalize

    return normalized_logits

def process_single_table(table, query):
    headers, rows = convert_to_tapas_format(table)  
    df = pd.DataFrame(rows, columns=headers) 
    
    inputs = tokenizer(table=df, queries=[query], return_tensors="pt", truncation=True)
    
    return inputs

def batch_tokenize(tables, query):
    input_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_table, table, query) for table in tables]
        # Collect results as they are completed
        for future in concurrent.futures.as_completed(futures):
            input_list.append(future.result())

    return input_list

def get_logits_from_tapas(headers, rows, query):
    df = pd.DataFrame(rows, columns=headers)

    #tokenize
    inputs = tokenizer(table=df, queries=[query], return_tensors="pt", truncation=True)
    # for k, v in inputs.items():
    #   print("input tensor shape")
    #   print(f"{k}: {v.shape}")

    outputs = model(**inputs)
    # Shape: (batch_size=1, seq_len)
    logits = outputs.logits
    # print(f"\nlogits shape: {logits.shape}")
    # print("All logits:", logits[0].tolist())

    # Masking logic

    # token_type_id shape: (channel,seq_len,7)
    token_type_ids = inputs["token_type_ids"][0]
    # shape: (1, seq_len)
    segment_ids = inputs["token_type_ids"][:, :, 0]
    # print(f"segment_ids shape: {segment_ids.shape}")

    # segment_ids == 1 to get table tokens
    cell_mask = (segment_ids[0] == 1) & (inputs["attention_mask"][0] == 1)
    # print(f"cell_mask shape: {cell_mask.shape}")
    valid_logits = logits[0][cell_mask]
    # print("Valid (masked) logits:", valid_logits.tolist())
    
    # Normalize the logits using Z-score standardization
    normalized_logits = normalize_logits(valid_logits)

    # logging.info(f"Normalized logits: {normalized_logits}")
    #return valid_logits
    return normalized_logits
    #return top_logits

# Compute mean, std, and score
#def score_logits(valid_logits, alpha=1.0):
def score_logits(normalized_logits, alpha=1.0):
  #if valid_logits.numel() > 0:
  #if len(valid_logits) > 0:
  if normalized_logits is not None and normalized_logits.numel() > 0:
    # mean = valid_logits.mean().item()
    # std = valid_logits.std().item()
    mean = normalized_logits.mean().item()
    std = normalized_logits.std().item()
    score = mean - alpha * std
    # print(f"\n Valid logits shape: {valid_logits.shape}")
    # print(f"Mean: {mean:.3f}, Std: {std:.3f}, Score: {score:.3f}")
    # logging.info(f"Mean: {mean:.3f}, Std: {std:.3f}, Score: {score:.3f}")
    return score, mean, std
  else:
    # print("No valid table cell logits found after masking!")
    logging.info("No valid table cell logits found after masking!")
    return float('-inf'), 0.0, 0.0


# Main pipeline
def rank_relevant_tables(csv_file):
    query = get_query_from_csv(csv_file)
    relevant_ids = load_relevant_table_ids(csv_file)
    matched_tables = load_matching_tables(jsonl_file, relevant_ids)
    # print(f"Loaded {len(matched_tables)} tables matching IDs.")
    
    # Extract headers and rows for each table to pass to batch_tokenize
    tables = [{'columns': table['columns'], 'rows': table['rows']} for table in matched_tables]
    # Batch tokenize the tables with the same query
    input_list = batch_tokenize(tables, query)
    
    ranked = []
    for i, table in enumerate(matched_tables):
    #for table in tqdm(matched_tables, desc="Scoring tables"):
        headers, rows = convert_to_tapas_format(table)
        logging.info(f"Get Logits for table: {table.get('tableId')}")
        # for i, row in enumerate(rows):
          # if len(row) != len(headers):
              # print(f"[Mismatch] Table ID: {table.get('tableId')}")
              # print(f"Header length: {len(headers)}")
              # print(f"Row {i} length: {len(row)} → Row content: {row}")

        # print("Headers:", headers)
        # print("First row:", rows[0] if rows else "No rows")
        # print("Query:", query)

        try:
            logits = get_logits_from_tapas(headers, rows, query)
            #logits = get_logits_from_tapas_topk(headers, rows, query)
            #logits = get_logits_from_tapas_clip(headers, rows, query, clip_min=-100, clip_max=100)
            score, mean, std = score_logits(logits, alpha)
            #score, mean, std = score_logits_topk(logits, alpha)
            #score, mean, std = score_logits_clip(logits, alpha, clip_min=-100, clip_max=100)
            ranked.append({
                "tableId": table["tableId"],
                "score": score,
                "mean": mean,
                "std": std
            })
        except Exception as e:
            # print(f"Failed to process table {table['tableId']}: {e}")
            logging.error(f"Failed to process table {table['tableId']}: {e}")

    ranked_sorted = sorted(ranked, key=lambda x: x["score"], reverse=True)

    # Print top 10
    # for i, entry in enumerate(ranked_sorted[:10]):
    #     print(f"{i+1}. {entry['tableId']} | Score: {entry['score']:.3f} | Mean: {entry['mean']:.3f} | Std: {entry['std']:.3f}")

    return ranked_sorted, query 

def calculate_recall_from_csv(ranked_tables):

    ranked_table_ids = [entry["tableId"] for entry in ranked_tables]

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) < 2 or len(rows[1]) < 3:
            #print("CSV does not have the target table.")
            return
        target_id = rows[1][2].strip()

    top_10_cutoff = max(1, int(0.10 * len(ranked_table_ids)))
    top_20_cutoff = max(1, int(0.20 * len(ranked_table_ids)))

    if target_id in ranked_table_ids:
        rank = ranked_table_ids.index(target_id) + 1
        in_10 = int(rank <= top_10_cutoff)
        in_20 = int(rank <= top_20_cutoff)
        # print(f"Target table {target_id} found at rank {rank}")
        # print(f"Recall@10%: {in_10}, Recall@20%: {in_20}")
        return target_id,rank,in_10, in_20
    else:
        #print(f" Target table {target_id} not found in reranked list.")
        return target_id, -1, 0, 0
    
output_rows = []
file_list = os.listdir(folder_path)
# file_list = ['query_138_top_10.csv']
i = 0
for file_name in file_list:
    i += 1
    # print(f"{i}. checking query in {file_name}")
    logging.info(f"{i}. checking query in {file_name}")
    
    if file_name.endswith(".csv"):
        csv_file = os.path.join(folder_path, file_name)
        print(f"Processing CSV file: {csv_file}")
        ranked_sorted, query = rank_relevant_tables(csv_file)
        target_id,rank, in_10, in_20 = calculate_recall_from_csv(ranked_sorted)
        output_rows.append([query, target_id, rank, in_10, in_20])

# Write the results to a CSV
output_csv = "ranked_results_18.csv"
header = ["Query", "Target table","Target Table Rank", "Recall at 10%", "Recall at 20%"]

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(output_rows)

print(f"Results written to {output_csv}")