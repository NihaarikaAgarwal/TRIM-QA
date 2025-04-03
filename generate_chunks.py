import json
import os
import csv
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import random

import nltk
from nltk.tokenize import word_tokenize

# Necessary Resources
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenize(text):
    return word_tokenize(text.lower())

################################ Chunking #################################

# Row chunking and its metadata
def chunk_row(row, row_id, table_name, columns):
    row_text = ' | '.join([f"{columns[i]['text']}: {cell['text']}" for i, cell in enumerate(row['cells']) if columns[i]['text']])
    return {
        "text": row_text,
        "metadata": {
            "table_name": table_name,
            "row_id": row_id,
            "chunk_id": f"{table_name}_row_{row_id}",
            "chunk_type": "row",
            "columns": [col["text"] for col in columns],
            "metadata_text": f"table: {table_name}, row: {row_id}, chunk_id: {table_name}_row_{row_id}, chunk_type: row, columns: {', '.join([col['text'] for col in columns if col['text']])}"
        }
    }

# Column chunk and its metadata
def chunk_column(rows, col_id, col_name, table_name):
    column_text = ' | '.join([row['cells'][col_id]['text'] for row in rows if row['cells'][col_id]['text']])

    return {
        "text": f"{col_name if col_name else ''}: {column_text}",
        "metadata": {
            "table_name": table_name,
            "col_id": col_id,
            "chunk_id": f"{table_name}_column_{col_id}",
            "chunk_type": "column",
            "metadata_text": f"table: {table_name}, col: {col_name if col_name else ''}, chunk_id: {table_name}_column_{col_id}, chunk_type: column"
        }
    }

# Table chunking with its metadata
def chunk_table(rows, table_id, columns):
    column_names = " | ".join([col['text'] for col in columns])
    table_text = '\n'.join([column_names] + [' | '.join([cell['text'] for cell in row['cells']]) for row in rows])

    return {
        "text": table_text,
        "metadata": {
            "table_name": table_id,
            "chunk_id": f"{table_id}_table",
            "chunk_type": "table",
            "columns": [col["text"] for col in columns],  # Adding column names
            "metadata_text": f"table_name: {table_id}, chunk_id: {table_id}_table, chunk_type: table, columns: {', '.join([col['text'] for col in columns])}"
        }
    }

######################## Processing ##################################

# Process jsonl file: chunking
def process_jsonl(file_path):

    metadata_list = []
    chunks = []

    with open(file_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            table_id = data['tableId']
            rows = data['rows']
            columns = data['columns']

            # Chunking row
            for row_id, row in enumerate(rows):
                row_chunk = chunk_row(row, row_id, table_id, columns)
                chunks.append(row_chunk)
                metadata_list.append(row_chunk["metadata"])

            # Chunking Column
            for col_id, col in enumerate(columns):
                if col["text"]:
                    col_chunk = chunk_column(rows, col_id, col["text"], table_id)
                    chunks.append(col_chunk)
                    metadata_list.append(col_chunk["metadata"])

            # Chunking table
            table_chunk = chunk_table(rows, table_id, columns)
            chunks.append(table_chunk)

    return metadata_list, chunks


# Rank Chunks
def rank_chunks_with_bm25(tokenized_chunks, query, top_n):
    # Using BM25
    bm25 = BM25Okapi([chunk['tokenized_text'] for chunk in tokenized_chunks])
    scores = bm25.get_scores(query)

    # Sort chunks by BM25 score in descending order
    ranked_chunks = sorted(zip(scores, tokenized_chunks), reverse=True, key=lambda x: x[0])

    # Get top N chunks
    top_ranked_chunks = ranked_chunks[:top_n]
    
    return top_ranked_chunks

# Save the top N chunks to a file
def save_top_chunks(top_chunks, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(top_chunks, f, indent=2)

    print(f"Saved top chunks to {output_path}")

# Calculate recall, rank, and 10%/20% check
def calculate_recall(ranked_chunks, correct_table_id, top_n):
    rank = None
    for idx, (_, chunk) in enumerate(ranked_chunks):
        if chunk['table_id'] == correct_table_id:
            rank = idx + 1

            # Check if it's in the top 10% or 20%
            is_in_top_10 = 1 if rank <= top_n * 0.1 else 0
            is_in_top_20 = 1 if rank <= top_n * 0.2 else 0
            return 1, rank, is_in_top_10, is_in_top_20

    return 0, None, 0, 0  # Relevant item not found

# Main script
def main(file_paths, tables_file_path, output_dir, top_n_values, queries_count, saved_queries_path):
    # Check if the random queries already exist
    if os.path.exists(saved_queries_path):
        print(f"Loading saved queries from {saved_queries_path}")
        with open(saved_queries_path, 'r') as f:
            selected_queries = [json.loads(line) for line in f]
    else:
        # Combine all queries from the three files
        all_queries = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    all_queries.append(data)
        
        # Shuffle queries and select the random queries_count
        random.shuffle(all_queries)
        selected_queries = all_queries[:queries_count]
        
        # Save the random queries to a JSONL file for future use
        print(f"Saving random queries to {saved_queries_path}")
        with open(saved_queries_path, 'w') as f:
            for query in selected_queries:
                json.dump(query, f)
                f.write("\n")

    # Process the tables in tables.jsonl
    table_metadata, table_chunks = process_jsonl(tables_file_path)
    selected_queries = selected_queries[:queries_count]

    total_recall = {top_n: 0 for top_n in top_n_values}
    total_queries = 0
    total_top_10 = {top_n: 0 for top_n in top_n_values}
    total_top_20 = {top_n: 0 for top_n in top_n_values}
    results = {top_n: [] for top_n in top_n_values}

    tokenized_chunks = []
    for i, chunk in enumerate(tqdm(table_chunks, desc="Tokenizing Chunks", unit="chunk")):
        table_id = chunk['metadata']['table_name']
        tokenized_text = tokenize(chunk['text'] + str(chunk['metadata']))

        tokenized_chunks.append({
            "table_id": table_id,
            "tokenized_text": tokenized_text,
        })

    # Process selected random queries
    for query_data in tqdm(selected_queries, desc="Processing Queries", unit="query"):
        query = query_data['questions'][0]['originalText']
        correct_table_id = query_data['table']['tableId']

        tokenized_query = tokenize(query)

        for top_n in top_n_values:
            ranked_chunks = rank_chunks_with_bm25(tokenized_chunks, tokenized_query, top_n)

            recall, rank, is_in_top_10, is_in_top_20 = calculate_recall(ranked_chunks, correct_table_id, top_n)
            print(f"Recall for query '{query}' (Top {top_n}) is: {recall * 100:.2f}%")

            total_recall[top_n] += recall
            total_top_10[top_n] += is_in_top_10
            total_top_20[top_n] += is_in_top_20
            total_queries += 1

            results[top_n].append({
                "Recall": recall * 100,  # Recall as percentage
                "Rank": rank if rank is not None else "Not found",
                "Ans table in top 10%": is_in_top_10 * 100,
                "Ans table in top 20%": is_in_top_20 * 100
            })

    # Calculate and print overall scores for each top_n
    for top_n in top_n_values:
        recall_percentage = (total_recall[top_n] / total_queries) * 100 if total_queries > 0 else 0
        total_top_10_percentage = (total_top_10[top_n] / total_queries) * 100 if total_queries > 0 else 0
        total_top_20_percentage = (total_top_20[top_n] / total_queries) * 100 if total_queries > 0 else 0
        print(f"Overall Recall (Top {top_n} chunks): {recall_percentage:.2f}%, Top 10%: {total_top_10_percentage:.2f}%, Top 20%: {total_top_20_percentage:.2f}%")

        # Save results to CSV for the current top_n
        csv_filename = f"query_results_top_{top_n}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Recall", "Rank", "Ans table in top 10%", "Ans table in top 20%"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results[top_n]:
                writer.writerow(result)

        print(f"Results saved to {csv_filepath}")

        # Save top ranked chunks for the current top_n
        save_top_chunks(results[top_n], output_dir, f"top_chunks_{top_n}.json")

# Execution parameters
file_paths = ["./NQ-Dataset/interactions/test.jsonl", "./NQ-Dataset/interactions/train.jsonl", "./NQ-Dataset/interactions/dev.jsonl"]
tables_file_path = "./NQ-Dataset/tables/tables.jsonl"
output_dir = "top_chunks_output"
top_n = [5000, 2500, 1250, 625, 312, 156, 78, 39, 18, 10]
queries_count = 150
saved_queries_path = "saved_random_queries.jsonl"

main(file_paths, tables_file_path, output_dir, top_n, queries_count, saved_queries_path)