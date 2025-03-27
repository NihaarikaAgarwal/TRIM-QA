import json
import os
import csv
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

# Ensure you download the necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenize(text):
    return word_tokenize(text.lower())


################################hChunking#########################

#Cell chunking and its metadata
  # def chunk_cell(cell_text, cell_id, table_name, row_id, col_id):
  #     return {
  #         "text": cell_text,
  #         "metadata": {
  #             "table_name": table_name,
  #             "row_id": row_id,
  #             "col_id": col_id,
  #             "chunk_id": f"{table_name}_cell_{row_id}_{col_id}"
  #         }
  #     }

# Row chunking and its metadata
def chunk_row(row, row_id, table_name, columns):
    row_text = ' '.join([f"{columns[i]['text']}: {cell['text']}" for i, cell in enumerate(row['cells']) if columns[i]['text']])
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

#Column chunk and its metadata
def chunk_column(rows, col_id, col_name, table_name):
    column_text = ' | '.join([row['cells'][col_id]['text'] for row in rows if row['cells'][col_id]['text']])

    return {
        "text": f"{col_name if col_name else ''}: {column_text}",  # Ensure blank column names are handled
        "metadata": {
            "table_name": table_name,
            "col_id": col_id,
            "chunk_id": f"{table_name}_column_{col_id}",
            "chunk_type": "column",
            "metadata_text": f"table: {table_name}, col: {col_name if col_name else ''}, chunk_id: {table_name}_column_{col_id}, chunk_type: column"
        }
    }

# Table chunking with its metadata
def chunk_table(rows, table_id):
    """
    Create a table-level chunk with metadata.
    """
    table_text = ' '.join([cell['text'] for row in rows for cell in row['cells']])
    return {
        "text": table_text,
        "metadata": {
            "table_name": table_id,
            "chunk_id": f"{table_id}_table",
            "chunk_type": "table",
            "metadata_text": f"table_name: {table_id},chunk_id: {table_id}_table, chunk_type: table"
        }
    }

########################Processing##################################

# Process jsonl file: chunking
def process_jsonl(file_path):


    metadata_list = []
    chunks = []
    chunk_embeddings = []
    table_chunks = []

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

            # #Chunking Column
            for col_id, col in enumerate(columns):
                if col["text"]:
                    col_chunk = chunk_column(rows, col_id, col["text"], table_id)
                    chunks.append(col_chunk)
                    metadata_list.append(col_chunk["metadata"])

            #Chunking table
            table_chunk = chunk_table(rows, table_id)
            chunks.append(table_chunk)
            table_chunks.append(table_chunk)

    return metadata_list, chunks, table_chunks

# Tokenize query
def rank_chunks_with_bm25(tokenized_chunks, query, top_n):
    # Tokenize query using a better tokenization method (e.g., spaCy or nltk)
    
    # Tokenize chunks
    #tokenized_chunks = [tokenize(chunk['text']) for chunk in chunks]

    # Initialize BM25
    bm25 = BM25Okapi([chunk['tokenized_text'] for chunk in tokenized_chunks])

    # Get BM25 scores for the query
    #scores = bm25.get_scores(tokenized_query)
    scores = bm25.get_scores(query)

    # Sort chunks by BM25 score in descending order
    ranked_chunks = sorted(zip(scores, tokenized_chunks), reverse=True, key=lambda x: x[0])

    # Get top N chunks
    top_ranked_chunks = ranked_chunks[:top_n]
    
    return top_ranked_chunks

# Save the top N chunks to a file
def save_top_chunks(top_chunks, output_dir, output_filename):
    # Ensure the output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(top_chunks, f, indent=2)

    print(f"Saved top chunks to {output_path}")

# Calculate recall, rank, and 10%/20% check
def calculate_recall(ranked_chunks, correct_table_id, top_n):
    """
    Calculate recall based on whether the correct table_id is found in the top N ranked chunks.
    Returns the recall, rank, and checks for top 10% and top 20%.
    """
    # Find the rank of the correct table
    rank = None
    for idx, (_, chunk) in enumerate(ranked_chunks):  # Only consider top N chunks
        if chunk['table_id'] == correct_table_id:
            rank = idx + 1  # Rank starts at 1

            # Check if it's in the top 10% or 20%
            is_in_top_10 = 1 if rank <= top_n * 0.1 else 0
            is_in_top_20 = 1 if rank <= top_n * 0.2 else 0
            return 1, rank, is_in_top_10, is_in_top_20  # Relevant item found

    return 0, None, 0, 0  # Relevant item not found

# Main script
def main(file_path, dev_file_path, output_dir, top_n, queries_count):
    # Step 1: Process the JSONL file and generate chunks
    metadata, chunks, table_chunks = process_jsonl(file_path)
    table_chunks = sorted(table_chunks, key=lambda x: x["metadata"]["table_name"])

    total_recall = 0
    total_queries = 0
    total_top_10 = 0
    total_top_20 = 0
    results = []

    tokenized_chunks = []

    for i, chunk in enumerate(tqdm(table_chunks, desc="Tokenizing Chunks", unit="chunk")):
        table_id = chunk['metadata']['table_name']
        tokenized_text = tokenize(chunk['text'] + str(chunk['metadata']))

        tokenized_chunks.append({
            "table_id": table_id,
            "tokenized_text": tokenized_text,
        })

    # Step 2: Process dev.jsonl file and run BM25 on each query
    with open(dev_file_path, 'r') as dev_file:
        for i, line in enumerate(tqdm(dev_file)):
            if i >= queries_count:  # Stop after processing the first x queries
                break
            
            data = json.loads(line.strip())
            query = data['questions'][0]['originalText']
            correct_table_id = data['table']['tableId']

            # Step 3: Rank chunks using BM25 for the current query
            tokenized_query = tokenize(query)
            ranked_chunks = rank_chunks_with_bm25(tokenized_chunks, tokenized_query, top_n)

            # Step 4: Calculate reciprocal rank for this query
            recall, rank, is_in_top_10, is_in_top_20 = calculate_recall(ranked_chunks, correct_table_id, top_n)
            print(f"Recall for query '{query}' is: {recall * 100:.2f}%")

            total_recall += recall
            total_top_10 += is_in_top_10
            total_top_20 += is_in_top_20
            total_queries += 1

            # print(correct_table_id)
            # Step 5: Save the results for this query
            results.append({
                "Recall": recall * 100,  # Recall as percentage
                "Rank": rank if rank is not None else "Not found",
                "Ans table in top 10%": is_in_top_10 * 100,
                "Ans table in top 20%": is_in_top_20 * 100
            })

    # Step 6: Calculate and print overall recall score
    recall_percentage = (total_recall / total_queries) * 100 if total_queries > 0 else 0
    total_top_10_percentage = (total_top_10 / total_queries) * 100 if total_queries > 0 else 0
    total_top_20_percentage = (total_top_20 / total_queries) * 100 if total_queries > 0 else 0
    print(f"Overall Recall (Top {top_n} chunks): {recall_percentage:.2f}%, Top 10%: {total_top_10_percentage:.2f}%, Top 20%: {total_top_20_percentage:.2f}%")

    # Step 7: Save the results to a CSV file
    csv_filename = "query_results_top_" + str(top_n) + ".csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    # Write results to CSV
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Recall", "Rank", "Ans table in top 10%", "Ans table in top 20%"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filepath}")

    save_top_chunks(ranked_chunks, output_dir, "top_chunks.json")

# Example usage
file_path = "tables.jsonl"
dev_file = "test.jsonl"
output_dir = "results"
top_n = 5000
queries_count = 150

main(file_path, dev_file, output_dir, top_n, queries_count)