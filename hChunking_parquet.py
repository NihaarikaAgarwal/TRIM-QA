import json
from tqdm import tqdm
import json
import pandas as pd
from tqdm import tqdm

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
def chunk_table(rows, table_id, columns):
    """
    Create a structured table-level chunk with metadata.
    """
    column_names = " | ".join([col['text'] for col in columns])  # Include column names
    table_text = '\n'.join([column_names] + [' | '.join([cell['text'] for cell in row['cells']]) for row in rows])  # Prepend column names

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

            # Chunking Column
            for col_id, col in enumerate(columns):
                if col["text"]:
                    col_chunk = chunk_column(rows, col_id, col["text"], table_id)
                    chunks.append(col_chunk)
                    metadata_list.append(col_chunk["metadata"])

            # Chunking table
            table_chunk = chunk_table(rows, table_id, columns)
            chunks.append(table_chunk)
            table_chunks.append(table_chunk)

    return metadata_list, chunks, table_chunks

# Main
file_path = "/content/sample_data/testing.jsonl"  # Replace with your actual file path
metadata, chunks, table_chunks = process_jsonl(file_path)

# Convert the chunks to a DataFrame
chunks_df = pd.DataFrame(chunks)

# Save the DataFrame to a Parquet file
parquet_file_path = "output.parquet"
chunks_df.to_parquet(parquet_file_path, engine='pyarrow', index=False)

print(f"Parquet file saved to: {parquet_file_path}")
