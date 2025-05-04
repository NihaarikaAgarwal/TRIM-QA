import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import sys
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import jsonlines
import time
import os
import warnings
import argparse
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import faiss
import mmap

# Add argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Table pruning script with configurable query range')
    parser.add_argument('--start_query', type=int, required=True, help='Start query number (inclusive)')
    parser.add_argument('--end_query', type=int, required=True, help='End query number (inclusive)')
    return parser.parse_args()

# Suppress CUDA compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global state to track if we need to fall back to CPU
cuda_fallback_needed = False

# Check GPU compatibility and set up devices
def check_gpu_compatibility_and_setup_devices():
    global cuda_fallback_needed
    cpu_device = torch.device("cpu")
    model_device = cpu_device
    gpu_faiss_device = None
    gpu_compatible_for_faiss = False

    if torch.cuda.is_available():
        try:
            # Try a simple tensor operation on GPU
            test_tensor = torch.zeros(1).cuda()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version_pytorch = torch.version.cuda
            print(f"GPU detected: {gpu_name}")
            print(f"PyTorch CUDA Version: {cuda_version_pytorch}")

            # Test more complex CUDA operation to check for kernel compatibility
            try:
                test_model = SentenceTransformer("all-MiniLM-L6-v2")
                test_model.to("cuda")
                test_model.encode("test sentence", device="cuda")
                print("CUDA kernel test successful")
            except RuntimeError as e:
                if "no kernel image is available for execution on the device" in str(e):
                    print("WARNING: CUDA kernel compatibility issue detected")
                    print("Falling back to CPU for model operations")
                    cuda_fallback_needed = True
                    return cpu_device, cpu_device, None, False
                raise e  # Re-raise if it's a different runtime error

            # For A100, we want to use GPU if kernel test passed
            if 'A100' in gpu_name and not cuda_fallback_needed:
                print("A100 GPU detected and validated - using GPU")
                model_device = torch.device("cuda:0")
                gpu_faiss_device = torch.device("cuda:0")
                gpu_compatible_for_faiss = True
                compute_capability = torch.cuda.get_device_capability(0)
                print(f"GPU Compute Capability: {compute_capability}")
            # For other GPUs, check CUDA version if kernel test passed
            elif cuda_version_pytorch and float(cuda_version_pytorch.split('.')[0]) >= 11 and not cuda_fallback_needed:
                print("CUDA version is compatible with GPU.")
                model_device = torch.device("cuda:0")
                gpu_faiss_device = torch.device("cuda:0")
                gpu_compatible_for_faiss = True
                compute_capability = torch.cuda.get_device_capability(0)
                print(f"GPU Compute Capability: {compute_capability}")
            else:
                print("Using CPU for model operations due to compatibility concerns.")
                gpu_faiss_device = None
                gpu_compatible_for_faiss = False

        except Exception as e:
            print(f"GPU detected but error during test: {e}")
            print("Falling back to CPU for all operations")
            model_device = cpu_device
            gpu_faiss_device = None
            gpu_compatible_for_faiss = False
    else:
        print("No GPU detected, using CPU for all operations")
        model_device = cpu_device
        gpu_faiss_device = None
        gpu_compatible_for_faiss = False

    return cpu_device, model_device, gpu_faiss_device, gpu_compatible_for_faiss

# Set devices based on compatibility check
cpu_device, model_device, gpu_faiss_device, gpu_compatible_for_faiss = check_gpu_compatibility_and_setup_devices()

# Load SBERT model - use the selected model_device
print(f"Loading model on {model_device}...")
model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to(model_device) # <--- Sends the model to GPU if available
print("Model loaded.")


# Pre-load chunk map
print("Loading data resources...")
with open("chunk_id_map.json", 'r', encoding='utf-8') as f:
    chunk_map = json.load(f)

chunk_map_id = {table_id: idx for idx, table_id in chunk_map.items()}

# Chunk cache to improve performance
chunk_cache = {}

# Initialize FAISS resources with GPU support if available
def initialize_faiss_resources(index_path, attempt_gpu_device):
    """Initialize FAISS index and move to GPU if available and compatible"""
    try:
        print("Loading FAISS index...")
        cpu_index = faiss.read_index(index_path)

        if attempt_gpu_device and gpu_compatible_for_faiss: # Check if GPU device is set and deemed compatible
            try:
                gpu_index_num = attempt_gpu_device.index if attempt_gpu_device.index is not None else 0
                print(f"Attempting to move FAISS index to GPU: {gpu_index_num} ({torch.cuda.get_device_name(gpu_index_num)})")
                res = faiss.StandardGpuResources()
                # Use the specific GPU index
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_index_num, cpu_index)
                print("FAISS index successfully moved to GPU")
                return gpu_index
            except Exception as e:
                print(f"Error moving FAISS index to GPU: {e}")
                print("Falling back to CPU index for FAISS")
                return cpu_index
        else:
             if not attempt_gpu_device:
                 print("GPU device not specified for FAISS, using CPU index.")
             elif not gpu_compatible_for_faiss:
                 print("GPU deemed incompatible or unavailable for FAISS during initial check, using CPU index.")
             return cpu_index
    except Exception as e:
        print(f"Error initializing FAISS index from path {index_path}: {e}")
        sys.exit(1)


def fetch_chunk_ids(table_id):
    """
    Fetches row and column chunk IDs for a given table_id
    """
    row_ids = []
    column_ids = []
    for chunk_id, idx in chunk_map_id.items():
        if chunk_id.startswith(table_id):
            if "_row_" in chunk_id:
                row_ids.append((chunk_id, idx))
            elif "_column_" in chunk_id:
                column_ids.append((chunk_id, idx))

    return row_ids, column_ids

# Optimized chunk fetching with caching
def fetch_chunks(row_ids, column_ids):
    """
    Fetches row and column chunks using known offsets with caching
    """
    if not row_ids and not column_ids:
        return [], []

    chunks_file_path = "chunks.json"
    index_file_path = "chunk_index.json"

    # Load index if not already in memory
    if not hasattr(fetch_chunks, "offset_index"):
        try:
            with open(index_file_path, 'r', encoding='utf-8') as idx_file:
                fetch_chunks.offset_index = json.load(idx_file)
        except FileNotFoundError:
            print(f"Error: Chunk index file not found at {index_file_path}")
            return [], [] # Return empty lists if index is missing
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from chunk index file {index_file_path}")
            return [], []

    # Extract just the chunk_ids from the tuples
    row_chunk_ids = [cid for cid, _ in row_ids]
    col_chunk_ids = [cid for cid, _ in column_ids]

    combined_ids = list(set(row_chunk_ids + col_chunk_ids))
    id_to_offset = {cid: fetch_chunks.offset_index.get(cid) for cid in combined_ids if cid in fetch_chunks.offset_index}
    sorted_items = sorted([(cid, offset) for cid, offset in id_to_offset.items() if offset is not None], key=lambda x: x[1])

    row_chunks, column_chunks = [], []

    # Use mmap for efficient file access
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for chunk_id, offset in sorted_items:
                    # Check cache first
                    if chunk_id in chunk_cache:
                        chunk = chunk_cache[chunk_id]
                    else:
                        try:
                            mm.seek(offset)
                            line = mm.readline()
                            chunk = json.loads(line.decode('utf-8'))
                            # Cache for future use, limit cache size
                            if len(chunk_cache) < 10000:  # Simple cache size limit
                                chunk_cache[chunk_id] = chunk
                        except ValueError: # Catch potential errors if offset is wrong or line is corrupted
                             print(f"Warning: Could not read or decode chunk at offset {offset} for chunk_id {chunk_id}. Skipping.")
                             continue
                        except Exception as e:
                             print(f"Warning: Unexpected error reading chunk {chunk_id} at offset {offset}: {e}. Skipping.")
                             continue


                    # Determine if this is a row or column chunk
                    if chunk_id in row_chunk_ids:
                        row_chunks.append(chunk)
                    elif chunk_id in col_chunk_ids:
                        column_chunks.append(chunk)
    except FileNotFoundError:
        print(f"Error: Chunks file not found at {chunks_file_path}")
        return [], [] # Return empty if chunks file is missing

    return row_chunks, column_chunks

# Optimized scoring function using CPU tensors for calculation
def score_chunks_from_faiss(query_embedding_cpu, chunk_tuples, faiss_index): # Renamed parameter for clarity
    """
    Scores chunks based on semantic similarity using CPU for tensor operations.
    Expects query_embedding_cpu to be a CPU tensor.
    """
    if not chunk_tuples:
        return []

    scores = []

    # Ensure query embedding is a CPU tensor
    if not isinstance(query_embedding_cpu, torch.Tensor):
         # If it was numpy or other, convert it
         try:
             query_embedding_cpu = torch.from_numpy(np.asarray(query_embedding_cpu)).float()
         except Exception as e:
             print(f"Error converting query embedding to tensor: {e}")
             return [] # Cannot proceed without a valid query tensor

    query_embedding_cpu = query_embedding_cpu.to(cpu_device) # Ensure it's on CPU
    if len(query_embedding_cpu.shape) == 1:
        query_embedding_cpu = query_embedding_cpu.unsqueeze(0) # Ensure shape is (1, dim)

    # Process chunks individually (batching reconstruct can be complex with error handling)
    for chunk_id, idx in chunk_tuples:
        try:
            # Reconstruct embedding from FAISS (returns numpy array)
            chunk_vector = faiss_index.reconstruct(int(idx))
            if chunk_vector is None:
                print(f"Warning: Could not reconstruct vector for index {idx} (chunk_id: {chunk_id}). Skipping.")
                continue

            # Convert reconstructed vector to CPU tensor for calculation
            chunk_embedding = torch.from_numpy(chunk_vector).unsqueeze(0).to(cpu_device) # Move to CPU

            # Compute similarity on CPU using PyTorch
            similarity = F.cosine_similarity(
                query_embedding_cpu.float(), # Ensure float32
                chunk_embedding.float()      # Ensure float32
            ).item() # Get scalar value

            score = round((similarity + 1) / 2, 4) # Normalize from [-1,1] to [0,1] range
            scores.append((chunk_id, score, idx))

        except IndexError:
             print(f"Warning: Index {idx} out of bounds for FAISS index (chunk_id: {chunk_id}). Skipping.")
             continue
        except ValueError: # Handle potential issues with int(idx)
             print(f"Warning: Invalid index format '{idx}' for chunk_id {chunk_id}. Skipping.")
             continue
        except AttributeError: # Handle case where faiss_index might not be initialized correctly
             print(f"Error: faiss_index object does not have 'reconstruct' method. FAISS initialization likely failed.")
             # Depending on desired behavior, you might want to return [] or raise the error
             return [] # Stop processing if FAISS index is broken
        except Exception as e:
            # Catch other potential errors during reconstruction or similarity calculation
            print(f"Error scoring chunk_id {chunk_id} (index {idx}): {e}")
            import traceback
            print(traceback.format_exc()) # Print stack trace for unexpected errors
            continue # Skip this chunk

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def dynamic_threshold(scores, alpha):
    """
    Computes a dynamic threshold over chunk scores
    """
    if not scores:
        return 0.0

    score_values = [float(score) for _, score, _ in scores]
    if not score_values: # Handle case where scores might be empty after filtering errors
        return 0.0

    median = np.median(score_values)
    std_dev = np.std(score_values)
    return median + alpha * std_dev

def filter_chunks_by_threshold(scored_chunks, threshold):
    """
    Filters chunk_ids whose scores meet or exceed the threshold
    """
    if not scored_chunks:
        return []
    # Ensure score is treated as float for comparison
    return [(chunk_id, idx) for chunk_id, score, idx in scored_chunks if float(score) >= threshold]


def column_chunks_to_dataframe(column_chunks):
    """
    Converts a list of column chunks into a pandas DataFrame
    """
    data = {}

    for chunk in column_chunks:
        # Ensure chunk is a dictionary and has 'text' key
        if not isinstance(chunk, dict) or "text" not in chunk:
            print(f"Warning: Skipping invalid column chunk format: {chunk}")
            continue

        text = chunk.get("text", "")

        if ":" in text:
            parts = text.split(":", 1)
            header = parts[0].strip()
            values_str = parts[1].strip() if len(parts) > 1 else ""

            if not header: # Skip if header is empty after stripping
                continue

            # Split values by '|', strip whitespace, and filter out empty strings
            values = [v.strip() for v in values_str.split("|") if v.strip()]
            data[header] = values
        # else: # Optional: Handle chunks without a colon if needed
            # print(f"Warning: Column chunk text does not contain ':'. Text: {text}")


    if not data:
        return pd.DataFrame()

    # Pad shorter columns with empty strings to ensure equal length for DataFrame creation
    max_len = max((len(vals) for vals in data.values()), default=0)
    for header, values in data.items():
        if len(values) < max_len:
            # Use empty string for padding
            values.extend([""] * (max_len - len(values)))

    try:
        df = pd.DataFrame.from_dict(data, orient='columns')
    except Exception as e:
        print(f"Error creating DataFrame from column chunks: {e}")
        print(f"Data dictionary causing error: {data}")
        return pd.DataFrame() # Return empty DataFrame on error

    return df


def row_chunks_to_dataframe(row_chunks):
    """
    Converts row-based chunks into a structured DataFrame
    """
    rows = []
    all_columns = set() # Keep track of all unique column names encountered

    for chunk in row_chunks:
         # Ensure chunk is a dictionary and has 'text' and 'metadata'
        if not isinstance(chunk, dict) or "text" not in chunk or "metadata" not in chunk:
            print(f"Warning: Skipping invalid row chunk format: {chunk}")
            continue

        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        # Ensure metadata is a dict and get columns, default to empty list
        column_names = metadata.get("columns", []) if isinstance(metadata, dict) else []

        # Initialize row_data with empty strings for all expected columns from metadata
        row_data = {col: "" for col in column_names}

        # Parse the text field (e.g., "col1: val1 | col2: val2")
        parts = [p.strip() for p in text.split("|") if p.strip()]
        for part in parts:
            if ":" in part:
                key_val = part.split(":", 1)
                key = key_val[0].strip()
                value = key_val[1].strip() if len(key_val) > 1 else ""
                # Only add if the key was listed in the metadata columns
                if key in column_names:
                    row_data[key] = value
                # else: # Optional: Warn if a key in text is not in metadata columns
                    # print(f"Warning: Key '{key}' found in row text but not in metadata columns {column_names} for chunk.")


        # Add the parsed row data
        rows.append(row_data)
        # Update the set of all columns encountered across all rows
        all_columns.update(column_names)

    if not rows:
        return pd.DataFrame()

    # Create DataFrame using the union of all columns, ensuring consistent structure
    # Sort columns alphabetically for consistent output
    sorted_all_columns = sorted(list(all_columns))
    try:
        # Reconstruct rows ensuring all columns are present, filling missing with ""
        structured_rows = []
        for row in rows:
            structured_row = {col: row.get(col, "") for col in sorted_all_columns}
            structured_rows.append(structured_row)

        df = pd.DataFrame(structured_rows, columns=sorted_all_columns)
    except Exception as e:
        print(f"Error creating DataFrame from row chunks: {e}")
        print(f"Processed rows causing error: {structured_rows if 'structured_rows' in locals() else 'Error before structuring'}")
        return pd.DataFrame() # Return empty DataFrame on error

    return df


def intersect_row_and_column_dfs(df_row, df_col):
    """
    Filters df_row to keep only the columns that are also present in df_col.
    Returns a DataFrame with columns sorted alphabetically.
    """
    # Check if either DataFrame is empty or invalid
    if not isinstance(df_row, pd.DataFrame) or df_row.empty or \
       not isinstance(df_col, pd.DataFrame) or df_col.empty:
        return pd.DataFrame() # Return an empty DataFrame

    # Find common columns
    common_cols = sorted(list(set(df_row.columns).intersection(set(df_col.columns))))

    if not common_cols:
        # If no common columns, return an empty DataFrame but with original index if needed
        return pd.DataFrame(index=df_row.index)

    # Return the row DataFrame filtered to only common columns, maintaining original rows
    return df_row[common_cols]


def dataframe_to_json_entry(df, table_id, query):
    """
    Convert a pandas DataFrame into a JSON-serializable dict for output.
    Handles potential non-string data types by converting to string.
    """
    if not isinstance(df, pd.DataFrame):
         print(f"Warning: Cannot convert non-DataFrame to JSON for table {table_id}. Input type: {type(df)}")
         # Return a minimal structure or None, depending on how you want to handle this
         return {
             "id": table_id,
             "query": query, # Include query for context
             "error": "Input was not a valid DataFrame",
             "table": {"columns": [], "rows": []}
         }

    try:
        # Ensure all column headers are strings
        columns_list = [{"text": str(col)} for col in df.columns]

        # Ensure all cell values are strings
        rows_list = []
        for _, row in df.iterrows():
            cells_list = [{"text": str(cell) if pd.notna(cell) else ""} for cell in row]
            rows_list.append({"cells": cells_list})

        json_entry = {
            "id": table_id,
            # "query": query, # Optionally include the query in each entry
            "table": {
                "columns": columns_list,
                "rows": rows_list
            }
        }
        return json_entry
    except Exception as e:
        print(f"Error converting DataFrame to JSON for table {table_id}: {e}")
        # Return an error structure
        return {
            "id": table_id,
            "query": query,
            "error": f"Failed to convert DataFrame to JSON: {e}",
            "table": {"columns": [], "rows": []}
        }


def prune_table(table_id, query_embedding_cpu, faiss_index_instance): # Accept CPU embedding and faiss_index
    """
    Optimized table pruning function with error handling.
    Expects query_embedding_cpu to be a CPU tensor.
    """
    try:
        # Fetch Row and Column IDs from chunk_id_map.json #
        row_ids, column_ids = fetch_chunk_ids(table_id)
        if not row_ids and not column_ids:
            print(f"No row or column chunks found for table_id: {table_id}")
            return pd.DataFrame() # Return empty if no chunks exist

        # Score chunks using the CPU embedding and the provided FAISS index
        # Pass the CPU embedding directly
        row_scores = score_chunks_from_faiss(query_embedding_cpu, row_ids, faiss_index_instance)
        column_scores = score_chunks_from_faiss(query_embedding_cpu, column_ids, faiss_index_instance)

        # Calculate dynamic thresholds
        # Adjusted alpha values based on experimentation (can be tuned)
        row_threshold = dynamic_threshold(row_scores, alpha=0.5) # Keep more rows initially
        column_threshold = dynamic_threshold(column_scores, alpha=-0.2) # Be more selective with columns

        # Filter by threshold
        filtered_row_ids = filter_chunks_by_threshold(row_scores, row_threshold)
        filtered_column_ids = filter_chunks_by_threshold(column_scores, column_threshold)

        # Fetch filtered chunks
        filtered_row_chunks, filtered_column_chunks = fetch_chunks(filtered_row_ids, filtered_column_ids)

        # Convert to dataframes
        filtered_row_df = row_chunks_to_dataframe(filtered_row_chunks)
        filtered_columns_df = column_chunks_to_dataframe(filtered_column_chunks)

        # Intersect based on columns and return the pruned DataFrame
        pruned_df = intersect_row_and_column_dfs(filtered_row_df, filtered_columns_df)

        if pruned_df.empty:
            print(f"Pruning resulted in an empty table for table_id: {table_id}")

        # return pruned_df
        return filtered_row_df

    except Exception as e:
        print(f"Error pruning table {table_id}: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame() # Return empty DataFrame on error

# Main execution with improved error handling
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        start_query = args.start_query
        end_query = args.end_query
        
        print(f"Processing queries in range: {start_query} to {end_query}")
        
        # Initialize FAISS using the determined device
        faiss_index_path = "RCT_embeddings.index"
        faiss_index = initialize_faiss_resources(faiss_index_path, gpu_faiss_device)

        if faiss_index is None:
             print("Fatal Error: FAISS index could not be loaded. Exiting.")
             sys.exit(1)

        folder_path = "top_chunks_top_5000"
        output_folder = "sbert_pruning_output_5000"
        os.makedirs(output_folder, exist_ok=True)

        print(f"Looking for input files in: {os.path.abspath(folder_path)}")
        print(f"Will write output to: {os.path.abspath(output_folder)}")

        # Dynamically find files matching the pattern
        file_pattern = "query_.*_top_5000.csv"
        try:
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('query_') and '_top_5000' in f]
            
            # Filter files based on user-specified query range
            file_list = []
            for f in all_files:
                try:
                    # Extract query number from filename (query_X_top_5000.csv)
                    query_num = int(f.split('_')[1])
                    if start_query <= query_num <= end_query:
                        file_path = os.path.join(folder_path, f)
                        if os.path.exists(file_path):
                            file_list.append(f)
                except (IndexError, ValueError):
                    # Skip files that don't match the expected naming pattern
                    continue
            
            # Sort files by query number for consistent processing
            file_list.sort(key=lambda x: int(x.split('_')[1]))
            
        except FileNotFoundError:
            print(f"Error: Input directory not found: {folder_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error listing files in {folder_path}: {e}")
            sys.exit(1)


        if not file_list:
            print(f"No input files matching pattern '{file_pattern}' found in {folder_path}")
            print("Available files in directory:")
            try:
                print(os.listdir(folder_path))
            except FileNotFoundError:
                 print("Directory not found.")
            sys.exit(1)

        print(f"Found {len(file_list)} files to process: {file_list}")

        # Pre-load offset index for chunks (already done in fetch_chunks, but can be explicit here too)
        try:
            with open("chunk_index.json", 'r', encoding='utf-8') as idx_file:
                fetch_chunks.offset_index = json.load(idx_file)
            print("Chunk index loaded successfully.")
        except FileNotFoundError:
            print("Error: chunk_index.json not found. Chunk fetching will fail.")
            # Decide whether to exit or let fetch_chunks handle it
            # sys.exit(1)
        except json.JSONDecodeError:
            print("Error: chunk_index.json is corrupted. Chunk fetching will fail.")
            # sys.exit(1)


        # Set CUDA launch blocking for potentially better error messages during GPU execution
        # Only set if actually using CUDA
        if model_device.type == 'cuda' or (gpu_faiss_device and gpu_faiss_device.type == 'cuda'):
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            print("CUDA_LAUNCH_BLOCKING set to 1 for debugging.")

        total_start_time = time.time()

        # Process files
        for file_idx, file_name in enumerate(file_list):
            print(f"\n--- Processing file {file_idx + 1}/{len(file_list)}: {file_name} ---")
            file_start_time = time.time()

            csv_file = os.path.join(folder_path, file_name)
            try:
                df_input = pd.read_csv(csv_file)
                if 'query' not in df_input.columns or 'top tables' not in df_input.columns:
                     print(f"Error: CSV file {file_name} is missing required columns 'query' or 'top tables'. Skipping.")
                     continue
                # Get the query from the first row since it's the same for all rows
                question = df_input['query'].iloc[0]
                # Get ALL table IDs from the CSV, not just unique ones
                table_list = df_input['top tables'].dropna().astype(str).tolist()
                print(f"Processing {len(table_list)} tables for query: {question}")

                if not question or not table_list:
                     print(f"Warning: No query or table list found in {file_name}. Skipping.")
                     continue

            except Exception as e:
                print(f"Error reading or parsing CSV file {file_name}: {e}")
                continue # Skip to the next file

            try:
                print(f"Encoding query on {model_device}: '{question[:100]}...'") # Print truncated query
                
                # Add fallback logic for encoding
                encoding_device = cpu_device if cuda_fallback_needed else model_device
                print(f"Using device {encoding_device} for encoding")
                
                # Encode on the selected device
                try:
                    query_embedding_tensor = model.encode(
                        question,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=encoding_device
                    )
                except RuntimeError as e:
                    if "no kernel image is available for execution on the device" in str(e):
                        print("CUDA kernel error during encoding, falling back to CPU")
                        cuda_fallback_needed = True
                        # Retry on CPU
                        query_embedding_tensor = model.encode(
                            question,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=cpu_device
                        )
                    else:
                        raise e

                query_embedding_tensor = query_embedding_tensor.reshape(1, -1).float()
                print(f"Query encoded successfully on {query_embedding_tensor.device}. Shape: {query_embedding_tensor.shape}")

                # Move the query embedding to CPU *before* passing it to pruning/scoring functions
                query_embedding_cpu = query_embedding_tensor.to(cpu_device)
                print(f"Query embedding moved to {query_embedding_cpu.device} for FAISS/scoring.")


                output_path = os.path.join(output_folder, f"{file_name.replace('.csv', '')}_OnlyRowsPruned.jsonl")
                print(f"Outputting pruned tables to: {output_path}")
                processed_tables_count = 0
                with open(output_path, "w", encoding="utf-8") as f_out:
                    # Limit processing for testing if needed, e.g., table_list[:10]
                    for table_idx, table_id in enumerate(table_list):
                        print(f"  Pruning Table {table_idx + 1}/{len(table_list)}: {table_id}")
                        table_prune_start = time.time()
                        # Pass the CPU tensor and the loaded FAISS index
                        pruned_df = prune_table(table_id, query_embedding_cpu, faiss_index)
                        table_prune_end = time.time()

                        if pruned_df is not None and not pruned_df.empty:
                            entry = dataframe_to_json_entry(pruned_df, table_id, question)
                            if entry and "error" not in entry: # Check if conversion was successful
                                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                processed_tables_count += 1
                                print(f"    -> Pruned table saved. Shape: {pruned_df.shape}. Time: {table_prune_end - table_prune_start:.2f}s")
                            else:
                                print(f"    -> Pruned table is empty or failed JSON conversion. Skipping save.")
                        else:
                            print(f"    -> Pruned table is empty. Skipping save.")


                file_end_time = time.time()
                print(f"Finished processing {file_name}. Saved {processed_tables_count} pruned tables.")
                print(f"File execution time: {file_end_time - file_start_time:.2f} seconds")

            except RuntimeError as e:
                 # Catch CUDA out-of-memory errors specifically if possible
                 if "CUDA out of memory" in str(e):
                     print(f"FATAL ERROR processing file {file_name}: CUDA out of memory. Try reducing batch sizes or processing fewer tables if applicable.")
                     # Optionally try to clear cache and continue, or just exit/skip
                     if model_device.type == 'cuda': torch.cuda.empty_cache()
                     # Decide whether to 'continue' to next file or 'break'/'sys.exit(1)'
                     continue
                 else:
                     # Handle other runtime errors
                     print(f"Runtime error processing file {file_name}: {e}")
                     import traceback
                     print(traceback.format_exc())
                     continue # Skip to next file

            except Exception as e:
                print(f"Unexpected error processing file {file_name}: {e}")
                import traceback
                print(traceback.format_exc())
                continue # Skip to the next file

            finally:
                # Clean up GPU memory after processing each file if GPU was used
                if model_device.type == 'cuda':
                    print("Clearing GPU Cache...")
                    torch.cuda.empty_cache()
                    print("GPU Cache Cleared.")

        total_end_time = time.time()
        print("\n--- Processing complete! ---")
        print(f"Total execution time for {len(file_list)} files: {total_end_time - total_start_time:.2f} seconds")

    except Exception as e:
        # Catch fatal errors happening outside the file loop (e.g., initial setup)
        print(f"Fatal error during script execution: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1) # Exit if setup fails