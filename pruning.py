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
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Supercomputer configurations
def setup_supercomputer_config():
    # Enable GPU if available
    if torch.cuda.is_available():
        # Set to use all available GPUs
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        
        # Set GPU memory management
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set optimal chunk size for GPU memory
        CHUNK_SIZE = 32  # Adjust based on GPU memory
        
        # Enable distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
    else:
        device = torch.device("cpu")
        n_gpus = 0
        CHUNK_SIZE = 16
    
    # Set number of CPU workers for data loading
    NUM_WORKERS = os.cpu_count()
    
    # Set batch size based on available resources
    BATCH_SIZE = 64 * max(1, n_gpus)  # Scale with number of GPUs
    
    return {
        'device': device,
        'n_gpus': n_gpus,
        'num_workers': NUM_WORKERS,
        'batch_size': BATCH_SIZE,
        'chunk_size': CHUNK_SIZE
    }

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# *********embedding module********* #
# This module is used to extract the embeddings from the input sequence.
# It uses a pre-trained language model (e.g., BERT, RoBERTa) to obtain the
# token embeddings. The embeddings are then passed to the URSModule and
# WeakSupervisionModule for further processing.
#-----------------------------------#
class EmbeddingModule(nn.Module):
    def __init__(self, model_name):
        super(EmbeddingModule, self).__init__()
        self.model = model_name  # Load your pre-trained model here


 #******************Unsupervided Relevance Score Module******************#
# This module computes the unsupervised relevance score for each token
# in the input sequence using a neural network. It uses the reparameterization
# trick to sample from a normal distribution defined by the mean and standard
# deviation computed from the input embeddings.
#--------------------------------------------------------------------------#
class URSModule(nn.Module):
    def __init__(self, hidden_dim):
        super(URSModule, self).__init__()
        # Additional intermediate layers for richer feature extraction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        
        # Fully-connected layers for mean and sigma computation
        self.fc_mu = nn.Linear(hidden_dim//4, 1) 
        self.fc_sigma = nn.Linear(hidden_dim//4, 1)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, h):
        """
        h: Tensor of shape (batch_size, hidden_dim) representing token embeddings.
        Returns:
            eta_uns: Tensor of shape (batch_size, 1) representing relevance scores.
        """
        # Pass through intermediate layers with ReLU activation
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Compute mean and standard deviation
        mu = self.fc_mu(x)   # shape: (batch_size, 1)
        sigma = F.softplus(self.fc_sigma(x))  # ensures sigma > 0
        
        # Sample s from normal distribution with mean mu and std sigma
        s = torch.normal(mean=mu, std=sigma)
        
        # Reparameterization: z = mu + s * sigma
        z = mu + s * sigma
        
        # Normalize with sigmoid
        eta_uns = torch.sigmoid(z)
        return eta_uns, mu, sigma

#******************Weak Supervision Module******************#
# This module computes the weak supervision score for each token in the input
# sequence. It uses a simple linear transformation followed by a sigmoid activation.
# this score is used to guide the training of the URSModule.
# it depends on the Query what user passes to the system.
# and it gives the relevance score for each token in the input sequence.
#--------------------------------------------------------------------------#
class WeakSupervisionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(WeakSupervisionModule, self).__init__()
        # Multiple fully-connected layers with decreasing dimensions
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4) 
        self.fc3 = nn.Linear(hidden_dim//4, hidden_dim//8)
        self.fc4 = nn.Linear(hidden_dim//8, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Layer normalization 
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h):
        """
        h: Tensor of shape (batch_size, hidden_dim) representing token embeddings.
        Returns:
            eta_ws: Tensor of shape (batch_size, 1) representing weak supervision scores.
        """
        # Apply layer normalization 
        x = self.layer_norm(h)
        
        # Pass through multiple layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Final layer with sigmoid activation
        eta_ws = torch.sigmoid(self.fc4(x))  # shape: (batch_size, 1)
        return eta_ws
    
# combining the two modules
#******************Combined Module******************#
# This module combines the outputs of the URSModule and WeakSupervisionModule.
# It takes the token embeddings and computes both the unsupervised relevance
# scores and the weak supervision scores, returning the combined relevance scores.
#--------------------------------------------------------------------------#
class CombinedModule(nn.Module):
    def __init__(self, hidden_dim):
        super(CombinedModule, self).__init__()
        self.urs_module = URSModule(hidden_dim)
        self.ws_module = WeakSupervisionModule(hidden_dim)
        self.lambda_urs = 0.6  # weight for unsupervised relevance score ----> can be tuned
        self.lambda_ws = 0.4   # weight for weak supervision score  ----> can be tuned
    
    def forward(self, h):
        """
        h: Tensor of shape (batch_size, hidden_dim) representing token embeddings.
        Returns:
            eta_combined: Tensor of shape (batch_size, 1) representing combined relevance scores.
        """
        # Get unsupervised relevance scores
        eta_uns, mu, sigma = self.urs_module(h)
        
        # Get weak supervision scores
        eta_ws = self.ws_module(h)
        
        # Combine the scores (you can adjust the combination method)
        eta_combined = self.lambda_urs*eta_uns + self.lambda_ws*eta_ws
        return eta_combined, mu, sigma
    
#******************Pruning Function******************#
# This function prunes the chunks based on the combined relevance scores.
# It filters out chunks with scores below a certain threshold.
#--------------------------------------------------------------------------#    
    
def prune_chunks(chunks, scores, threshold=0.6):
    """
    chunks: list of chunk dicts.
    scores: list of combined relevance scores corresponding to each chunk.
    threshold: minimum score to keep a chunk (default 0.6).
    Returns a list of pruned chunks.
    """
    pruned = []
    for chunk, score in zip(chunks, scores):
        if score >= threshold:  # Only keep chunks with scores >= threshold
            chunk['score'] = score  # Add the score to the chunk for reference
            pruned.append(chunk)
    return pruned

# Helper function to get a list of available tables
def list_available_tables(limit=10):
    tables = []
    with jsonlines.open('tables.jsonl', 'r') as reader:
        for i, table in enumerate(reader):
            if i >= limit:
                break;
            tables.append({
                'id': table['tableId'],
                'title': table.get('documentTitle', 'Unknown title')
            })
    return tables

# Convert chunks to pandas DataFrame for visualization
def chunks_to_dataframe(chunks, is_pruned_chunks=False, normalized_scores=None, threshold=0.6):
    """Convert chunks to a pandas DataFrame for better visualization."""
    data = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
        chunk_id = chunk['metadata'].get('chunk_id', f'chunk_{i}')
        
        # Extract additional metadata if available
        col_id = chunk['metadata'].get('col_id', '')
        row_id = chunk['metadata'].get('row_id', '')
        
        # Add score and pruning status
        score = chunk.get('score', normalized_scores[i] if normalized_scores else 0.0)
        is_pruned = "Yes" if score < threshold else "No"  # Use passed threshold
        
        data.append({
            'chunk_id': chunk_id,
            'chunk_type': chunk_type,
            'col_id': col_id,
            'row_id': row_id,
            'text': chunk_text,
            'score': score,
            'is_pruned': is_pruned
        })
    
    return pd.DataFrame(data)

def load_pruning_test_data(directory='pruningTestData'):
    """
    Load queries from the pruningTestData directory.
    Each file should contain query, target table ID, and expected answer.
    Returns a list of test items.
    """
    test_items = []
    
    try:
        # Make sure the directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return test_items
            
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for file in files:
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        # If the file contains a single query
                        if 'question' in data and 'tableId' in data:
                            test_items.append({
                                'question': data['question'],
                                'table_id': data['tableId'],
                                'answer': data.get('answer', [])
                            })
                    elif isinstance(data, list):
                        # If the file contains multiple queries
                        for item in data:
                            if 'question' in item and 'tableId' in item:
                                test_items.append({
                                    'question': item['question'],
                                    'table_id': item['tableId'],
                                    'answer': item.get('answer', [])
                                })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
                
        print(f"Loaded {len(test_items)} queries from {directory}")
    except Exception as e:
        print(f"Error loading pruning test data: {e}")
    
    return test_items

def extract_table_ids_from_test_jsonl():
    """Extract table IDs from test.jsonl file."""
    table_ids = []
    table_info = []
    
    try:
        with jsonlines.open('test.jsonl', 'r') as reader:
            for item in reader:
                if 'table' in item and 'tableId' in item['table']:
                    table_id = item['table']['tableId']
                    title = item['table'].get('documentTitle', 'Unknown title')
                    
                    # Only add if not already in the list
                    if table_id not in [t['id'] for t in table_info]:
                        table_info.append({
                            'id': table_id,
                            'title': title,
                            'questions': [q['originalText'] for q in item.get('questions', [])]
                        })
                        table_ids.append(table_id)
    except Exception as e:
        print(f"Error reading test.jsonl: {e}")
    
    return table_info, table_ids

def merge_pruned_chunks(pruned_chunks):
    """Merge pruned chunks that belong to the same row/column"""
    merged = []
    row_chunks = {}
    col_chunks = {}
    
    for chunk in pruned_chunks:
        chunk_type = chunk['metadata'].get('chunk_type', '')
        
        if chunk_type == 'row':
            row_id = chunk['metadata'].get('row_id', '')
            if row_id not in row_chunks:
                row_chunks[row_id] = []
            row_chunks[row_id].append(chunk)
        elif chunk_type == 'col':
            col_id = chunk['metadata'].get('col_id', '')
            if col_id not in col_chunks:
                col_chunks[col_id] = []
            col_chunks[col_id].append(chunk)
        else:
            # Keep non-row/col chunks as is
            merged.append(chunk)
    
    # Merge row chunks
    for row_id, chunks in row_chunks.items():
        merged_text = " ".join([str(c['text']) for c in chunks])
        merged_score = sum([c.get('score', 0) for c in chunks]) / len(chunks)
        merged.append({
            'text': merged_text,
            'score': merged_score,
            'metadata': {
                'chunk_type': 'merged_row',
                'row_id': row_id,
                'original_chunks': len(chunks)
            }
        })
    
    # Merge column chunks
    for col_id, chunks in col_chunks.items():
        merged_text = " ".join([str(c['text']) for c in chunks])
        merged_score = sum([c.get('score', 0) for c in chunks]) / len(chunks)
        merged.append({
            'text': merged_text,
            'score': merged_score,
            'metadata': {
                'chunk_type': 'merged_col',
                'col_id': col_id,
                'original_chunks': len(chunks)
            }
        })
    
    return merged

def process_top_queries(query_range, top_n_tables):
    """Process queries from Top-150-Queries directory"""
    try:
        # Create results directory structure
        base_dir = "pruning_results"
        for subdir in ['original', 'pruned', 'merged', 'reports']:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        
        # Load queries from the specified range
        queries_dir = "Top-150-Quries"
        
        # Get the specific query files based on query numbers
        selected_files = []
        for query_num in range(query_range[0], query_range[1] + 1):  # Include end query number
            query_file = f"query{query_num}_TopTables.csv"
            if os.path.exists(os.path.join(queries_dir, query_file)):
                selected_files.append(query_file)
            else:
                print(f"Warning: Query file {query_file} not found")
        
        if not selected_files:
            print(f"No query files found for range {query_range[0]} to {query_range[1]}")
            return []
        
        print(f"\nProcessing queries {query_range[0]} to {query_range[1]}")
        print(f"Found {len(selected_files)} query files")
        print(f"Will process top {top_n_tables} tables for each query")
        
        results = []
        
        for query_file in selected_files:
            query_num = query_file.split('_')[0].replace('query', '')
            print(f"\nProcessing Query {query_num}...")
            
            # Read the query CSV file
            query_df = pd.read_csv(os.path.join(queries_dir, query_file))
            
            # Get the query text (same for all rows)
            query_text = query_df['query'].iloc[0]
            target_table = query_df['target table'].iloc[0]
            target_answer = query_df['target answer'].iloc[0]
            
            print(f"Query: {query_text}")
            print(f"Target table: {target_table}")
            print(f"Target answer: {target_answer}")
            
            # Take exactly top N tables
            top_tables = query_df.head(top_n_tables)
            print(f"Processing top {len(top_tables)} tables for this query")
            
            for idx, row in top_tables.iterrows():
                table_id = row['top tables']
                print(f"\nProcessing table {idx + 1}/{top_n_tables}: {table_id}")
                
                # Load chunks for this table
                chunks = []
                with jsonlines.open('chunks.json', 'r') as reader:
                    for chunk in reader:
                        if ('metadata' in chunk and 
                            'table_name' in chunk['metadata'] and 
                            chunk['metadata']['table_name'] == table_id and
                            'chunk_type' in chunk['metadata'] and 
                            chunk['metadata']['chunk_type'] != 'table'):  # Skip table chunks
                            chunks.append(chunk)
                
                if not chunks:
                    print(f"No valid chunks found for table {table_id}")
                    continue
                
                # Create query item
                query_item = {
                    'question': query_text,
                    'table_id': table_id,
                    'target_table': target_table,
                    'answer': target_answer
                }
                
                # Process query
                result = process_query(
                    query_item,
                    chunks,
                    embedding_model,
                    combined_model,
                    tokenizer,
                    thresholds,
                    final_threshold
                )
                
                # Save results in pruning_results directory
                if result:
                    # Create merged version of pruned chunks
                    pruned_chunks_file = os.path.join(base_dir, "pruned", 
                                                    f"{query_num}_{table_id}_pruned.json")
                    if os.path.exists(pruned_chunks_file):
                        with open(pruned_chunks_file, 'r') as f:
                            pruned_chunks = json.load(f)
                            merged_chunks = merge_pruned_chunks(pruned_chunks)
                            
                            # Save merged chunks
                            merged_file = os.path.join(base_dir, "merged", 
                                                     f"{query_num}_{table_id}_merged.json")
                            with open(merged_file, 'w') as f:
                                json.dump(merged_chunks, f, indent=2)
                    
                    results.append(result)
        
        # Save summary for this batch
        if results:
            summary_df = pd.DataFrame([{
                'query_num': r['question'],
                'table_id': r['table_id'],
                'target_table': r['target_table'],
                'original_chunks': r['original_chunks'],
                'pruned_chunks': r['pruned_chunks'],
                'reduction_percentage': r['reduction_percentage']
            } for r in results])
            
            summary_path = os.path.join(base_dir, f"summary_queries_{query_range[0]}-{query_range[1]}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to {summary_path}")
        
        return results
    
    except Exception as e:
        print(f"Error processing queries: {e}")
        return []

# Process a single query and its associated table
def process_query(query_item, chunks, embedding_model, combined_model, tokenizer, thresholds, final_threshold):
    """
    Process a single query against its table chunks.
    
    Parameters:
    - query_item: dict containing 'question', 'table_id', and 'answer'
    - chunks: list of chunks for the associated table
    - embedding_model, combined_model, tokenizer: pre-loaded models
    - thresholds: list of thresholds to test
    - final_threshold: threshold to use for saving pruned chunks
    
    Returns:
    - Dictionary with processing results
    """
    question = query_item['question']
    table_id = query_item['table_id']
    expected_answer = query_item['answer']
    
    # Get device from model
    device = next(combined_model.parameters()).device
    
    print(f"Processing question: {question}")
    print(f"Expected answer(s): {expected_answer}")
    
    # Tokenize and get embeddings for chunks and question
    print("Computing embeddings for chunks...")
    start_time = time.time()
    
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
        
        # Combine chunk content into a single string
        if isinstance(chunk['text'], list):
            chunk_text = " ".join([str(cell) for cell in chunk['text']])
        else:
            chunk_text = str(chunk['text'])
        
        inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to the right device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1)
            chunk_embeddings.append(embeddings)
    
    print(f"Generated embeddings for {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
    
    # Get question embedding
    print("Computing embedding for question...")
    question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    # Move question inputs to the right device
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    
    with torch.no_grad():
        question_embedding = embedding_model.model(**question_inputs).last_hidden_state.mean(dim=1)
    
    chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
    
    print("Computing relevance scores...")
    # Get combined relevance scores using both question and chunk embeddings
    scores, _, _ = combined_model(chunk_embeddings)
    scores = scores.squeeze().cpu().tolist()  # Move to CPU before converting to Python list
    
    # Scale scores based on similarity with question
    similarity_scores = torch.nn.functional.cosine_similarity(
        chunk_embeddings, 
        question_embedding.expand(chunk_embeddings.size(0), -1)
    ).cpu().tolist()  # Move to CPU before converting to Python list
    
    print(f"Computing similarity scores... done!")
    
    # Combine URS scores with question similarity
    final_scores = [(s + sim) / 2 for s, sim in zip(scores, similarity_scores)]
    
    # Apply score normalization to spread out the scores
    min_score = min(final_scores)
    max_score = max(final_scores)
    if max_score > min_score:  # Avoid division by zero
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in final_scores]
    else:
        normalized_scores = final_scores
    
    # Print score statistics for both raw and normalized scores
    print(f"Raw score statistics - Min: {min(final_scores):.4f}, Max: {max(final_scores):.4f}, Avg: {sum(final_scores)/len(final_scores):.4f}")
    print(f"Normalized score statistics - Min: {min(normalized_scores):.4f}, Max: {max(normalized_scores):.4f}, Avg: {sum(normalized_scores)/len(normalized_scores):.4f}")
    
    # Show top 5 highest scoring chunks
    top_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:5]
    print("Top 5 chunks:")
    for i, idx in enumerate(top_indices):
        chunk_preview = str(chunks[idx]['text'])[:50] + "..." if len(str(chunks[idx]['text'])) > 50 else str(chunks[idx]['text'])
        chunk_type = chunks[idx]['metadata']['chunk_type'] if 'metadata' in chunks[idx] and 'chunk_type' in chunks[idx]['metadata'] else "unknown"
        print(f"  {i+1}. Score: {normalized_scores[idx]:.4f} (raw: {final_scores[idx]:.4f}) - Type: {chunk_type} - {chunk_preview}")
    
    # Prune chunks based on normalized scores with different thresholds
    threshold_results = {}
    for threshold in thresholds:
        pruned_chunks = prune_chunks(chunks, normalized_scores, threshold)
        print(f"Pruning threshold: {threshold:.1f}")
        print(f"Chunks with scores above threshold: {len(pruned_chunks)}/{len(chunks)} ({len(pruned_chunks)/len(chunks)*100:.1f}%)")
        threshold_results[threshold] = len(pruned_chunks)
    
    # Use final_threshold for saving
    pruned_chunks = prune_chunks(chunks, normalized_scores, final_threshold)
    
    # Create a unique and safe filename based on question and table ID
    safe_question = "".join(c if c.isalnum() else "_" for c in question[:20])
    output_filename = f"pruned_chunks_{table_id}_{safe_question}.json"
    
    # Save pruned chunks in pruning_results/pruned directory
    pruned_chunks_file = os.path.join("pruning_results", "pruned", output_filename)
    os.makedirs(os.path.dirname(pruned_chunks_file), exist_ok=True)
    with open(pruned_chunks_file, 'w') as f:
        json.dump(pruned_chunks, f, indent=2)
    
    print(f"Original chunks: {len(chunks)}, Pruned chunks: {len(pruned_chunks)}")
    print(f"Saved pruned chunks to {pruned_chunks_file}")
    
    # Create and save DataFrames for visualization
    print("Converting chunks to DataFrames for visualization...")
    
    # Convert chunks to DataFrames
    original_df = chunks_to_dataframe(chunks)
    original_df['raw_score'] = final_scores
    original_df['normalized_score'] = normalized_scores
    pruned_df = chunks_to_dataframe(pruned_chunks)
    
    # Save DataFrames to CSV in pruning_results/original and pruning_results/pruned
    csv_base = os.path.join("pruning_results")
    original_csv = os.path.join(csv_base, "original", f"{table_id}_{safe_question}_original.csv")
    pruned_csv = os.path.join(csv_base, "pruned", f"{table_id}_{safe_question}_pruned.csv")
    
    os.makedirs(os.path.dirname(original_csv), exist_ok=True)
    os.makedirs(os.path.dirname(pruned_csv), exist_ok=True)
    
    original_df.to_csv(original_csv, index=False)
    pruned_df.to_csv(pruned_csv, index=False)
    
    print(f"Saved original chunks DataFrame to {original_csv}")
    print(f"Saved pruned chunks DataFrame to {pruned_csv}")
    
    # Generate HTML report in pruning_results/reports directory
    html_file = os.path.join(csv_base, "reports", f"{table_id}_{safe_question}_report.html")
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    
    try:
        # Create a styled DataFrame with conditional formatting for scores
        styled_df = original_df.copy()
        styled_df['pruned'] = styled_df['chunk_id'].isin(pruned_df['chunk_id']).map({True: 'Yes', False: 'No'})
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Pruning Report for {question}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .stats {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                .table-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .kept {{ background-color: #d4edda; }}
                .pruned {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Pruning Report</h1>
            <p>Question: {question}</p>
            <p>Expected answer(s): {expected_answer}</p>
            
            <div class="stats">
                <h2>Statistics</h2>
                <p>Total chunks: {len(chunks)}</p>
                <p>Pruned chunks: {len(pruned_chunks)} ({len(pruned_chunks)/len(chunks)*100:.1f}%)</p>
                <p>Pruning threshold: {final_threshold}</p>
                <p>Raw score statistics - Min: {min(final_scores):.4f}, Max: {max(final_scores):.4f}, Avg: {sum(final_scores)/len(final_scores):.4f}</p>
                <p>Normalized score statistics - Min: {min(normalized_scores):.4f}, Max: {max(normalized_scores):.4f}, Avg: {sum(normalized_scores)/len(normalized_scores):.4f}</p>
            </div>
            
            <div class="table-container">
                <h2>Chunk Data with Scores</h2>
        """
        
        try:
            # Try to use pandas styling with the correct method
            # For older pandas versions, we need to use render() instead of to_html()
            styled_df_style = styled_df.style.apply(
                lambda row: ['background-color: #d4edda' if row['pruned'] == 'Yes' else 'background-color: #f8d7da' for _ in row], 
                axis=1
            )
            
            # Try different methods to convert to HTML based on pandas version
            try:
                styled_html = styled_df_style.to_html()  # Newer pandas versions
            except AttributeError:
                try:
                    styled_html = styled_df_style.render()  # Older pandas versions
                except AttributeError:
                    # If neither method works, fall back to basic table
                    raise ImportError("Pandas styling methods not available")
                    
            html_content += styled_html
        except ImportError:
            # Fall back to basic HTML table if Jinja2 is not available
            print("Warning: Jinja2 not available. Using basic HTML table instead.")
            table_html = "<table border='1'>\n"
            
            # Add header row
            table_html += "<tr>"
            for col in styled_df.columns:
                table_html += f"<th>{col}</th>"
            table_html += "</tr>\n"
            
            # Add data rows
            for _, row in styled_df.iterrows():
                bg_color = "#d4edda" if row['pruned'] == 'Yes' else "#f8d7da"
                table_html += f"<tr style='background-color: {bg_color}'>"
                for col in styled_df.columns:
                    table_html += f"<td>{row[col]}</td>"
                table_html += "</tr>\n"
            
            table_html += "</table>"
            html_content += table_html
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Generated HTML pruning report: {html_file}")
    except Exception as e:
        print(f"Warning: Could not generate HTML report due to error: {e}")
        print("CSV files were still generated successfully.")
    
    # Return results for summary
    return {
        'question': question,
        'table_id': table_id,
        'expected_answer': expected_answer,
        'original_chunks': len(chunks),
        'pruned_chunks': len(pruned_chunks),
        'reduction_percentage': (1 - len(pruned_chunks)/len(chunks)) * 100 if len(chunks) > 0 else 0,
        'threshold_results': threshold_results,
        'output_path': pruned_chunks_file,
        'html_report': html_file
    }

# Main execution block
if __name__ == "__main__":
    print("TRIM-QA Pruning Script")
    print("=" * 50)
    
    # Setup supercomputer configuration to use GPU if available
    config = setup_supercomputer_config()
    device = config['device']
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Initialize models
    hidden_dim = 768
    model_name = "bert-base-uncased"
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = EmbeddingModule(AutoModel.from_pretrained(model_name))
    
    # Move models to the appropriate device (GPU if available)
    embedding_model.model = embedding_model.model.to(device)
    combined_model = CombinedModule(hidden_dim).to(device)
    
    # Define thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5]
    final_threshold = 0.6
    
    # Get query range from user
    while True:
        try:
            print("\nEnter query range (1-150, format: start end):")
            start, end = map(int, input("> ").split())
            if 1 <= start <= end <= 150:
                break
            print("Invalid range. Start must be <= end, and both must be between 1 and 150.")
        except ValueError:
            print("Please enter two numbers separated by space.")
    
    # Get number of top tables to process
    while True:
        try:
            print("\nEnter number of top tables to process for each query:")
            top_n = int(input("> "))
            if top_n > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Process queries
    print(f"\nProcessing queries {start}-{end} with top {top_n} tables each...")
    results = process_top_queries((start, end), top_n)
    
    # Generate final summary
    if results:
        print("\n===== Final Summary =====")
        print(f"Processed {len(results)} table-query pairs")
        avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
        print(f"Average chunk reduction: {avg_reduction:.1f}%")
        
        # Save summary to CSV
        summary_df = pd.DataFrame([{
            'query': r['question'],
            'table_id': r['table_id'],
            'original_chunks': r['original_chunks'],
            'pruned_chunks': r['pruned_chunks'],
            'reduction_percentage': r['reduction_percentage']
        } for r in results])
        
        summary_path = "pruning_results/summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
    else:
        print("No results generated. Please check the errors above.")


