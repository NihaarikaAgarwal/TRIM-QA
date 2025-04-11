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

# Configuration settings
USE_BATCHED_PROCESSING = True        # Process chunks in batches instead of one by one
ENABLE_CACHING = True                # Cache embeddings to avoid recomputation
GENERATE_FULL_REPORTS = False        # Generate detailed HTML reports (slower)
USE_EARLY_STOPPING = True            # Try to detect high-relevance chunks early
USE_PARALLEL_PROCESSING = True       # Process multiple tables in parallel
USE_MODEL_QUANTIZATION = False       # Use int8 quantization (faster but less precise)
BATCH_SIZE = 64                      # Number of chunks to process at once
MAX_CHUNKS_PER_TABLE = 5000         # Limit number of chunks per table (set to -1 for no limit)

# Global cache for chunk embeddings
chunk_embedding_cache = {}

def optimize_models_for_inference(model, device):
    """Optimize models for faster inference"""
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    # Enable CUDA optimizations if available
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
    
    # Use mixed precision where applicable
    if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
        model = torch.cuda.amp.autocast()(model)
    
    return model

def quantize_model(model):
    """Quantize model to int8 for faster inference and lower memory footprint"""
    if not USE_MODEL_QUANTIZATION:
        return model
        
    try:
        import torch.quantization
        
        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
        print("Model successfully quantized to int8")
        return model
    except Exception as e:
        print(f"Quantization failed with error: {e}")
        print("Continuing with full precision model")
        return model

def compute_chunk_embeddings_batch(chunks, embedding_model, tokenizer, device, batch_size=BATCH_SIZE):
    """Process chunks in batches for faster embedding computation"""
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = []
        
        for chunk in batch_chunks:
            if isinstance(chunk['text'], list):
                chunk_text = " ".join([str(cell) for cell in chunk['text']])
            else:
                chunk_text = str(chunk['text'])
            batch_texts.append(chunk_text)
        
        # Process the entire batch at once
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Process entire batch in one forward pass using the forward method
            embeddings = embedding_model(**inputs)
            all_embeddings.append(embeddings)
            
        # Print progress at reasonable intervals
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(chunks):
            print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])

def get_chunk_embedding(chunk, embedding_model, tokenizer, device):
    """Get embedding for a single chunk, using cache if available"""
    if not ENABLE_CACHING:
        return compute_chunk_embeddings_batch([chunk], embedding_model, tokenizer, device)[0]
        
    # Create a cache key based on chunk content
    if isinstance(chunk['text'], list):
        chunk_text = " ".join([str(cell) for cell in chunk['text']])
    else:
        chunk_text = str(chunk['text'])
    
    cache_key = hash(chunk_text)
    
    # Return cached embedding if available
    if cache_key in chunk_embedding_cache:
        return chunk_embedding_cache[cache_key]
    
    # Compute embedding if not in cache
    embedding = compute_chunk_embeddings_batch([chunk], embedding_model, tokenizer, device)[0]
    chunk_embedding_cache[cache_key] = embedding
    return embedding

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
    def __init__(self, model):
        super(EmbeddingModule, self).__init__()
        self.model = model  # Store the pre-trained model

    def forward(self, **inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the last hidden state (CLS token) as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings


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
        print(f"Will process top {len(selected_files)} tables for this query")
        
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
    - query_item: dict containing 'question', 'table_id', and target information
    - chunks: list of chunks for the associated table
    - embedding_model, combined_model, tokenizer: pre-loaded models
    - thresholds: list of thresholds to test
    - final_threshold: threshold to use for saving pruned chunks
    
    Returns:
    - Dictionary with processing results
    """
    question = query_item['question']
    table_id = query_item['table_id']
    # Handle both 'answer' and 'target_answer' fields
    expected_answer = query_item.get('answer', query_item.get('target_answer', ['Unknown']))
    
    print(f"Processing question: {question}")
    print(f"Expected answer(s): {expected_answer}")
    
    # Filter chunks to keep only row chunks
    row_chunks = [chunk for chunk in chunks if chunk['metadata'].get('chunk_type') == 'row']
    print(f"Filtered {len(chunks)} total chunks to {len(row_chunks)} row chunks")
    
    # Process chunks in batches using the optimized function
    print("Computing embeddings for row chunks...")
    start_time = time.time()
    
    # Use the batch processing function with progress tracking
    chunk_embeddings = compute_chunk_embeddings_batch(row_chunks, embedding_model, tokenizer, device)
    
    print(f"Generated embeddings for {len(row_chunks)} row chunks in {time.time() - start_time:.2f} seconds")
    
    # Get question embedding
    print("Computing embedding for question...")
    question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    
    with torch.no_grad():
        question_embedding = embedding_model(**question_inputs)
    
    print("Computing relevance scores...")
    # Compute scores with the combined model
    with torch.no_grad():
        scores, _, _ = combined_model(chunk_embeddings)
        scores = scores.squeeze().cpu().tolist()
    
    # Calculate similarity with question embedding
    question_embedding_expanded = question_embedding.expand(chunk_embeddings.size(0), -1)
    similarity_scores = F.cosine_similarity(
        chunk_embeddings, question_embedding_expanded
    ).cpu().tolist()
    
    # Combine scores
    final_scores = [(s + sim) / 2 for s, sim in zip(scores, similarity_scores)]
    
    # Normalize scores
    min_score = min(final_scores)
    max_score = max(final_scores)
    if max_score > min_score:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in final_scores]
    else:
        normalized_scores = [0.5] * len(final_scores)  # Default to 0.5 if all scores are equal
    
    # Print score statistics
    print(f"Raw score stats - Min: {min(final_scores):.4f}, Max: {max(final_scores):.4f}, Avg: {sum(final_scores)/len(final_scores):.4f}")
    print(f"Normalized stats - Min: {min(normalized_scores):.4f}, Max: {max(normalized_scores):.4f}, Avg: {sum(normalized_scores)/len(normalized_scores):.4f}")
    
    # Prune chunks based on normalized scores with the specified threshold
    pruned_chunks = prune_chunks(row_chunks, normalized_scores, threshold=final_threshold)
    print(f"Pruning threshold: {final_threshold:.2f}")
    print(f"Kept chunks: {len(pruned_chunks)}/{len(row_chunks)} ({len(pruned_chunks)/len(row_chunks)*100:.1f}%)")
    
    # Return results for summary
    return {
        'question': question,
        'table_id': table_id,
        'target_table': query_item.get('target_table', 'Unknown'),
        'target_answer': expected_answer,
        'original_chunks': len(row_chunks),
        'pruned_chunks': pruned_chunks,
        'pruned_chunks_count': len(pruned_chunks),
        'reduction_percentage': (1 - len(pruned_chunks)/len(row_chunks)) * 100 if len(row_chunks) > 0 else 0,
        'normalized_scores': normalized_scores
    }

def process_query_with_early_stopping(query_item, chunks, embedding_model, combined_model, tokenizer, device, threshold, 
                                     early_stopping_threshold=0.8):
    """Process query with early stopping if top chunks have very high scores."""
    # First, process a small sample of chunks to see if we can stop early
    sample_size = min(50, len(chunks))
    sample_chunks = chunks[:sample_size]
    
    start_time = time.time()
    print(f"Testing early stopping with {sample_size} chunks...")
    
    # Process question
    question = query_item['question']
    question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    
    # Handle both 'answer' and 'target_answer' fields
    expected_answer = query_item.get('answer', query_item.get('target_answer', ['Unknown']))
    
    with torch.no_grad():
        question_embedding = embedding_model(**question_inputs)
    
    # Compute embeddings for sample in batch
    sample_embeddings = compute_chunk_embeddings_batch(sample_chunks, embedding_model, tokenizer, device)
    
    with torch.no_grad():
        sample_scores, _, _ = combined_model(sample_embeddings)
        sample_scores = sample_scores.squeeze().cpu().tolist()
    
    # If top scores are very high, we might be able to stop
    top_sample_scores = sorted(sample_scores, reverse=True)[:10]
    if len(top_sample_scores) > 0 and min(top_sample_scores) > early_stopping_threshold:
        print(f"Early stopping triggered after {time.time() - start_time:.2f}s - found {len(top_sample_scores)} high-scoring chunks!")
        
        # Keep only the top scoring chunks
        top_indices = sorted(range(len(sample_scores)), key=lambda i: sample_scores[i], reverse=True)[:10]
        pruned_chunks = [sample_chunks[i] for i in top_indices]
        
        # Add scores to the chunks
        for i, chunk in enumerate(pruned_chunks):
            chunk['score'] = sample_scores[top_indices[i]]
        
        return {
            'question': query_item['question'],
            'table_id': query_item['table_id'],
            'target_table': query_item.get('target_table', 'Unknown'),
            'target_answer': expected_answer,
            'original_chunks': len(chunks),
            'pruned_chunks': len(pruned_chunks),
            'reduction_percentage': (1 - len(pruned_chunks)/len(chunks)) * 100,
            'early_stopped': True,
            'pruned_chunks': pruned_chunks  # Include pruned chunks in the return value
        }
    
    print("No early stopping, continuing with full processing...")
    return None

def save_pruning_results(query, table_id, original_chunks, pruned_chunks, threshold, target_table=None, target_answer=None, pruning_dir="pruning_results"):
    """Save pruning results with optimized report generation."""
    # Create a query-specific directory name (first 30 chars of query, sanitized)
    query_dir_name = query[:30].replace(' ', '_').replace('?', '').replace('/', '_')
    table_dir_name = table_id
    
    # Create directory structure
    query_dir = os.path.join(pruning_dir, query_dir_name)
    table_dir = os.path.join(query_dir, table_dir_name)
    ensure_directory(table_dir)
    
    base_filename = "results"
    
    # Always save pruned chunks as this is essential
    pruned_path = os.path.join(table_dir, f"{base_filename}_pruned.json")
    with open(pruned_path, 'w') as f:
        json.dump(pruned_chunks, f)
    
    if not GENERATE_FULL_REPORTS:
        # Return early if full reports are disabled
        return {
            'pruned_path': os.path.join(query_dir_name, table_dir_name, f"{base_filename}_pruned.json")
        }
    
    # Get scores for chunk visualization
    pruned_chunk_ids = [chunk['metadata'].get('chunk_id') for chunk in pruned_chunks]
    
    # Create DataFrames for original and pruned chunks
    original_df = chunks_to_dataframe(original_chunks, normalized_scores=[chunk.get('score', 0.0) for chunk in original_chunks], threshold=threshold)
    original_df['is_pruned'] = original_df['chunk_id'].apply(lambda x: "No" if x in pruned_chunk_ids else "Yes")
    
    # Save CSV files
    original_path = os.path.join(table_dir, f"{base_filename}_original.csv")
    original_df.to_csv(original_path, index=False)
    
    # Generate simplified HTML report
    report_path = os.path.join(table_dir, f"{base_filename}_report.html")
    
    # Calculate statistics
    total_chunks = len(original_chunks)
    pruned_count = len(pruned_chunks)
    reduction_pct = ((total_chunks - pruned_count) / total_chunks * 100) if total_chunks > 0 else 0
    
    # Generate basic HTML report
    html_content = f"""
    <html>
    <head>
        <title>Pruning Report for {query}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .stats {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .pruned {{ background-color: #f8d7da; }}
            .kept {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <h1>Pruning Summary</h1>
        <div class="stats">
            <p>Query: {query}</p>
            <p>Total chunks: {total_chunks}</p>
            <p>Pruned chunks: {pruned_count}</p>
            <p>Reduction: {reduction_pct:.1f}%</p>
            <p>Threshold: {threshold}</p>
            <p>Target Table: {target_table or "Not specified"}</p>
            <p>Target Answer: {target_answer or "Not specified"}</p>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return {
        'original_path': os.path.join(query_dir_name, table_dir_name, f"{base_filename}_original.csv"),
        'pruned_path': os.path.join(query_dir_name, table_dir_name, f"{base_filename}_pruned.json"),
        'report_path': os.path.join(query_dir_name, table_dir_name, f"{base_filename}_report.html")
    }

def get_user_input():
    """Get user input for number of queries, tables to process, and pruning threshold."""
    try:
        num_queries = input("Enter number of queries to test (press Enter for all queries): ").strip()
        num_queries = int(num_queries) if num_queries else None
        
        num_tables = input("Enter number of tables to test per query (press Enter for all tables): ").strip()
        num_tables = int(num_tables) if num_tables else None
        
        threshold = input("Enter pruning threshold (0.0 to 1.0, press Enter for default 0.6): ").strip()
        threshold = float(threshold) if threshold else 0.6
        
        # Validate threshold
        if not 0 <= threshold <= 1:
            print("Invalid threshold. Using default value of 0.6")
            threshold = 0.6
        
        print(f"\nConfiguration:")
        print(f"Number of queries: {'All' if num_queries is None else num_queries}")
        print(f"Tables per query: {'All' if num_tables is None else num_tables}")
        print(f"Pruning threshold: {threshold}")
        
        return num_queries, num_tables, threshold
    except ValueError:
        print("Invalid input. Using all queries, all tables, and default threshold of 0.6")
        return None, None, 0.6

def process_query_file(file_path):
    """Extract query information from a query CSV file.
    
    Args:
        file_path: Path to the CSV file containing query and table information
        
    Returns:
        tuple: (query string, list of table info dictionaries)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get the query (same for all rows)
        query = df['query'].iloc[0]
        
        # Process each table
        tables_info = []
        for _, row in df.iterrows():
            table_info = {
                'table_id': row['top tables'],
                'target_table': row['target table'],
                'target_answer': row['target answer']
            }
            tables_info.append(table_info)
        
        print(f"Processed query file: {file_path}")
        print(f"Query: {query}")
        print(f"Found {len(tables_info)} tables")
        
        return query, tables_info
    except Exception as e:
        print(f"Error processing query file {file_path}: {e}")
        return "", []

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_query_numbers(query_numbers, tables_per_query, embedding_model, combined_model, tokenizer, device, threshold):
    """Process specific query numbers from Top-150-Quries directory.
    
    Args:
        query_numbers: List of query numbers to process (e.g., [1, 2])
        tables_per_query: Number of top tables to process per query
        embedding_model, combined_model, tokenizer: Models for embedding and scoring
        device: Device to run models on
        threshold: Pruning threshold
        
    Returns:
        Dictionary of results
    """
    results = []
    table_scores = {}  # Dictionary to store average scores per table
    
    # Get all query files
    query_dir = "Top-150-Quries"
    all_query_files = sorted([f for f in os.listdir(query_dir) 
                             if f.startswith("query") and f.endswith("_TopTables.csv")])
    
    # Filter for specific query numbers
    query_files = []
    for query_num in query_numbers:
        # Find the corresponding file
        pattern = f"query{query_num}_"
        matching_files = [f for f in all_query_files if f.startswith(pattern)]
        
        if matching_files:
            query_files.append(matching_files[0])
        else:
            print(f"Warning: Query number {query_num} not found in {query_dir}")
    
    if not query_files:
        print(f"No query files found for the specified query numbers: {query_numbers}")
        return {"results": [], "table_scores": {}}
    
    print(f"\nProcessing {len(query_files)} query files: {query_files}")
    
    # Load chunks once at the start
    print("Loading chunks from chunks.json...")
    all_chunks = {}
    with jsonlines.open('chunks.json', 'r') as reader:
        for chunk in reader:
            if 'metadata' in chunk and 'table_name' in chunk['metadata']:
                table_id = chunk['metadata']['table_name']
                if chunk['metadata'].get('chunk_type') != 'table':  # Skip table chunks
                    if table_id not in all_chunks:
                        all_chunks[table_id] = []
                    all_chunks[table_id].append(chunk)
    
    print(f"Loaded chunks for {len(all_chunks)} tables")
    
    # Process each query file
    for query_file in query_files:
        query_num = query_file.split('_')[0].replace('query', '')
        print(f"\nProcessing {query_file} (Query {query_num})")
        
        query, tables_info = process_query_file(os.path.join(query_dir, query_file))
        
        # Take only the specified number of tables
        if tables_per_query and tables_per_query > 0:
            tables_info = tables_info[:tables_per_query]
        
        print(f"Processing {len(tables_info)} tables for query: {query}")
        
        # Process each table for this query
        for table_info in tables_info:
            table_id = table_info['table_id']
            target_table = table_info['target_table']
            target_answer = table_info['target_answer']
            
            print(f"\nProcessing table {table_id}")
            print(f"Target table: {target_table}")
            print(f"Target answer: {target_answer}")
            
            chunks = all_chunks.get(table_id, [])
            
            if not chunks:
                print(f"No chunks found for table {table_id}")
                continue
            
            # Limit chunks if needed
            if MAX_CHUNKS_PER_TABLE > 0:
                chunks = chunks[:MAX_CHUNKS_PER_TABLE]
            
            # Try early stopping first if enabled
            if USE_EARLY_STOPPING:
                early_result = process_query_with_early_stopping(
                    {'question': query, 'table_id': table_id, 'target_table': target_table, 'target_answer': target_answer},
                    chunks,
                    embedding_model,
                    combined_model,
                    tokenizer,
                    device,
                    threshold
                )
                
                if early_result:
                    results.append(early_result)
                    
                    # Calculate and store average score for this table
                    avg_score = sum(chunk.get('score', 0) for chunk in early_result.get('pruned_chunks', [])) / len(early_result.get('pruned_chunks', []))
                    table_scores[table_id] = {
                        'avg_score': avg_score,
                        'query': query,
                        'is_target': (table_id == target_table),
                        'target_table': target_table,
                        'target_answer': target_answer
                    }
                    continue
            
            # Process normally if early stopping didn't succeed
            print(f"Processing table {table_id} ({len(chunks)} chunks)...")
            try:
                result = process_query(
                    {'question': query, 'table_id': table_id, 'target_table': target_table, 'target_answer': target_answer},
                    chunks,
                    embedding_model,
                    combined_model,
                    tokenizer,
                    [threshold],  # Use single threshold
                    threshold
                )
                
                if result:
                    results.append(result)
                    
                    # Calculate and store average score for this table
                    avg_score = sum(chunk.get('score', 0) for chunk in result.get('pruned_chunks', [])) / len(result.get('pruned_chunks', []))
                    table_scores[table_id] = {
                        'avg_score': avg_score,
                        'query': query,
                        'is_target': (table_id == target_table),
                        'target_table': target_table,
                        'target_answer': target_answer
                    }
                    
                    # Save results with target information
                    save_pruning_results(
                        query=query,
                        table_id=table_id,
                        original_chunks=chunks,
                        pruned_chunks=result['pruned_chunks'],
                        threshold=threshold,
                        target_table=target_table,
                        target_answer=target_answer
                    )
            except Exception as e:
                print(f"Error processing table {table_id}: {e}")
                continue
    
    return {
        "results": results,
        "table_scores": table_scores
    }

# Main execution block
if __name__ == "__main__":
    print("TRIM-QA Pruning Script - Optimized Version")
    print("=" * 50)
    
    # Get user input for configuration
    num_queries, num_tables, threshold = get_user_input()
    
    # Setup supercomputer configuration
    config = setup_supercomputer_config()
    device = config['device']
    BATCH_SIZE = config['batch_size']
    
    print(f"\nSystem Configuration:")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Number of GPUs: {config['n_gpus']}")
    
    print("\nOptimization Settings:")
    print(f"Batch Processing: {USE_BATCHED_PROCESSING}")
    print(f"Caching Enabled: {ENABLE_CACHING}")
    print(f"Early Stopping: {USE_EARLY_STOPPING}")
    print(f"Full Reports: {GENERATE_FULL_REPORTS}")
    print(f"Model Quantization: {USE_MODEL_QUANTIZATION}")
    print(f"Max Chunks Per Table: {MAX_CHUNKS_PER_TABLE}")
    
    # Initialize models with optimizations
    print("\nInitializing models...")
    hidden_dim = 768
    model_name = "bert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = EmbeddingModule(AutoModel.from_pretrained(model_name))
    combined_model = CombinedModule(hidden_dim)
    
    # Apply optimizations to models
    embedding_model = optimize_models_for_inference(embedding_model, device)
    combined_model = optimize_models_for_inference(combined_model, device)
    
    if USE_MODEL_QUANTIZATION:
        embedding_model = quantize_model(embedding_model)
        combined_model = quantize_model(combined_model)
    
    # Process specific query numbers
    if num_queries is not None:
        if num_queries <= 0:
            print("Number of queries must be positive")
            sys.exit(1)
            
        # Generate query numbers list [1, 2, ..., num_queries]
        query_numbers = list(range(1, num_queries + 1))
        print(f"Processing specific query numbers: {query_numbers}")
        
        # Process these query numbers
        processing_results = process_query_numbers(
            query_numbers=query_numbers,
            tables_per_query=num_tables if num_tables else None,
            embedding_model=embedding_model,
            combined_model=combined_model,
            tokenizer=tokenizer,
            device=device,
            threshold=threshold
        )
        
        # Extract results
        results = processing_results["results"]
        table_scores = processing_results["table_scores"]
        
        # Create a table scores CSV
        if table_scores:
            # Convert table scores dictionary to DataFrame
            scores_df = pd.DataFrame([
                {
                    'table_id': table_id,
                    'query': info['query'],
                    'avg_score': info['avg_score'],
                    'is_target': info['is_target'],
                    'target_table': info['target_table'],
                    'target_answer': info['target_answer']
                }
                for table_id, info in table_scores.items()
            ])
            
            # Sort by average score
            scores_df = scores_df.sort_values(by='avg_score', ascending=False)
            
            # Save to CSV
            table_scores_path = os.path.join("pruning_results", f"table_scores_summary.csv")
            scores_df.to_csv(table_scores_path, index=False)
            print(f"\nTable scores summary saved to {table_scores_path}")
            
    else:
        # Process all queries (using the existing logic)
        print("\nProcessing all queries...")
        results = []
        
        # Load chunks once at the start
        print("Loading chunks from chunks.json...")
        all_chunks = {}
        with jsonlines.open('chunks.json', 'r') as reader:
            for chunk in reader:
                if 'metadata' in chunk and 'table_name' in chunk['metadata']:
                    table_id = chunk['metadata']['table_name']
                    if chunk['metadata'].get('chunk_type') != 'table':  # Skip table chunks
                        if table_id not in all_chunks:
                            all_chunks[table_id] = []
                        all_chunks[table_id].append(chunk)
        
        print(f"Loaded chunks for {len(all_chunks)} tables")
        
        # Process all queries
        query_files = sorted([f for f in os.listdir("Top-150-Quries") 
                             if f.startswith("query") and f.endswith("_TopTables.csv")])
        
        # Process each query
        for query_file in query_files:
            print(f"\nProcessing {query_file}")
            query, tables_info = process_query_file(os.path.join("Top-150-Quries", query_file))
            
            if num_tables:
                tables_info = tables_info[:num_tables]
            
            # Process tables
            # ...existing code for processing tables...
    
    # Generate final summary
    if results:
        print("\n===== Final Summary =====")
        print(f"Processed {len(results)} table-query pairs")
        
        # Fix the summary calculation to handle both count and list for pruned_chunks
        total_original = sum(r.get('original_chunks', 0) for r in results)
        
        # Use pruned_chunks_count if available, otherwise count the pruned_chunks list
        total_pruned = sum(r.get('pruned_chunks_count', len(r.get('pruned_chunks', []))) for r in results)
        
        # Calculate average reduction correctly
        avg_reduction = sum(r.get('reduction_percentage', 0) for r in results) / len(results)
        
        print(f"Total original chunks: {total_original}")
        print(f"Total pruned chunks: {total_pruned}")
        print(f"Average reduction: {avg_reduction:.1f}%")
        
        # Save summary with correct handling of pruned chunks
        summary_df = pd.DataFrame([{
            'query': r.get('question', ''),
            'table_id': r.get('table_id', ''),
            'target_table': r.get('target_table', ''),
            'target_answer': r.get('target_answer', ''),
            'original_chunks': r.get('original_chunks', 0),
            'pruned_chunks': r.get('pruned_chunks_count', len(r.get('pruned_chunks', []))),
            'reduction_percentage': r.get('reduction_percentage', 0),
            'early_stopped': r.get('early_stopped', False)
        } for r in results])
        
        summary_path = os.path.join("pruning_results", "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
    else:
        print("No results generated. Please check the errors above.")


