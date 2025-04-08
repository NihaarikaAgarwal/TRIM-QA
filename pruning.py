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
        # Fully-connected layers for mean and sigma computation
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)
    
    def forward(self, h):
        """
        h: Tensor of shape (batch_size, hidden_dim) representing token embeddings.
        Returns:
            eta_uns: Tensor of shape (batch_size, 1) representing relevance scores.
        """
        # Compute mean and standard deviation
        mu = self.fc_mu(h)   # shape: (batch_size, 1)
        sigma = F.softplus(self.fc_sigma(h))  # ensures sigma > 0
        
        # Sample s from normal distribution with mean mu and std sigma
        s = torch.normal(mean=mu, std=sigma)  # this tweak is ours different from the CABINET
        
        # Reparameterization: z = mu + s * sigma --- latent variable
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
        # Fully-connected layer for weak supervision score computation
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, h):
        """
        h: Tensor of shape (batch_size, hidden_dim) representing token embeddings.
        Returns:
            eta_ws: Tensor of shape (batch_size, 1) representing weak supervision scores.
        """
        # Compute weak supervision score
        eta_ws = torch.sigmoid(self.fc(h))  # shape: (batch_size, 1)
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
    
def prune_chunks(chunks, scores, threshold=0.6): # here this threshold can be tuned
    """
    chunks: list of chunk dicts.
    scores: list of combined relevance scores corresponding to each chunk.
    threshold: minimum score to keep a chunk.
    Returns a list of pruned chunks.
    """
    pruned = []
    for chunk, score in zip(chunks, scores):
        if score >= threshold:
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
def chunks_to_dataframe(chunks, is_pruned_chunks=False, normalized_scores=None):
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
        is_pruned = "Yes" if is_pruned_chunks else "No"
        
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

def process_query_file(query_file_path):
    """Extract query information from a query CSV file."""
    df = pd.read_csv(query_file_path)
    # Get the first row since query is same for all rows
    query = df['query'].iloc[0]
    tables_info = []
    for _, row in df.iterrows():
        tables_info.append({
            'table_id': row['top tables'],
            'target_table': row['target table'],
            'target_answer': row['target answer']
        })
    return query, tables_info

def merge_chunks_to_dataframe(pruned_chunks):
    """Merge row and column chunks into a single dataframe."""
    # Filter out table chunks and organize remaining chunks
    rows = []
    columns = []
    
    for chunk in pruned_chunks:
        if 'metadata' in chunk and 'chunk_type' in chunk['metadata']:
            chunk_type = chunk['metadata']['chunk_type']
            if chunk_type == 'row':
                rows.append({
                    'content': chunk['text'],
                    'type': 'row',
                    'row_id': chunk['metadata'].get('row_id', ''),
                    'col_id': chunk['metadata'].get('col_id', ''),
                    'score': chunk.get('score', 0.0)  # Add score if available
                })
            elif chunk_type == 'column':
                columns.append({
                    'content': chunk['text'],
                    'type': 'column',
                    'row_id': chunk['metadata'].get('row_id', ''),
                    'col_id': chunk['metadata'].get('col_id', ''),
                    'score': chunk.get('score', 0.0)  # Add score if available
                })
    
    # Create dataframes and sort by row_id/col_id for better organization
    row_df = pd.DataFrame(rows)
    col_df = pd.DataFrame(columns)
    
    # Combine and sort
    combined_df = pd.concat([row_df, col_df], ignore_index=True)
    combined_df = combined_df.sort_values(['type', 'row_id', 'col_id'])
    
    return combined_df

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_query_number(filename):
    """Extract query number from filename."""
    return int(''.join(filter(str.isdigit, filename)))

def get_user_input():
    """Get user input for number of queries and tables to process."""
    try:
        num_queries = input("Enter number of queries to test (press Enter for all queries): ").strip()
        num_queries = int(num_queries) if num_queries else None
        
        num_tables = input("Enter number of tables to test per query (press Enter for all tables): ").strip()
        num_tables = int(num_tables) if num_tables else None
        
        return num_queries, num_tables
    except ValueError:
        print("Invalid input. Using all queries and tables.")
        return None, None

def save_pruning_results(query, table_id, original_chunks, pruned_chunks, pruning_dir="pruning_results"):
    """Save original chunks, pruned chunks and generate a report."""
    # Create results directory if it doesn't exist
    ensure_directory(pruning_dir)
    
    # Create base filename
    base_filename = f"{table_id}_{query[:30].replace(' ', '_')}"
    
    # Get the scores from pruned chunks to identify which chunks were kept
    pruned_chunk_ids = [chunk['metadata'].get('chunk_id') for chunk in pruned_chunks]
    
    # Save original chunks with pruning status
    original_df = chunks_to_dataframe(original_chunks, normalized_scores=[chunk.get('score', 0.0) for chunk in original_chunks])
    original_df['is_pruned'] = original_df['chunk_id'].apply(lambda x: "No" if x in pruned_chunk_ids else "Yes")
    original_path = os.path.join(pruning_dir, f"{base_filename}_original.csv")
    original_df.to_csv(original_path, index=False)
    
    # Save pruned chunks
    pruned_df = chunks_to_dataframe(pruned_chunks, is_pruned_chunks=True)
    pruned_path = os.path.join(pruning_dir, f"{base_filename}_pruned.csv")
    pruned_df.to_csv(pruned_path, index=False)
    
    # Create and save merged row/column dataframe
    merged_df = merge_chunks_to_dataframe(pruned_chunks)
    merged_path = os.path.join(pruning_dir, f"{base_filename}_merged.csv")
    merged_df.to_csv(merged_path, index=False)
    
    # Calculate raw score statistics
    raw_scores = [chunk.get('score', 0.0) for chunk in original_chunks]
    if raw_scores:
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        avg_score = sum(raw_scores) / len(raw_scores)
    else:
        min_score = max_score = avg_score = 0
    
    # Calculate normalized score statistics from the original chunks
    scores = original_df['score'].tolist()
    if scores:
        min_norm = min(scores)
        max_norm = max(scores)
        avg_norm = sum(scores) / len(scores)
    else:
        min_norm = max_norm = avg_norm = 0
    
    # Generate HTML report with enhanced styling
    html_content = f"""
    <html>
    <head>
        <title>Pruning Report for {query}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .stats {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
            .table-container {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr.kept {{ background-color: #d4edda; }}
            tr.pruned {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>Pruning Report</h1>
        <p>Question: {query}</p>
        <p>Expected answer(s): {table_info.get('target_answer', ['Unknown'])}</p>
        
        <div class="stats">
            <h2>Statistics</h2>
            <p>Total chunks: {len(original_chunks)}</p>
            <p>Pruned chunks: {len(original_chunks) - len(pruned_chunks)} ({(len(original_chunks) - len(pruned_chunks))/len(original_chunks)*100:.1f}%)</p>
            <p>Pruning threshold: {0.6}</p>
            <p>Raw score statistics - Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}</p>
            <p>Normalized score statistics - Min: {min_norm:.4f}, Max: {max_norm:.4f}, Avg: {avg_norm:.4f}</p>
        </div>
        
        <div class="table-container">
            <h2>Chunk Data with Scores</h2>
    """
    
    # Apply CSS classes based on pruning status
    styled_df = original_df.copy()
    styled_df['_class'] = styled_df['is_pruned'].apply(lambda x: 'pruned' if x == 'Yes' else 'kept')
    
    # Convert DataFrame to HTML with proper row classes
    df_html = styled_df.to_html(index=False, escape=False)
    
    # Add CSS classes to table rows based on pruning status
    df_html = df_html.replace('<tr>', '<tr class="kept">', styled_df['is_pruned'].value_counts()['No'])
    df_html = df_html.replace('<tr>', '<tr class="pruned">', styled_df['is_pruned'].value_counts()['Yes'])
    
    html_content += df_html
    html_content += """
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(pruning_dir, f"{base_filename}_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return {
        'original_path': f"{base_filename}_original.csv",
        'pruned_path': f"{base_filename}_pruned.csv",
        'merged_path': f"{base_filename}_merged.csv",
        'report_path': f"{base_filename}_report.html"
    }

# Main execution block
if __name__ == "__main__":
    # Get user input for query and table limits
    num_queries, num_tables = get_user_input()
    
    # Initialize models
    hidden_dim = 768  # BERT/RoBERTa hidden dimension
    model_name = "bert-base-uncased"
    
    print("Loading tokenizer and models...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = EmbeddingModule(AutoModel.from_pretrained(model_name))
    combined_model = CombinedModule(hidden_dim)
    print(f"Models loaded in {time.time() - start_time:.2f} seconds")
    
    # Create output directories
    pruned_base_dir = os.path.expanduser("~/pruned")
    html_base_dir = os.path.expanduser("~/html")
    ensure_directory(pruned_base_dir)
    ensure_directory(html_base_dir)
    
    # Get list of query files
    query_files = sorted(
        [f for f in os.listdir("Top-150-Quries") if f.startswith("query") and f.endswith("_TopTables.csv")],
        key=get_query_number
    )
    
    # Apply query limit if specified
    if num_queries is not None:
        query_files = query_files[:num_queries]
        print(f"Processing {len(query_files)} queries...")
    else:
        print("Processing all queries...")
    
    # Load all chunks once
    print("Loading chunks from chunks.json...")
    all_chunks = {}
    with jsonlines.open('chunks.json', 'r') as reader:
        for chunk in reader:
            if 'metadata' in chunk and 'table_name' in chunk['metadata']:
                table_id = chunk['metadata']['table_name']
                # Skip table-type chunks
                if chunk['metadata'].get('chunk_type') != 'table':
                    if table_id not in all_chunks:
                        all_chunks[table_id] = []
                    all_chunks[table_id].append(chunk)
    
    print(f"Loaded chunks for {len(all_chunks)} unique tables")
    
    # Process each query file
    for query_file in query_files:
        query_num = get_query_number(query_file)
        print(f"\nProcessing {query_file} (Query {query_num})")
        
        # Create query-specific directories
        query_pruned_dir = os.path.join(pruned_base_dir, f"query{query_num}")
        query_html_dir = os.path.join(html_base_dir, f"query{query_num}")
        ensure_directory(query_pruned_dir)
        ensure_directory(query_html_dir)
        
        # Process query file
        query, tables_info = process_query_file(os.path.join("Top-150-Quries", query_file))
        print(f"Query: {query}")
        
        # Apply table limit if specified
        if num_tables is not None:
            tables_info = tables_info[:num_tables]
            print(f"Processing {len(tables_info)} tables...")
        else:
            print(f"Processing all {len(tables_info)} tables...")
        
        # Get question embedding once for this query
        question_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            question_embedding = embedding_model.model(**question_inputs).last_hidden_state.mean(dim=1)
        
        # Process each table
        for table_info in tables_info:
            table_id = table_info['table_id']
            
            # Skip if no chunks found for this table
            if table_id not in all_chunks:
                print(f"No chunks found for table {table_id}, skipping...")
                continue
            
            chunks = all_chunks[table_id]
            print(f"\nProcessing table {table_id} ({len(chunks)} chunks)")
            
            # Get embeddings for chunks
            chunk_embeddings = []
            for chunk in chunks:
                # Combine chunk content into a single string
                if isinstance(chunk['text'], list):
                    chunk_text = " ".join([str(cell) for cell in chunk['text']])
                else:
                    chunk_text = str(chunk['text'])
                
                inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1)
                    chunk_embeddings.append(embeddings)
            
            if not chunk_embeddings:
                print(f"No valid chunks to process for table {table_id}, skipping...")
                continue
                
            chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
            
            # Get combined relevance scores
            scores, _, _ = combined_model(chunk_embeddings)
            scores = scores.squeeze().tolist()
            
            # Scale scores based on similarity with question
            similarity_scores = torch.nn.functional.cosine_similarity(
                chunk_embeddings,
                question_embedding.expand(chunk_embeddings.size(0), -1)
            ).tolist()
            
            # Combine scores and normalize
            final_scores = [(s + sim) / 2 for s, sim in zip(scores, similarity_scores)]
            min_score = min(final_scores)
            max_score = max(final_scores)
            if max_score > min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in final_scores]
            else:
                normalized_scores = final_scores
            
            # Prune chunks and create dataframe
            pruned_chunks = prune_chunks(chunks, normalized_scores, threshold=0.3)
            if pruned_chunks:
                # Add scores to pruned chunks for reference
                for chunk, score in zip(pruned_chunks, normalized_scores):
                    chunk['score'] = score
                
                # Save results
                result_files = save_pruning_results(
                    query=query,
                    table_id=table_id,
                    original_chunks=chunks,
                    pruned_chunks=pruned_chunks
                )
                print(f"\nResults saved:")
                print(f"Original chunks: {result_files['original_path']}")
                print(f"Pruned chunks: {result_files['pruned_path']}")
                print(f"Merged row/column view: {result_files['merged_path']}")
                print(f"HTML report: {result_files['report_path']}")
                
                print(f"Processed {table_id}: {len(chunks)} chunks -> {len(pruned_chunks)} pruned chunks")


