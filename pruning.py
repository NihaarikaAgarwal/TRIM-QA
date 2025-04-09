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
        with torch.no_grad():
            embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1)
            chunk_embeddings.append(embeddings)
    
    print(f"Generated embeddings for {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
    
    # Get question embedding
    print("Computing embedding for question...")
    question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        question_embedding = embedding_model.model(**question_inputs).last_hidden_state.mean(dim=1)
    
    chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
    
    print("Computing relevance scores...")
    # Get combined relevance scores using both question and chunk embeddings
    scores, _, _ = combined_model(chunk_embeddings)
    scores = scores.squeeze().tolist()
    
    # Scale scores based on similarity with question
    similarity_scores = torch.nn.functional.cosine_similarity(
        chunk_embeddings, 
        question_embedding.expand(chunk_embeddings.size(0), -1)
    ).tolist()
    
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
    
    # Save pruned chunks
    output_dir = "pruned_chunks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(pruned_chunks, f, indent=2)
    
    print(f"Original chunks: {len(chunks)}, Pruned chunks: {len(pruned_chunks)}")
    print(f"Saved pruned chunks to {output_path}")
    
    # Create and save DataFrames for visualization
    print("Converting chunks to DataFrames for visualization...")
    
    # Convert original chunks to DataFrame
    original_df = chunks_to_dataframe(chunks)
    
    # Add score columns to the original DataFrame
    original_df['raw_score'] = final_scores
    original_df['normalized_score'] = normalized_scores
    
    # Convert pruned chunks to DataFrame
    pruned_df = chunks_to_dataframe(pruned_chunks)
    
    # Save DataFrames to CSV
    csv_dir = "pruning_results"
    os.makedirs(csv_dir, exist_ok=True)
    
    csv_base = os.path.join(csv_dir, f"{table_id}_{safe_question}")
    original_csv = f"{csv_base}_original.csv"
    pruned_csv = f"{csv_base}_pruned.csv"
    
    original_df.to_csv(original_csv, index=False)
    pruned_df.to_csv(pruned_csv, index=False)
    
    print(f"Saved original chunks DataFrame to {original_csv}")
    print(f"Saved pruned chunks DataFrame to {pruned_csv}")
    
    # Generate a basic HTML report with pandas styling
    html_file = f"{csv_base}_report.html"
    
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
        'output_path': output_path,
        'html_report': html_file
    }

# Main execution block
if __name__ == "__main__":
    # Check for command line arguments to determine run mode
    run_mode = "interactive"
    target_id = None
    batch_process = False
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "batch":
            run_mode = "batch"
            batch_process = True
        else:
            run_mode = "specific"
            target_id = sys.argv[1]
    
    # Initialize models (only need to do this once)
    hidden_dim = 768  # BERT/RoBERTa hidden dimension
    model_name = "bert-base-uncased"  # or any other preferred model
    
    print("Loading tokenizer and models...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = EmbeddingModule(AutoModel.from_pretrained(model_name))
    combined_model = CombinedModule(hidden_dim)
    print(f"Models loaded in {time.time() - start_time:.2f} seconds")
    
    # Define thresholds for testing
    thresholds = [0.2, 0.3, 0.4, 0.5]
    final_threshold = 0.6  # Use this threshold for saving pruned chunks
    
    # Process based on run mode
    if run_mode == "batch" or run_mode == "specific":
        # Load all test queries from the pruningTestData directory
        test_queries = load_pruning_test_data()
        
        if not test_queries:
            print("No test queries found in the pruningTestData directory. Exiting.")
            exit(1)
            
        if run_mode == "specific" and target_id:
            # Filter queries by target ID
            test_queries = [q for q in test_queries if q['table_id'] == target_id]
            if not test_queries:
                print(f"No test queries found for table ID {target_id}. Exiting.")
                exit(1)
        
        # Process all queries in batch mode
        results = []
        total_queries = len(test_queries)
        
        print(f"Processing {total_queries} queries in {'batch' if batch_process else 'specific'} mode")
        print("-" * 80)
        
        for i, query_item in enumerate(test_queries):
            print(f"\n[{i+1}/{total_queries}] Processing query for table {query_item['table_id']}")
            
            # Load chunks for this table ID
            chunks = []
            chunk_count = 0
            matched_chunks = 0
            
            with jsonlines.open('chunks.json', 'r') as reader:
                for chunk in reader:
                    chunk_count += 1
                    if chunk_count % 10000 == 0:
                        print(f"  Processed {chunk_count} chunks, matched {matched_chunks} so far...")
                    
                    # Check if the chunk belongs to our target table
                    if 'metadata' in chunk and 'table_name' in chunk['metadata'] and chunk['metadata']['table_name'] == query_item['table_id']:
                        chunks.append(chunk)
                        matched_chunks += 1
            
            if not chunks:
                print(f"Warning: No chunks found for table ID {query_item['table_id']}. Skipping this query.")
                continue
                
            print(f"Found {matched_chunks} chunks for table ID {query_item['table_id']}")
            
            # Process this query
            result = process_query(
                query_item, 
                chunks,
                embedding_model, 
                combined_model, 
                tokenizer, 
                thresholds, 
                final_threshold
            )
            
            results.append(result)
            print("-" * 80)
        
        # Generate summary report
        print("\n===== SUMMARY REPORT =====")
        print(f"Processed {len(results)}/{total_queries} queries")
        
        avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results) if results else 0
        print(f"Average chunk reduction: {avg_reduction:.1f}%")
        
        # Save summary to CSV
        summary_df = pd.DataFrame([{
            'question': r['question'],
            'table_id': r['table_id'],
            'original_chunks': r['original_chunks'],
            'pruned_chunks': r['pruned_chunks'],
            'reduction_percentage': r['reduction_percentage'],
            'output_path': r['output_path']
        } for r in results])
        
        summary_path = "pruning_summary_report.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to {summary_path}")
        
    else:
        # Interactive mode - original behavior
        # Extract table IDs from test.jsonl
        test_table_info, test_table_ids = extract_table_ids_from_test_jsonl()
        
        if test_table_ids:
            print("Available tables from test.jsonl:")
            for i, table_info in enumerate(test_table_info):
                print(f"{i+1}. {table_info['id']} - {table_info['title']}")
                # Show questions if available
                if table_info['questions']:
                    print(f"   Questions: {', '.join(table_info['questions'][:2])}")
                    if len(table_info['questions']) > 2:
                        print(f"   ...and {len(table_info['questions'])-2} more questions")
            
            # Prompt user to select a table ID
            print("\nEnter the table ID you want to use (or press Enter to use the first one):")
            user_input = input("> ").strip()
            
            if user_input:
                TARGET_TABLE_ID = user_input
            else:
                TARGET_TABLE_ID = test_table_ids[0]
        else:
            # Fall back to tables.jsonl if no tables found in test.jsonl
            print("No tables found in test.jsonl. Showing available tables from tables.jsonl:")
            tables = list_available_tables()
            for i, table in enumerate(tables):
                print(f"{i+1}. {table['id']} - {table['title']}")
            
            # Prompt user to select a table ID
            print("\nEnter the table ID you want to use (or press Enter to use the first one):")
            user_input = input("> ").strip()
            
            if user_input:
                TARGET_TABLE_ID = user_input
            else:
                TARGET_TABLE_ID = tables[0]['id']
        
        print(f"\nStarting pruning process for table: {TARGET_TABLE_ID}")
        print("-" * 80)
        
        # Read target table from tables.jsonl
        print("Loading target table...")
        target_table = None
        with jsonlines.open('tables.jsonl', 'r') as reader:
            for table in reader:
                if table['tableId'] == TARGET_TABLE_ID:
                    target_table = table
                    break
        
        if not target_table:
            print(f"Error: Could not find table with ID {TARGET_TABLE_ID}")
            exit(1)
        
        print(f"Found target table: {target_table.get('documentTitle', 'Unknown title')}")
        
        # Get all table IDs (primary + alternatives) for the target table
        target_table_ids = [TARGET_TABLE_ID]
        if 'alternativeTableIds' in target_table and isinstance(target_table['alternativeTableIds'], list):
            target_table_ids.extend(target_table['alternativeTableIds'])
        
        print(f"Looking for chunks with table IDs: {target_table_ids}")
        
        # Read chunks from chunks.json that match the target table
        print("Loading and filtering chunks for target table...")
        chunks = []
        chunk_count = 0
        matched_chunks = 0
        
        with jsonlines.open('chunks.json', 'r') as reader:
            for chunk in reader:
                chunk_count += 1
                if chunk_count % 10000 == 0:
                    print(f"  Processed {chunk_count} chunks, matched {matched_chunks} so far...")
                
                # Check if the chunk belongs to our target table or any of its alternative IDs
                if 'metadata' in chunk and 'table_name' in chunk['metadata'] and chunk['metadata']['table_name'] in target_table_ids:
                    chunks.append(chunk)
                    matched_chunks += 1
        
        print(f"Found {matched_chunks} chunks that match table ID '{TARGET_TABLE_ID}' out of {chunk_count} total chunks")
        
        if not chunks:
            print("Error: No matching chunks found for the target table")
            exit(1)
        
        # Read questions for the target table from test.jsonl
        print("Loading test questions for target table...")
        test_items = []
        with jsonlines.open('test.jsonl', 'r') as reader:
            for item in reader:
                # Check both tableId and alternativeTableIds fields
                item_table_ids = [item.get('table_id', ''), item.get('tableId', '')]
                
                # Add alternativeTableIds if available
                if 'alternativeTableIds' in item and isinstance(item['alternativeTableIds'], list):
                    item_table_ids.extend(item['alternativeTableIds'])
                # Check tableId from the table object if available
                if 'table' in item and 'tableId' in item['table']:
                    item_table_ids.append(item['table']['tableId'])
                # Check alternativeTableIds from the table object if available
                if 'table' in item and 'alternativeTableIds' in item['table'] and isinstance(item['table']['alternativeTableIds'], list):
                    item_table_ids.extend(item['table']['alternativeTableIds'])
                
                # Check if any of the tableIds match our target
                if TARGET_TABLE_ID in item_table_ids and 'questions' in item:
                    for question in item['questions']:
                        test_items.append({
                            'question': question['originalText'],
                            'table_id': TARGET_TABLE_ID,
                            'answer': question['answer']['answerTexts']
                        })
        
        print(f"Found {len(test_items)} test questions for target table")
        
        if not test_items:
            # Create a sample question for demonstration if no test questions exist
            test_items = [{
                'question': f"What information is available about {target_table.get('documentTitle', 'this table')}?",
                'table_id': TARGET_TABLE_ID,
                'answer': ["Sample answer"]
            }]
            print(f"No test questions found. Created a sample question for demonstration.")
        
        # Process each question
        for idx, item in enumerate(test_items):
            print(f"\n[{idx+1}/{len(test_items)}] Processing question: {item['question']}")
            result = process_query(
                item, 
                chunks, 
                embedding_model, 
                combined_model, 
                tokenizer, 
                thresholds,
                final_threshold
            )
            print("-" * 80)


