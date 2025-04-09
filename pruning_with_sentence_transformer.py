import json
import pandas as pd
import argparse
import os
import jsonlines
import time
import torch  # Adding torch import at top level
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_chunks(file_path):
    """Load chunks from a JSON file."""
    chunks = []
    try:
        with open(file_path, 'r') as f:
            # Handle both JSON array and line-by-line JSON objects
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':  # JSON array
                chunks = json.load(f)
            else:  # Line-by-line JSON objects
                chunks = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading chunks: {e}")
        
    return chunks

def chunks_to_dataframe(chunks):
    """Convert chunks to a pandas DataFrame for better visualization."""
    data = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
        chunk_id = chunk['metadata'].get('chunk_id', f'chunk_{i}')
        
        # Extract additional metadata if available
        col_id = chunk['metadata'].get('col_id', '')
        row_id = chunk['metadata'].get('row_id', '')
        table_name = chunk['metadata'].get('table_name', '')
        
        data.append({
            'chunk_id': chunk_id,
            'chunk_type': chunk_type,
            'table_name': table_name,
            'col_id': col_id,
            'row_id': row_id,
            'text': chunk_text
        })
    
    return pd.DataFrame(data)

def extract_table_ids_from_test_jsonl(file_path='test.jsonl'):
    """Extract table IDs from test.jsonl file."""
    table_info = []
    table_ids = []
    
    try:
        with jsonlines.open(file_path, 'r') as reader:
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
        print(f"Error reading {file_path}: {e}")
    
    return table_info, table_ids

class SentenceTransformerPruner:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        self.batch_size = batch_size
        # Try to use GPU with specific configurations for A100
        try:
            import torch
            if torch.cuda.is_available():
                # Enable TF32 for better performance on A100
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Set device to CUDA
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                print(f"CUDA Version: {torch.version.cuda}")
                # Initialize model on GPU
                self.model = SentenceTransformer(model_name).to(self.device)
                # Enable parallel processing if multiple GPUs are available
                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs!")
                    self.model = torch.nn.DataParallel(self.model)
            else:
                print("No GPU available, falling back to CPU")
                self.device = torch.device("cpu")
                self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error initializing GPU: {e}")
            print("Falling back to CPU")
            self.device = torch.device("cpu")
            self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        """Get embeddings for a list of texts using batched processing."""
        embeddings = []
        # Process in batches for memory efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            with torch.no_grad():  # Disable gradient calculation for inference
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device
                )
                # Move embeddings to CPU if they were on GPU
                if self.device.type == "cuda":
                    batch_embeddings = batch_embeddings.cpu()
                embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if self.device.type == "cuda":
            return torch.cat(embeddings).numpy()
        return torch.cat(embeddings).numpy()

    def compute_similarity_scores(self, chunk_texts, query):
        """Compute similarity scores between chunks and query using GPU acceleration."""
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )
            if self.device.type == "cuda":
                query_embedding = query_embedding.cpu()
            query_embedding = query_embedding.numpy()

        # Get chunk embeddings in batches
        chunk_embeddings = self.get_embeddings(chunk_texts)
        
        # Compute cosine similarity
        scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        return scores

    def remove_redundant_chunks(self, chunks, scores, similarity_threshold=0.8):
        """Remove redundant chunks while preserving the most relevant ones."""
        # Sort chunks by score in descending order
        sorted_items = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        selected_indices = []
        
        # Get embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.get_embeddings(chunk_texts)
        
        # Compute pairwise similarity matrix
        sim_matrix = cosine_similarity(chunk_embeddings)
        
        # Track which chunks are covered/redundant
        covered = [False] * len(chunks)
        
        # Process chunks in order of relevance
        for i, (chunk, score) in enumerate(sorted_items):
            if not covered[i]:
                selected_chunks.append((chunk, score))
                selected_indices.append(i)
                
                # Mark similar chunks as covered
                for j in range(len(chunks)):
                    if not covered[j] and sim_matrix[i, j] > similarity_threshold:
                        covered[j] = True
        
        return selected_chunks
    
    def prune_chunks(self, chunks, query, relevance_threshold=0.5, similarity_threshold=0.8, top_k=None):
        """
        Prune chunks based on query similarity and redundancy.
        
        Args:
            chunks: List of chunk dictionaries
            query: Question query for relevance scoring
            relevance_threshold: Minimum relevance score to keep a chunk
            similarity_threshold: Threshold for redundancy removal
            top_k: Maximum number of chunks to return
            
        Returns:
            List of (chunk, score) tuples
        """
        # Extract text from chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Compute similarity scores
        scores = self.compute_similarity_scores(chunk_texts, query)
        
        # Filter by relevance threshold
        filtered_chunks = []
        filtered_scores = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            if score >= relevance_threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
        
        # Remove redundant chunks
        if filtered_chunks:
            selected_chunks = self.remove_redundant_chunks(filtered_chunks, filtered_scores, similarity_threshold)
        else:
            selected_chunks = []
        
        # Limit to top_k if specified
        if top_k is not None and selected_chunks:
            selected_chunks = selected_chunks[:top_k]
        
        return selected_chunks

def process_top_queries(query_range, top_n_tables, pruner, relevance_threshold=0.5, similarity_threshold=0.8):
    """Process queries from Top-150-Queries directory using sentence transformer"""
    try:
        # Create results directory structure
        base_dir = "pruning_results_st"  # Changed to differentiate from original pruning results
        for subdir in ['original', 'pruned', 'merged', 'reports']:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        
        # Load queries from the specified range
        queries_dir = "Top-150-Quries"
        
        # Get the specific query files based on query numbers
        selected_files = []
        for query_num in range(query_range[0], query_range[1] + 1):
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
                
                print(f"Found {len(chunks)} chunks for table {table_id}")
                
                # Prune chunks using sentence transformer
                pruned_chunks_with_scores = pruner.prune_chunks(
                    chunks,
                    query_text,
                    relevance_threshold=relevance_threshold,
                    similarity_threshold=similarity_threshold
                )
                
                if not pruned_chunks_with_scores:
                    print(f"No chunks remained after pruning for table {table_id}")
                    continue
                
                pruned_chunks = [chunk for chunk, _ in pruned_chunks_with_scores]
                scores = [score for _, score in pruned_chunks_with_scores]
                
                # Create safe filenames
                safe_query = "".join(c if c.isalnum() else "_" for c in query_text[:30])
                
                # Save original chunks
                original_file = os.path.join(base_dir, "original", f"{query_num}_{table_id}_original.json")
                with open(original_file, 'w') as f:
                    json.dump(chunks, f, indent=2)
                
                # Save pruned chunks
                pruned_file = os.path.join(base_dir, "pruned", f"{query_num}_{table_id}_pruned.json")
                with open(pruned_file, 'w') as f:
                    json.dump(pruned_chunks, f, indent=2)
                
                # Create merged version (combining similar chunks)
                merged_chunks = []
                chunk_groups = {}
                
                for chunk in pruned_chunks:
                    chunk_type = chunk['metadata'].get('chunk_type', '')
                    if chunk_type in ['row', 'col']:
                        key = f"{chunk_type}_{chunk['metadata'].get(f'{chunk_type}_id', '')}"
                        if key not in chunk_groups:
                            chunk_groups[key] = []
                        chunk_groups[key].append(chunk)
                    else:
                        merged_chunks.append(chunk)
                
                # Merge chunks in each group
                for group_chunks in chunk_groups.values():
                    if group_chunks:
                        merged_text = " ".join([c['text'] for c in group_chunks])
                        merged_score = np.mean([c.get('score', 0) for c in group_chunks])
                        merged_chunks.append({
                            'text': merged_text,
                            'score': merged_score,
                            'metadata': {
                                **group_chunks[0]['metadata'],
                                'merged_from': len(group_chunks)
                            }
                        })
                
                # Save merged chunks
                merged_file = os.path.join(base_dir, "merged", f"{query_num}_{table_id}_merged.json")
                with open(merged_file, 'w') as f:
                    json.dump(merged_chunks, f, indent=2)
                
                # Generate HTML report
                df_data = []
                for chunk, score in zip(pruned_chunks, scores):
                    df_data.append({
                        'chunk_id': chunk['metadata'].get('chunk_id', 'unknown'),
                        'chunk_type': chunk['metadata'].get('chunk_type', 'unknown'),
                        'text': chunk['text'],
                        'score': score
                    })
                
                df = pd.DataFrame(df_data)
                
                html_file = os.path.join(base_dir, "reports", f"{query_num}_{table_id}_report.html")
                
                html_content = f"""
                <html>
                <head>
                    <title>Sentence Transformer Pruning Report - Query {query_num}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        .stats {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Sentence Transformer Pruning Report</h1>
                    <div class="stats">
                        <h2>Query Information</h2>
                        <p><strong>Query:</strong> {query_text}</p>
                        <p><strong>Table ID:</strong> {table_id}</p>
                        <p><strong>Target Table:</strong> {target_table}</p>
                        <p><strong>Target Answer:</strong> {target_answer}</p>
                        <h2>Pruning Statistics</h2>
                        <p>Original chunks: {len(chunks)}</p>
                        <p>Pruned chunks: {len(pruned_chunks)}</p>
                        <p>Reduction: {((1 - len(pruned_chunks)/len(chunks)) * 100):.1f}%</p>
                    </div>
                    {df.to_html()}
                </body>
                </html>
                """
                
                with open(html_file, 'w') as f:
                    f.write(html_content)
                
                # Add to results
                results.append({
                    'query_num': query_num,
                    'question': query_text,
                    'table_id': table_id,
                    'target_table': target_table,
                    'original_chunks': len(chunks),
                    'pruned_chunks': len(pruned_chunks),
                    'reduction_percentage': (1 - len(pruned_chunks)/len(chunks)) * 100,
                    'avg_score': np.mean(scores)
                })
        
        # Save summary for all results
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(base_dir, f"summary_queries_{query_range[0]}-{query_range[1]}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to {summary_path}")
        
        return results
    
    except Exception as e:
        print(f"Error processing queries: {e}")
        return []

def main():
    print("\n=== TRIM-QA Sentence Transformer Pruning ===")
    print("=" * 45)
    
    # Get query range from user
    while True:
        try:
            print("\nEnter query range (1-150, format: start end):")
            start, end = map(int, input("> ").split())
            if 1 <= start <= end <= 150:
                break
            print("Invalid range. Start must be <= end, and both must be between 1 and 150.")
        except ValueError:
            print("Please enter two numbers separated by space (e.g., '1 5')")
    
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
    
    # Get relevance threshold
    while True:
        try:
            print("\nEnter relevance threshold (0.0-1.0, default: 0.5):")
            print("This determines how similar a chunk must be to the query to be kept.")
            threshold_input = input("> ").strip()
            if not threshold_input:  # Use default if empty
                relevance_threshold = 0.5
                break
            relevance_threshold = float(threshold_input)
            if 0 <= relevance_threshold <= 1:
                break
            print("Threshold must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number between 0 and 1.")
    
    # Get similarity threshold for redundancy removal
    while True:
        try:
            print("\nEnter similarity threshold for redundancy removal (0.0-1.0, default: 0.8):")
            print("Higher values mean chunks must be more similar to be considered redundant.")
            threshold_input = input("> ").strip()
            if not threshold_input:  # Use default if empty
                similarity_threshold = 0.8
                break
            similarity_threshold = float(threshold_input)
            if 0 <= similarity_threshold <= 1:
                break
            print("Threshold must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number between 0 and 1.")
    
    # Choose model
    print("\nSelect sentence transformer model:")
    print("1. all-MiniLM-L6-v2 (default, faster)")
    print("2. all-mpnet-base-v2 (more accurate but slower)")
    print("3. paraphrase-multilingual-MiniLM-L12-v2 (multilingual support)")
    while True:
        try:
            choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
            if not choice:  # Use default if empty
                model_name = 'all-MiniLM-L6-v2'
                break
            choice = int(choice)
            if choice == 1:
                model_name = 'all-MiniLM-L6-v2'
                break
            elif choice == 2:
                model_name = 'all-mpnet-base-v2'
                break
            elif choice == 3:
                model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                break
            print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number between 1 and 3.")
    
    # Show selected configuration
    print("\n=== Configuration Summary ===")
    print(f"Query Range: {start}-{end}")
    print(f"Top Tables per Query: {top_n}")
    print(f"Relevance Threshold: {relevance_threshold}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print(f"Model: {model_name}")
    
    # Confirm and proceed
    while True:
        confirm = input("\nProceed with pruning? (y/n): ").lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no']:
            print("Pruning cancelled.")
            return
        else:
            print("Please enter 'y' or 'n'.")
    
    start_time = time.time()
    
    # Create pruner instance
    print(f"\nInitializing sentence transformer with model: {model_name}")
    pruner = SentenceTransformerPruner(model_name=model_name)
    
    # Process queries
    results = process_top_queries(
        query_range=(start, end),
        top_n_tables=top_n,
        pruner=pruner,
        relevance_threshold=relevance_threshold,
        similarity_threshold=similarity_threshold
    )
    
    if results:
        print("\n===== Final Summary =====")
        print(f"Processed {len(results)} table-query pairs")
        avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
        avg_score = sum(r['avg_score'] for r in results) / len(results)
        print(f"Average chunk reduction: {avg_reduction:.1f}%")
        print(f"Average relevance score: {avg_score:.3f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()