import json
import pandas as pd
import argparse
import os
import jsonlines
import time
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
    """
    A simple class that uses sentence transformers to prune chunks based on
    query similarity. This is separate from your custom URS and weak supervision approach.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts."""
        return self.model.encode(texts)
    
    def compute_similarity_scores(self, chunk_texts, query):
        """Compute similarity scores between chunks and query."""
        # Get embeddings
        chunk_embeddings = self.get_embeddings(chunk_texts)
        query_embedding = self.get_embeddings([query])[0]
        
        # Compute cosine similarity
        scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
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

def main():
    parser = argparse.ArgumentParser(description='Prune chunks using sentence transformers')
    parser.add_argument('input_file', help='Path to input JSON file with chunks')
    parser.add_argument('output_file', help='Path to output JSON file for pruned chunks')
    parser.add_argument('--query', required=True, help='Question query for pruning')
    parser.add_argument('--relevance-threshold', type=float, default=0.5, 
                        help='Minimum relevance score to keep a chunk (default: 0.5)')
    parser.add_argument('--similarity-threshold', type=float, default=0.8, 
                        help='Similarity threshold for redundancy removal (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=None, 
                        help='Maximum number of chunks to return (default: no limit)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load chunks
    print(f"Loading chunks from {args.input_file}...")
    chunks = load_chunks(args.input_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create pruner
    pruner = SentenceTransformerPruner(model_name=args.model)
    
    # Prune chunks
    print(f"Pruning chunks with query: '{args.query}'...")
    pruned_chunks_with_scores = pruner.prune_chunks(
        chunks, 
        args.query, 
        relevance_threshold=args.relevance_threshold,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k
    )
    
    pruned_chunks = [chunk for chunk, _ in pruned_chunks_with_scores]
    scores = [score for _, score in pruned_chunks_with_scores]
    
    print(f"Pruned to {len(pruned_chunks)} chunks")
    
    # Save pruned chunks
    print(f"Saving pruned chunks to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for chunk in pruned_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Save detailed results with scores
    detailed_output_file = args.output_file + ".detailed.json"
    with open(detailed_output_file, 'w') as f:
        results = []
        for (chunk, score) in pruned_chunks_with_scores:
            result = {
                "chunk": chunk,
                "relevance_score": float(score)
            }
            results.append(result)
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    if pruned_chunks:
        df_data = []
        for (chunk, score) in pruned_chunks_with_scores:
            df_data.append({
                'chunk_id': chunk['metadata'].get('chunk_id', 'unknown'),
                'chunk_type': chunk['metadata'].get('chunk_type', 'unknown'),
                'table_name': chunk['metadata'].get('table_name', 'unknown'),
                'score': score,
                'text': chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text']
            })
        
        df = pd.DataFrame(df_data)
        summary_file = args.output_file + ".summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
    
    elapsed_time = time.time() - start_time
    print(f"Done! Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()