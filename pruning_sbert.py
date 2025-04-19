################cell 1########################
import sys
import json
import numpy as np
import pandas as pd
import argparse  # Added for command line arguments

import time
import os
from typing import List, Dict

# Try to import optional dependencies with fallbacks
try:
    import jsonlines
    JSONLINES_AVAILABLE = True
except ImportError:
    print("Warning: jsonlines not installed. Using built-in json for reading files.")
    JSONLINES_AVAILABLE = False

# Try to import torch and transformers dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch or transformers not available. Running in basic mode.")
    TORCH_AVAILABLE = False
    
# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Using basic similarity scoring.")
    SBERT_AVAILABLE = False

# Check for TAPAS
try:
    from transformers import TapasTokenizer, TapasForQuestionAnswering
    
    # Also check for torch_scatter which is required by TAPAS
    try:
        import torch_scatter
        TAPAS_AVAILABLE = True
    except ImportError:
        print("Warning: torch_scatter not available. TAPAS functionality will be disabled.")
        print("To use TAPAS, please install torch_scatter: https://github.com/rusty1s/pytorch_scatter")
        TAPAS_AVAILABLE = False
except ImportError:
    print("Warning: TAPAS models not available. Continuing without TAPAS functionality.")
    TAPAS_AVAILABLE = False

# Check for FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS is available for fast vector similarity search")
except ImportError:
    print("Warning: FAISS not available. Using slower similarity search method.")
    print("To use FAISS, install it with: pip install faiss-cpu or faiss-gpu")
    FAISS_AVAILABLE = False

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

###############cell 2########################
# Load models only if dependencies are available
models_loaded = False
column_model = None
row_model = None
tokenizer = None
tapas_model = None
tapas_tokenizer = None

# Load SBERT model if available
if TORCH_AVAILABLE and SBERT_AVAILABLE:
    try:
        print("Loading sentence-transformer model...")
        column_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast model
        
        print("Loading BERT model...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        row_model = BertModel.from_pretrained("bert-base-uncased")
        if row_model:
            row_model.eval()
        
        models_loaded = True
        print("Sentence-transformer models loaded successfully")
    except Exception as e:
        print(f"Error loading SBERT models: {e}")
        models_loaded = False

# Load TAPAS model if available
if TORCH_AVAILABLE and TAPAS_AVAILABLE:
    try:
        print("Loading TAPAS model...")
        model_name = "google/tapas-base-finetuned-wtq"  # Smaller model for faster loading
        tapas_tokenizer = TapasTokenizer.from_pretrained(model_name)
        tapas_model = TapasForQuestionAnswering.from_pretrained(model_name)
        print("TAPAS model loaded successfully")
    except Exception as e:
        print(f"Error loading TAPAS model: {e}")
        tapas_model = None
        tapas_tokenizer = None

###############cell 5########################
def read_jsonlines(filename):
    """Read jsonlines file with fallback to built-in json"""
    data = []
    if JSONLINES_AVAILABLE:
        try:
            with jsonlines.open(filename, 'r') as reader:
                for item in reader:
                    data.append(item)
            return data
        except Exception as e:
            print(f"Error reading with jsonlines: {e}, falling back to manual reading")
    
    # Manual reading as fallback
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []

def fetch_chunks(table_id, chunk_indexer=None):
    """
    Read chunks from chunks.json that match the target table.
    Uses FAISS indexing when available for faster retrieval.
    Returns both row and column chunks.
    
    Args:
        table_id: ID of the target table
        chunk_indexer: Optional ChunkIndexer instance for faster retrieval
    """
    print("Loading chunks for target table...")
    row_chunks = []
    column_chunks = []
    chunk_count = 0
    matched_chunks = 0
    
    # Try to use the indexer first if available
    if chunk_indexer is not None and FAISS_AVAILABLE:
        # Try to load existing index
        index_loaded = chunk_indexer.load_index(table_id)
        
        if index_loaded:
            print(f"Loaded existing FAISS index for table ID '{table_id}'")
            # Retrieve all chunk IDs from the index
            row_ids = chunk_indexer.chunk_map[table_id].get('row_ids', [])
            col_ids = chunk_indexer.chunk_map[table_id].get('col_ids', [])
            
            print(f"Found {len(row_ids)} row chunks and {len(col_ids)} column chunks in index")
            
            # We still need to fetch the actual chunks
            chunks_data = read_jsonlines('chunks.json')
            chunk_id_map = {chunk['metadata'].get('chunk_id', ''): chunk for chunk in chunks_data 
                          if 'metadata' in chunk and 'table_name' in chunk['metadata'] 
                          and chunk['metadata']['table_name'] == table_id}
            
            # Get row chunks
            for row_id in row_ids:
                if row_id in chunk_id_map:
                    chunk = chunk_id_map[row_id]
                    chunk['metadata']['chunk_type'] = 'row'
                    row_chunks.append(chunk)
                    
            # Get column chunks
            for col_id in col_ids:
                if col_id in chunk_id_map:
                    chunk = chunk_id_map[col_id]
                    chunk['metadata']['chunk_type'] = 'column'
                    column_chunks.append(chunk)
                    
            # If we found all chunks, return them
            if len(row_chunks) == len(row_ids) and len(column_chunks) == len(col_ids):
                print(f"Retrieved {len(row_chunks)} row chunks and {len(column_chunks)} column chunks from index")
                return row_chunks, column_chunks
            else:
                print(f"Warning: Could not find all chunks in chunks.json. Found {len(row_chunks)}/{len(row_ids)} row chunks and {len(column_chunks)}/{len(col_ids)} column chunks.")
                # Continue with normal fetch, but keep the chunks we found
    
    # Normal fetch if indexer not available or not all chunks found
    chunks_data = read_jsonlines('chunks.json')
    
    for chunk in chunks_data:
        chunk_count += 1
        # Check if the chunk belongs to our target table
        if 'metadata' in chunk and 'table_name' in chunk['metadata'] and chunk['metadata']['table_name'] == table_id:
            matched_chunks += 1
            
            # Determine chunk type
            chunk_type = chunk['metadata'].get('chunk_type', '')
            chunk_id = chunk['metadata'].get('chunk_id', '')
            
            # Identify chunk type by explicit chunk_type field
            if chunk_type == 'row':
                row_chunks.append(chunk)
            elif chunk_type == 'column' or chunk_type == 'col':
                column_chunks.append(chunk)
            # Fallback: identify by chunk_id pattern
            elif 'row' in chunk_id.lower():
                chunk['metadata']['chunk_type'] = 'row'
                row_chunks.append(chunk)
            elif 'col' in chunk_id.lower():
                chunk['metadata']['chunk_type'] = 'column'
                column_chunks.append(chunk)
            # Further fallback: check text format (column chunks typically have "header: value1 | value2")
            elif ':' in str(chunk.get('text', '')) and '|' in str(chunk.get('text', '')):
                # Text starts with a column name followed by a list of values
                text_parts = chunk.get('text', '').split(':', 1)
                if len(text_parts) == 2 and '|' in text_parts[1]:
                    chunk['metadata']['chunk_type'] = 'column'
                    column_chunks.append(chunk)
                else:
                    # Default to row if can't determine
                    chunk['metadata']['chunk_type'] = 'row'
                    row_chunks.append(chunk)
            else:
                # Default to row chunk if can't determine
                chunk['metadata']['chunk_type'] = 'row'
                row_chunks.append(chunk)

    print(f"Found {matched_chunks} chunks that match table ID '{table_id}'")
    print(f"- {len(row_chunks)} row chunks")
    print(f"- {len(column_chunks)} column chunks")

    # If we're using an indexer and the index wasn't loaded, build it now
    if chunk_indexer is not None and FAISS_AVAILABLE and not chunk_indexer.load_index(table_id):
        print("Building FAISS index for faster future retrieval...")
        chunk_indexer.build_index_for_table(table_id, row_chunks, column_chunks)

    return row_chunks, column_chunks

###############cell 6########################
class ChunkIndexer:
    """
    Manages FAISS index for efficient chunk retrieval and similarity search.
    Creates and stores embeddings for all chunks to avoid recomputing them.
    Uses a global index for all chunks rather than per-table indexes.
    """
    def __init__(self, embedding_model=None, device='cpu', index_dir='faiss_indexes', chunk_file='chunks.json'):
        self.embedding_model = embedding_model
        self.device = device
        self.index_dir = index_dir
        self.chunk_file = chunk_file
        self.global_index = None  # Single index for all chunks
        self.row_index = None  # Index for all row chunks
        self.col_index = None  # Index for all column chunks
        self.chunk_map = {}  # Maps table_id -> {chunk_ids: [ids]} for filtering
        self.id_to_chunk = {}  # Maps chunk_id -> chunk for retrieval
        self.id_to_embedding = {}  # Maps chunk_id -> embedding for reuse
        
        # Create index directory if it doesn't exist
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
    
    def encode_chunks(self, chunks):
        """Encode chunks using the embedding model"""
        if not self.embedding_model or not chunks:
            return None
            
        # Extract chunk texts
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        
        # Use the model to encode texts in batches for efficiency
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        
        # Convert to numpy for FAISS
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
        elif not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        return embeddings
    
    def build_global_index(self):
        """
        Build global FAISS indexes for all chunks in chunks.json.
        Creates separate indexes for row and column chunks.
        """
        if not FAISS_AVAILABLE or not self.embedding_model:
            return False
            
        # Check if index already exists
        if os.path.exists(os.path.join(self.index_dir, "global_row_index.faiss")) and \
           os.path.exists(os.path.join(self.index_dir, "global_col_index.faiss")):
            self.load_global_index()
            print(f"Loaded existing global index with {len(self.id_to_chunk)} chunks")
            return True
            
        print(f"Building global index from {self.chunk_file}...")
        start_time = time.time()
        
        # Load all chunks
        all_chunks = read_jsonlines(self.chunk_file)
        if not all_chunks:
            print("No chunks found in chunks.json")
            return False
            
        row_chunks = []
        col_chunks = []
        row_ids = []
        col_ids = []
        table_chunks = {}  # Dict to organize chunks by table_id
        
        # Organize chunks by type and table
        for chunk in all_chunks:
            if 'metadata' not in chunk or 'table_name' not in chunk['metadata']:
                continue
                
            # Get chunk details
            table_id = chunk['metadata']['table_name']
            chunk_id = chunk['metadata'].get('chunk_id', '')
            chunk_type = chunk['metadata'].get('chunk_type', '')
            
            # Skip chunks without proper identifiers
            if not chunk_id:
                continue
                
            # Determine chunk type
            if chunk_type == 'row' or ('row' in chunk_id.lower() and 'column' not in chunk_id.lower()):
                row_chunks.append(chunk)
                row_ids.append(chunk_id)
                chunk['metadata']['chunk_type'] = 'row'  # Ensure type is set
            elif chunk_type == 'column' or chunk_type == 'col' or 'column' in chunk_id.lower() or 'col' in chunk_id.lower():
                col_chunks.append(chunk)
                col_ids.append(chunk_id)
                chunk['metadata']['chunk_type'] = 'column'  # Ensure type is set
            else:
                # Try to detect by content
                text = str(chunk.get('text', ''))
                if ':' in text and '|' in text and len(text.split(':', 1)[0].strip()) < 30:
                    # Likely a column chunk (header: value | value |...)
                    col_chunks.append(chunk)
                    col_ids.append(chunk_id)
                    chunk['metadata']['chunk_type'] = 'column'
                else:
                    # Default to row
                    row_chunks.append(chunk)
                    row_ids.append(chunk_id)
                    chunk['metadata']['chunk_type'] = 'row'
            
            # Store the chunk in the id_to_chunk map
            self.id_to_chunk[chunk_id] = chunk
            
            # Organize by table
            if table_id not in table_chunks:
                table_chunks[table_id] = {'row_ids': [], 'col_ids': []}
                
            if chunk['metadata']['chunk_type'] == 'row':
                table_chunks[table_id]['row_ids'].append(chunk_id)
            else:
                table_chunks[table_id]['col_ids'].append(chunk_id)
        
        # Store the table -> chunk mappings
        self.chunk_map = table_chunks
        
        # Create embeddings for row chunks
        print(f"Creating embeddings for {len(row_chunks)} row chunks...")
        row_embeddings = self.encode_chunks(row_chunks)
        
        # Create embeddings for column chunks
        print(f"Creating embeddings for {len(col_chunks)} column chunks...")
        col_embeddings = self.encode_chunks(col_chunks)
        
        # Build row index if we have row embeddings
        if row_embeddings is not None and len(row_embeddings) > 0:
            dimension = row_embeddings.shape[1]
            self.row_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(row_embeddings)
            
            # Add to index
            self.row_index.add(row_embeddings)
            
            # Save embeddings for each chunk ID
            for i, chunk_id in enumerate(row_ids):
                self.id_to_embedding[chunk_id] = row_embeddings[i]
            
            # Save index to disk
            row_index_path = os.path.join(self.index_dir, "global_row_index.faiss")
            faiss.write_index(self.row_index, row_index_path)
            
            # Save row IDs mapping
            row_ids_path = os.path.join(self.index_dir, "global_row_ids.json")
            with open(row_ids_path, 'w') as f:
                json.dump(row_ids, f)
        
        # Build column index if we have column embeddings
        if col_embeddings is not None and len(col_embeddings) > 0:
            dimension = col_embeddings.shape[1]
            self.col_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(col_embeddings)
            
            # Add to index
            self.col_index.add(col_embeddings)
            
            # Save embeddings for each chunk ID
            for i, chunk_id in enumerate(col_ids):
                self.id_to_embedding[chunk_id] = col_embeddings[i]
            
            # Save index to disk
            col_index_path = os.path.join(self.index_dir, "global_col_index.faiss")
            faiss.write_index(self.col_index, col_index_path)
            
            # Save column IDs mapping
            col_ids_path = os.path.join(self.index_dir, "global_col_ids.json")
            with open(col_ids_path, 'w') as f:
                json.dump(col_ids, f)
        
        # Save table chunk mappings
        table_map_path = os.path.join(self.index_dir, "table_chunk_map.json")
        with open(table_map_path, 'w') as f:
            json.dump(self.chunk_map, f)
            
        # Save id_to_chunk mapping (limited to metadata to save space)
        id_to_metadata = {chunk_id: chunk.get('metadata', {}) for chunk_id, chunk in self.id_to_chunk.items()}
        id_to_chunk_path = os.path.join(self.index_dir, "id_to_metadata.json")
        with open(id_to_chunk_path, 'w') as f:
            json.dump(id_to_metadata, f)
        
        end_time = time.time()
        print(f"Global index built in {end_time - start_time:.2f} seconds")
        print(f"Indexed {len(row_chunks)} row chunks and {len(col_chunks)} column chunks")
        print(f"Covering {len(table_chunks)} tables")
        
        return True
    
    def load_global_index(self):
        """Load global FAISS indexes from disk if available"""
        if not FAISS_AVAILABLE:
            return False
            
        # Check if files exist
        row_index_path = os.path.join(self.index_dir, "global_row_index.faiss")
        row_ids_path = os.path.join(self.index_dir, "global_row_ids.json")
        col_index_path = os.path.join(self.index_dir, "global_col_index.faiss")
        col_ids_path = os.path.join(self.index_dir, "global_col_ids.json")
        table_map_path = os.path.join(self.index_dir, "table_chunk_map.json")
        id_to_chunk_path = os.path.join(self.index_dir, "id_to_metadata.json")
        
        # Load row index if available
        try:
            if os.path.exists(row_index_path) and os.path.exists(row_ids_path):
                self.row_index = faiss.read_index(row_index_path)
                with open(row_ids_path, 'r') as f:
                    self.row_ids = json.load(f)
                print(f"Loaded row index with {len(self.row_ids)} vectors")
            else:
                print("Row index not found")
                self.row_index = None
                self.row_ids = []
                
            # Load column index if available
            if os.path.exists(col_index_path) and os.path.exists(col_ids_path):
                self.col_index = faiss.read_index(col_index_path)
                with open(col_ids_path, 'r') as f:
                    self.col_ids = json.load(f)
                print(f"Loaded column index with {len(self.col_ids)} vectors")
            else:
                print("Column index not found")
                self.col_index = None
                self.col_ids = []
                
            # Load table chunk mapping
            if os.path.exists(table_map_path):
                with open(table_map_path, 'r') as f:
                    self.chunk_map = json.load(f)
                print(f"Loaded chunk mappings for {len(self.chunk_map)} tables")
                
            # Load id_to_chunk mapping (metadata only)
            if os.path.exists(id_to_chunk_path):
                with open(id_to_chunk_path, 'r') as f:
                    id_to_metadata = json.load(f)
                    # Create skeleton chunks with metadata
                    for chunk_id, metadata in id_to_metadata.items():
                        self.id_to_chunk[chunk_id] = {'metadata': metadata}
                        
            # If we have at least one index, consider it a success
            return self.row_index is not None or self.col_index is not None
            
        except Exception as e:
            print(f"Error loading global index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, table_id, query, k=None, chunk_type='row'):
        """
        Search for similar chunks for a specific table using the global index.
        Filters results to only include chunks from the specified table.
        
        Args:
            table_id: ID of the table to search
            query: Query text
            k: Number of results (if None, return all relevant chunks)
            chunk_type: 'row' or 'col' to specify which index to search
            
        Returns:
            List of dictionaries with chunk_id and score
        """
        if not FAISS_AVAILABLE or not self.embedding_model:
            return []
        
        # Load global index if not already loaded
        if (chunk_type == 'row' and self.row_index is None) or (chunk_type == 'col' and self.col_index is None):
            if not self.load_global_index():
                if not self.build_global_index():
                    return []
        
        # Check if we have chunk mappings for this table
        if table_id not in self.chunk_map:
            print(f"No chunk mapping for table {table_id}")
            return []
        
        # Get relevant chunk IDs for this table and chunk type
        relevant_ids = self.chunk_map[table_id].get(f'{chunk_type}_ids', [])
        if not relevant_ids:
            print(f"No {chunk_type} chunks for table {table_id}")
            return []
        
        # Select appropriate index and global ID list
        if chunk_type == 'row':
            index = self.row_index
            all_ids = self.row_ids if hasattr(self, 'row_ids') else []
        else:  # column
            index = self.col_index
            all_ids = self.col_ids if hasattr(self, 'col_ids') else []
        
        if not index:
            print(f"No index available for {chunk_type} chunks")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu().numpy()
        elif not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the global index
        # Use a larger k to ensure we find all relevant chunks for the specific table
        search_k = min(index.ntotal, max(100, len(relevant_ids) * 2))
        distances, indices = index.search(query_embedding, search_k)
        
        # Map indices to chunk IDs and filter for the target table
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(all_ids):
                continue
                
            chunk_id = all_ids[idx]
            
            # Check if this chunk belongs to our target table
            if chunk_id in relevant_ids:
                # Convert distance to similarity score (0-1 range)
                # FAISS gives inner product, higher is better
                score = float(max(0.0, min(1.0, (distance + 1) / 2)))
                results.append({
                    "chunk_id": chunk_id,
                    "score": score
                })
        
        # Limit to k results if specified
        if k is not None and k < len(results):
            results = results[:k]
            
        return results
            
    def get_chunks_for_table(self, table_id):
        """
        Retrieve all chunks for a specific table.
        Uses the preloaded chunks if available, otherwise loads from chunks.json.
        
        Args:
            table_id: ID of the table
            
        Returns:
            Tuple of (row_chunks, column_chunks)
        """
        # Try to use the preloaded mappings
        if table_id in self.chunk_map:
            row_ids = self.chunk_map[table_id].get('row_ids', [])
            col_ids = self.chunk_map[table_id].get('col_ids', [])
            
            # If we have complete chunks in memory (not just metadata)
            if all(id in self.id_to_chunk and 'text' in self.id_to_chunk[id] for id in row_ids + col_ids):
                row_chunks = [self.id_to_chunk[id] for id in row_ids]
                col_chunks = [self.id_to_chunk[id] for id in col_ids]
                return row_chunks, col_chunks
        
        # Fall back to loading from file
        print(f"Loading chunks for {table_id} from chunks.json file...")
        return fetch_chunks_from_file(table_id)
    
    def load_index(self, table_id):
        """
        Compatibility method for the old per-table index approach.
        Loads the global index and checks if it contains data for the specified table.
        
        Args:
            table_id: ID of the table
            
        Returns:
            True if the table data is available in the global index
        """
        # If global index is not loaded yet, load it
        if not hasattr(self, 'chunk_map') or not self.chunk_map:
            if not self.load_global_index():
                return False
        
        # Check if the table exists in our mappings
        if table_id in self.chunk_map:
            return True
            
        return False

    def build_index_for_table(self, table_id, row_chunks, column_chunks):
        """
        Compatibility method for the old per-table indexing approach.
        Adds the given chunks to the global index.
        
        Args:
            table_id: ID of the table
            row_chunks: List of row chunks
            column_chunks: List of column chunks
            
        Returns:
            True if the chunks were successfully added to the index
        """
        # If global index hasn't been built yet, build it
        if not hasattr(self, 'chunk_map') or not self.chunk_map:
            self.build_global_index()
            return table_id in self.chunk_map
            
        # If this table is already in the index, we're good
        if table_id in self.chunk_map:
            return True
            
        # We need to add this table's chunks to our global index
        print(f"Adding table {table_id} to global index...")
        
        # Track the chunks for this table
        table_entry = {'row_ids': [], 'col_ids': []}
        
        # Process row chunks
        if row_chunks:
            row_ids = []
            new_row_embeds = []
            
            for chunk in row_chunks:
                chunk_id = chunk['metadata'].get('chunk_id', '')
                if not chunk_id:
                    continue
                    
                # Add to our mappings
                row_ids.append(chunk_id)
                self.id_to_chunk[chunk_id] = chunk
                table_entry['row_ids'].append(chunk_id)
            
            # Only encode if we have chunks to add
            if row_ids:
                row_embeddings = self.encode_chunks(row_chunks)
                
                # Add to row index if we have one
                if self.row_index is not None and row_embeddings is not None:
                    # Normalize embeddings
                    faiss.normalize_L2(row_embeddings)
                    
                    # Add to index
                    self.row_index.add(row_embeddings)
                    
                    # Save embeddings for each chunk ID
                    for i, chunk_id in enumerate(row_ids):
                        self.id_to_embedding[chunk_id] = row_embeddings[i]
                        
                    # Update row_ids list
                    if hasattr(self, 'row_ids'):
                        self.row_ids.extend(row_ids)
        
        # Process column chunks
        if column_chunks:
            col_ids = []
            
            for chunk in column_chunks:
                chunk_id = chunk['metadata'].get('chunk_id', '')
                if not chunk_id:
                    continue
                    
                # Add to our mappings
                col_ids.append(chunk_id)
                self.id_to_chunk[chunk_id] = chunk
                table_entry['col_ids'].append(chunk_id)
            
            # Only encode if we have chunks to add
            if col_ids:
                col_embeddings = self.encode_chunks(column_chunks)
                
                # Add to column index if we have one
                if self.col_index is not None and col_embeddings is not None:
                    # Normalize embeddings
                    faiss.normalize_L2(col_embeddings)
                    
                    # Add to index
                    self.col_index.add(col_embeddings)
                    
                    # Save embeddings for each chunk ID
                    for i, chunk_id in enumerate(col_ids):
                        self.id_to_embedding[chunk_id] = col_embeddings[i]
                        
                    # Update col_ids list
                    if hasattr(self, 'col_ids'):
                        self.col_ids.extend(col_ids)
        
        # Add to the table mapping
        self.chunk_map[table_id] = table_entry
        
        # Save updates to disk
        try:
            self._save_index_updates()
            print(f"Successfully added table {table_id} to index")
            return True
        except Exception as e:
            print(f"Error saving index updates: {e}")
            return False
    
    def _save_index_updates(self):
        """Save updated indexes and mappings to disk"""
        # Create index directory if it doesn't exist
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            
        # Save row index if it exists
        if hasattr(self, 'row_index') and self.row_index is not None:
            row_index_path = os.path.join(self.index_dir, "global_row_index.faiss")
            faiss.write_index(self.row_index, row_index_path)
            
        # Save row IDs mapping
        if hasattr(self, 'row_ids'):
            row_ids_path = os.path.join(self.index_dir, "global_row_ids.json")
            with open(row_ids_path, 'w') as f:
                json.dump(self.row_ids, f)
        
        # Save column index if it exists
        if hasattr(self, 'col_index') and self.col_index is not None:
            col_index_path = os.path.join(self.index_dir, "global_col_index.faiss")
            faiss.write_index(self.col_index, col_index_path)
            
        # Save column IDs mapping
        if hasattr(self, 'col_ids'):
            col_ids_path = os.path.join(self.index_dir, "global_col_ids.json")
            with open(col_ids_path, 'w') as f:
                json.dump(self.col_ids, f)
        
        # Save table chunk mappings
        table_map_path = os.path.join(self.index_dir, "table_chunk_map.json")
        with open(table_map_path, 'w') as f:
            json.dump(self.chunk_map, f)
            
        # Save id_to_chunk mapping (limited to metadata to save space)
        id_to_metadata = {chunk_id: chunk.get('metadata', {}) for chunk_id, chunk in self.id_to_chunk.items()}
        id_to_chunk_path = os.path.join(self.index_dir, "id_to_metadata.json")
        with open(id_to_chunk_path, 'w') as f:
            json.dump(id_to_metadata, f)

###############cell 8########################
def score_chunks_with_simple_similarity(chunks: List[Dict], query: str) -> List[Dict]:
    """
    Scores chunks using simple term matching when advanced models aren't available.
    Works for both row and column chunks.
    
    Args:
        chunks: List of chunk dictionaries
        query: The query string
        
    Returns:
        List of dictionaries with chunk_id and score
    """
    query_words = query.lower().split()
    
    scored_chunks = []
    for chunk in chunks:
        chunk_text = str(chunk.get("text", "")).lower()
        chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
        
        # Simple word overlap metric
        matches = sum(1 for word in query_words if word in chunk_text)
        # Normalize by query length
        score = matches / max(1, len(query_words))
        
        # Boost score for exact phrase matches
        if query.lower() in chunk_text:
            score += 0.2  # Add bonus for containing exact phrase
            
        # Ensure score is between 0 and 1
        score = min(1.0, score)
        
        scored_chunks.append({
            "chunk_id": chunk_id,
            "score": score
        })
    
    return scored_chunks

def score_chunks_with_sbert(chunks: List[Dict], query: str) -> List[Dict]:
    """
    Scores chunks using sentence-transformers model.
    Works for both row and column chunks.
    
    Args:
        chunks: List of chunk dictionaries
        query: The query string
        
    Returns:
        List of dictionaries with chunk_id and score
    """
    global column_model
    
    if not SBERT_AVAILABLE or not models_loaded or column_model is None:
        return score_chunks_with_simple_similarity(chunks, query)
        
    # Encode the query once
    query_embedding = column_model.encode(query, convert_to_tensor=True)
    
    scored_chunks = []
    # Process chunks in batches to improve speed
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Get text from each chunk
        texts = [str(chunk.get("text", "")) for chunk in batch]
        chunk_ids = [chunk.get("metadata", {}).get("chunk_id", "") for chunk in batch]
        
        # Encode all texts in the batch
        chunk_embeddings = column_model.encode(texts, convert_to_tensor=True)
        
        # Compute cosine similarities for the batch
        cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
        
        # Process scores
        for idx, chunk_id in enumerate(chunk_ids):
            # Convert from (-1, 1) to (0, 1) range
            score = (cosine_scores[0][idx].item() + 1) / 2
            scored_chunks.append({
                "chunk_id": chunk_id,
                "score": score
            })
    
    return scored_chunks

def normalize_logits(logits):
    """Normalize the logits using Z-score standardization."""
    # Calculate mean and std for the logits
    mean = logits.mean().item()
    std = logits.std().item()
    
    if std > 0:  # Avoid division by zero
        normalized_logits = (logits - mean) / std
    else:
        normalized_logits = logits  # If std is 0, don't normalize

    return normalized_logits

def get_tapas_score_for_table(headers, rows, query):
    """
    Get TAPAS score for a single table with headers and rows.
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of cell values)
        query: The query string
        
    Returns:
        Float score or None if TAPAS isn't available
    """
    global tapas_model, tapas_tokenizer
    
    if not TAPAS_AVAILABLE or tapas_model is None or tapas_tokenizer is None:
        return None
    
    try:
        import torch_scatter
        
        # Get device that the model is on
        device = next(tapas_model.parameters()).device
        
        # Create DataFrame from headers and rows
        df = pd.DataFrame(rows, columns=headers)
        
        # Skip empty dataframes
        if df.empty:
            print("Empty table data, skipping TAPAS scoring")
            return None
        
        # Tokenize
        inputs = tapas_tokenizer(table=df, queries=[query], return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = tapas_model(**inputs)
        logits = outputs.logits
        
        # Extract table token logits only
        segment_ids = inputs["token_type_ids"][:, :, 0]
        cell_mask = (segment_ids[0] == 1) & (inputs["attention_mask"][0] == 1)
        valid_logits = logits[0][cell_mask]
        
        # Normalize and score
        if valid_logits.numel() > 0:
            normalized_logits = normalize_logits(valid_logits)
            mean = normalized_logits.mean().item()
            std = normalized_logits.std().item()
            alpha = 1.0  # Penalty for high variance
            score = mean - alpha * std
            return score
        
    except Exception as e:
        print(f"Error getting TAPAS score: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def score_table_with_tapas(row_chunks, column_chunks, query):
    """
    Score table using TAPAS model based on the row and column chunks.
    
    Args:
        row_chunks: List of row chunks
        column_chunks: List of column chunks
        query: The query string
        
    Returns:
        Tuple of (row_scores, column_scores) with TAPAS scores
    """
    if not TAPAS_AVAILABLE or tapas_model is None:
        return None, None
    
    print("Using TAPAS for table scoring...")
    
    # Function to extract headers and rows from chunks
    def extract_from_row_chunks(chunks):
        rows = []
        all_columns = set()
        
        # Get all column names first
        for chunk in chunks:
            col_names = chunk.get("metadata", {}).get("columns", [])
            all_columns.update(col_names)
        
        headers = sorted(list(all_columns))
        
        # Extract row data
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            row_data = dict.fromkeys(headers, "")  # Initialize with empty strings
            
            # Parse text into key-value pairs
            for part in text.split("|"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in headers:
                        row_data[key] = value
            
            rows.append([row_data.get(h, "") for h in headers])
        
        return headers, rows
    
    # Function to extract headers and rows from column chunks
    def extract_from_column_chunks(chunks):
        data = {}
        
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            
            if ":" in text:
                header, values_str = text.split(":", 1)
                header = header.strip()
                
                if header:
                    values = [v.strip() for v in values_str.strip().split("|") if v.strip()]
                    data[header] = values
        
        if not data:
            return [], []
        
        # Normalize lengths
        max_len = max((len(vals) for vals in data.values()), default=0)
        for header, values in data.items():
            if len(values) < max_len:
                values.extend([""] * (max_len - len(values)))
        
        headers = list(data.keys())
        rows = [[data[h][i] for h in headers] for i in range(max_len)]
        
        return headers, rows
    
    # Score rows
    row_score = None
    if row_chunks:
        headers, rows = extract_from_row_chunks(row_chunks)
        if headers and rows:
            row_score = get_tapas_score_for_table(headers, rows, query)
    
    # Score columns
    col_score = None
    if column_chunks:
        headers, rows = extract_from_column_chunks(column_chunks)
        if headers and rows:
            col_score = get_tapas_score_for_table(headers, rows, query)
    
    return row_score, col_score

def score_chunks(chunks: List[Dict], query: str, use_tapas=False) -> List[Dict]:
    """
    Scores chunks using the best available method.
    
    Args:
        chunks: List of chunk dictionaries
        query: The query string
        use_tapas: Whether to use TAPAS for scoring (if available)
        
    Returns:
        List of dictionaries with chunk_id and score
    """
    if not chunks:
        return []
    
    # For TAPAS, we handle scoring differently, but it's applied later in prune_table
    # This function still returns SBERT or simple scores per chunk
    if SBERT_AVAILABLE and models_loaded and column_model is not None:
        try:
            print("Using sentence-transformers for scoring")
            return score_chunks_with_sbert(chunks, query)
        except Exception as e:
            print(f"Error using SBERT for scoring: {e}")
            print("Falling back to simple similarity")
            return score_chunks_with_simple_similarity(chunks, query)
    else:
        print("Using simple word matching for scoring")
        return score_chunks_with_simple_similarity(chunks, query)

###############cell 9########################
def filter_chunks_by_score(chunks, scored_chunks, threshold):
    """
    Filters chunks based on precomputed semantic similarity scores.

    Args:
        chunks (list): List of all chunk dictionaries.
        scored_chunks (list of dict): List of {'chunk_id': str, 'score': float} dicts.
        threshold (float): Minimum score required to retain a chunk.

    Returns:
        filtered_chunks (list): Chunks whose score >= threshold.
    """
    # Build a set of valid chunk_ids to retain
    valid_ids = {chunk['chunk_id'] for chunk in scored_chunks if chunk['score'] >= threshold}

    # Filter chunks based on matching chunk_id
    filtered_chunks = [
        chunk for chunk in chunks
        if chunk['metadata'].get('chunk_id', '') in valid_ids
    ]
    
    # Add scores to filtered chunks for reference
    for chunk in filtered_chunks:
        chunk_id = chunk['metadata'].get('chunk_id', '')
        for scored_chunk in scored_chunks:
            if scored_chunk['chunk_id'] == chunk_id:
                chunk['score'] = scored_chunk['score']
                break
    
    return filtered_chunks

###############cell 11########################
def dynamic_threshold(scores, alpha=0.0):
    """
    Calculate a dynamic threshold using median + alpha * std deviation.
    Default alpha=0 means use the median as threshold.

    Args:
        scores (list or np.array): List of similarity scores (floats between 0 and 1).
        alpha (float): Multiplier for standard deviation (default 0.0).

    Returns:
        float: Threshold value
    """
    if not scores:
        return 0.5  # default threshold when no scores available
        
    median = np.median(scores)
    std_dev = np.std(scores)
    return median + alpha * std_dev

###############cell 12########################
def column_chunks_to_dataframe(column_chunks):
    """
    Converts a list of column chunks into a pandas DataFrame.

    Args:
        column_chunks (list): List of chunks, each containing a column in 'text' and metadata.

    Returns:
        pd.DataFrame: Structured DataFrame with headers and rows.
    """
    if not column_chunks:
        return pd.DataFrame()
        
    data = {}

    for chunk in column_chunks:
        text = str(chunk.get("text", ""))

        # Only process if the format is correct
        if ":" in text:
            header, values_str = text.split(":", 1)
            header = header.strip()

            if not header:
                continue

            # Split values and remove empty strings
            values = [v.strip() for v in values_str.strip().split("|") if v.strip()]
            data[header] = values

    # Normalize column lengths by padding with empty strings
    max_len = max((len(vals) for vals in data.values()), default=0)
    for header, values in data.items():
        if len(values) < max_len:
            values.extend([""] * (max_len - len(values)))

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='columns')

    return df

###############cell 13########################
def row_chunks_to_dataframe(row_chunks):
    """
    Converts row-based chunks with inline 'Header: Value' format into a structured DataFrame.

    Args:
        row_chunks (list): List of row-type chunk dicts.

    Returns:
        pd.DataFrame: Structured table with one row per chunk and appropriate columns.
    """
    if not row_chunks:
        return pd.DataFrame()
        
    rows = []
    all_columns = set()

    for chunk in row_chunks:
        text = str(chunk.get("text", ""))
        column_names = chunk.get("metadata", {}).get("columns", [])
        row_data = {}
        
        # Initialize all columns to empty string
        if column_names:
            row_data = dict.fromkeys(column_names, "")
        
        # Try to extract key-value pairs from the text
        for part in text.split("|"):
            if ":" in part:
                parts = part.split(":", 1)
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()
                    value = value.strip()
                    
                    # Add to row data and track column names
                    row_data[key] = value
                    all_columns.add(key)

        rows.append(row_data)

    # Ensure all rows have all columns
    for row in rows:
        for col in all_columns:
            if col not in row:
                row[col] = ""

    # Build DataFrame with all columns
    df = pd.DataFrame(rows, columns=sorted(all_columns))
    return df

###############cell 14########################
def combine_dataframes(row_df, col_df):
    """
    Intelligently combine row and column dataframes.
    If they have overlapping columns, use those; otherwise keep all data.

    Args:
        row_df: DataFrame from row chunks
        col_df: DataFrame from column chunks

    Returns:
        Combined DataFrame
    """
    # If either is empty, return the other
    if row_df.empty:
        return col_df
    if col_df.empty:
        return row_df
        
    # Check for overlapping columns
    common_cols = set(row_df.columns).intersection(set(col_df.columns))
    
    if common_cols:
        print(f"Found {len(common_cols)} common columns between row and column data")
        # Use common columns to merge dataframes
        try:
            # Try to merge based on common values
            result = pd.merge(row_df, col_df, on=list(common_cols), how='outer')
            
            # If the result is empty, fall back to using row data
            if result.empty:
                print("Merge resulted in empty DataFrame, using row data")
                return row_df
            return result
        except Exception as e:
            print(f"Error merging dataframes: {e}")
            print("Using row data")
            return row_df
    else:
        # No common columns, return row data with column data appended as new columns
        print("No common columns found, combining all columns")
        return row_df

###############cell 15########################
def dataframe_to_json_entry(df, table_id):
    """
    Convert a pandas DataFrame into a JSON-serializable dict matching the required format.

    Args:
        df (pd.DataFrame): The pruned table.
        table_id (str): Unique table identifier.

    Returns:
        dict: A JSON-compatible dictionary.
    """
    if df.empty:
        return None
        
    json_entry = {
        "id": table_id,
        "table": {
            "columns": [{"text": str(col)} for col in df.columns],
            "rows": [{"cells": [{"text": str(cell)} for cell in row]} for _, row in df.iterrows()],
            "tableId": table_id,
        }
    }

    return json_entry

###############cell 16########################
def prune_table(table_id, question, use_tapas=True, chunk_indexer=None):
    """
    Prune a table by keeping only relevant row and column chunks.
    Both row and column chunks are scored and filtered based on relevance to the question.
    
    Args:
        table_id (str): The table ID to prune
        question (str): The query question
        use_tapas (bool): Whether to use TAPAS for scoring (if available)
        chunk_indexer (ChunkIndexer): Optional indexer for fast similarity search
        
    Returns:
        pd.DataFrame: The pruned table
    """
    # Fetch both row and column chunks for the given table ID
    row_chunks, column_chunks = fetch_chunks(table_id, chunk_indexer)
    
    if not row_chunks and not column_chunks:
        print(f"No chunks found for table {table_id}")
        return pd.DataFrame()
    
    # If TAPAS is available and requested, use it for supplementary scoring
    row_tapas_score = None
    col_tapas_score = None
    
    if use_tapas and TAPAS_AVAILABLE and tapas_model is not None:
        print("Using TAPAS to evaluate table relevance...")
        row_tapas_score, col_tapas_score = score_table_with_tapas(row_chunks, column_chunks, question)
        if row_tapas_score is not None:
            print(f"TAPAS row score: {row_tapas_score:.4f}")
        if col_tapas_score is not None:
            print(f"TAPAS column score: {col_tapas_score:.4f}")
    
    # Process row chunks
    row_df = pd.DataFrame()
    if row_chunks:
        # Score row chunks (using FAISS if available)
        print(f"Scoring {len(row_chunks)} row chunks...")
        if chunk_indexer is not None and FAISS_AVAILABLE:
            # Use FAISS for faster scoring
            row_scores = chunk_indexer.search(table_id, question, chunk_type='row')
            if not row_scores:  # If FAISS search failed, fall back to standard scoring
                row_scores = score_chunks(row_chunks, question)
        else:
            row_scores = score_chunks(row_chunks, question)
        
        if row_scores:
            # Calculate threshold
            row_scores_only = [chunk["score"] for chunk in row_scores]
            row_threshold = dynamic_threshold(row_scores_only, alpha=0.0)  # using median as threshold
            print(f"Row threshold: {row_threshold:.4f}")
            
            # Filter row chunks
            filtered_row_chunks = filter_chunks_by_score(row_chunks, row_scores, row_threshold)
            print(f"Kept {len(filtered_row_chunks)}/{len(row_chunks)} row chunks ({len(filtered_row_chunks)/len(row_chunks)*100:.1f}%)")
            
            # Convert to DataFrame
            row_df = row_chunks_to_dataframe(filtered_row_chunks)
    
    # Process column chunks
    col_df = pd.DataFrame()
    if column_chunks:
        # Score column chunks (using FAISS if available)
        print(f"Scoring {len(column_chunks)} column chunks...")
        if chunk_indexer is not None and FAISS_AVAILABLE:
            # Use FAISS for faster scoring
            col_scores = chunk_indexer.search(table_id, question, chunk_type='col')
            if not col_scores:  # If FAISS search failed, fall back to standard scoring
                col_scores = score_chunks(column_chunks, question)
        else:
            col_scores = score_chunks(column_chunks, question)
        
        if col_scores:
            # Calculate threshold
            col_scores_only = [chunk["score"] for chunk in col_scores]
            col_threshold = dynamic_threshold(col_scores_only, alpha=0.0)  # using median as threshold
            print(f"Column threshold: {col_threshold:.4f}")
            
            # Filter column chunks
            filtered_col_chunks = filter_chunks_by_score(column_chunks, col_scores, col_threshold)
            print(f"Kept {len(filtered_col_chunks)}/{len(column_chunks)} column chunks ({len(filtered_col_chunks)/len(column_chunks)*100:.1f}%)")
            
            # Convert to DataFrame
            col_df = column_chunks_to_dataframe(filtered_col_chunks)
    
    # Combine row and column DataFrames
    print("Combining row and column data...")
    final_df = combine_dataframes(row_df, col_df)
    
    print('-' * 80)
    print("Final pruned table:")
    print(final_df.head(5) if len(final_df) > 5 else final_df)
    if len(final_df) > 5:
        print(f"... and {len(final_df) - 5} more rows")
    
    return final_df

###############cell 17########################
def extract_query_files():
    """Find query files in the current directory"""
    # Look for individual query files first
    query_files = [f for f in os.listdir() if f.startswith("query") and f.endswith("_TopTables.csv")]
    
    # If zip file exists but no query files, try to extract it
    if not query_files and os.path.exists("Top-150-Quries.zip"):
        try:
            import zipfile
            with zipfile.ZipFile("Top-150-Quries.zip", 'r') as zip_ref:
                zip_ref.extractall("Top-150-Quries")
            print("Extracted query files from Top-150-Quries.zip")
            # Look in the extracted directory
            if os.path.exists("Top-150-Quries"):
                query_files = [os.path.join("Top-150-Quries", f) 
                              for f in os.listdir("Top-150-Quries") 
                              if f.startswith("query") and f.endswith("_TopTables.csv")]
        except Exception as e:
            print(f"Failed to extract zip file: {e}")
    
    return query_files

# Main execution block
if __name__ == "__main__":
    print("TRIM-QA Row and Column Pruning Script with TAPAS Support")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Table pruning script with SBERT and TAPAS")
    parser.add_argument("--query-start", type=int, default=1, help="Starting query index (1-based)")
    parser.add_argument("--query-end", type=int, default=None, help="Ending query index (inclusive)")
    parser.add_argument("--max-tables", type=int, default=None, help="Maximum number of tables to process per query")
    parser.add_argument("--specific-table", type=int, default=None, help="Process only this specific table index (1-based)")
    parser.add_argument("--no-tapas", action="store_true", help="Disable TAPAS scoring")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-faiss", action="store_true", help="Disable FAISS indexing for chunk retrieval")
    parser.add_argument("--index-dir", type=str, default="faiss_indexes", help="Directory to store FAISS indexes")
    parser.add_argument("--build-index-only", action="store_true", help="Only build the FAISS global index and exit")
    parser.add_argument("--force-rebuild-index", action="store_true", help="Force rebuilding the FAISS index even if it exists")
    parser.add_argument("--query", type=str, help="Run with a specific query text instead of using query files")
    parser.add_argument("--table-id", type=str, help="Run on a specific table ID (requires --query)")
    parser.add_argument("--query-index", type=int, help="Run a specific query by its index (1-based)")
    parser.add_argument("--tables-for-query", type=str, help="Specify the number of tables for specific queries, format: 'query_index:num_tables,...' (e.g., '2:10,5:20')")
    args = parser.parse_args()
    
    use_tapas = not args.no_tapas
    use_faiss = not args.no_faiss and FAISS_AVAILABLE
    
    if use_tapas and not (TAPAS_AVAILABLE and tapas_model is not None):
        print("TAPAS requested but not available. Using only SBERT/basic scoring.")
        use_tapas = False
    
    # Setup device (GPU/CPU)
    device = None
    if TORCH_AVAILABLE:
        if args.cpu:
            device = torch.device("cpu")
            print("Forcing CPU usage as requested")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            # Move models to the selected device
            if device.type == "cuda":
                if row_model is not None:
                    row_model.to(device)
                if tapas_model is not None:
                    tapas_model.to(device)
                print("Models loaded to GPU successfully")
    
    # Initialize ChunkIndexer if FAISS is available and not disabled
    chunk_indexer = None
    if use_faiss and SBERT_AVAILABLE and column_model is not None:
        print(f"Initializing FAISS indexer in {args.index_dir} directory")
        chunk_indexer = ChunkIndexer(
            embedding_model=column_model,
            device=device if device else 'cpu',
            index_dir=args.index_dir
        )
        print("FAISS indexer initialized - embeddings will be cached for faster retrieval")
        
        # Always try to load the global index first
        print("Checking for existing global FAISS index...")
        index_loaded = chunk_indexer.load_global_index()
        
        # Force rebuild if requested or build if not found
        if args.force_rebuild_index or not index_loaded:
            if args.force_rebuild_index:
                print("Forcing rebuild of global index as requested")
                should_build_index = True
            else:
                print("Global index not found.")
                
                # Ask user whether to build the index or continue without it
                if not args.build_index_only:  # Only ask if not already in build-index-only mode
                    user_choice = input("Building a global FAISS index will significantly speed up queries but may take time initially.\n"
                                        "Do you want to build the index now? (y/n): ").strip().lower()
                    should_build_index = user_choice == 'y' or user_choice == 'yes'
                    
                    if not should_build_index:
                        print("Proceeding without FAISS indexing.")
                        use_faiss = False
                        chunk_indexer = None
                else:
                    should_build_index = True
                    
            # Build the index if requested
            if should_build_index:
                # This will take time on first run but saves time later
                print("Building global FAISS index for all chunks (this may take several minutes)...")
                chunk_indexer.build_global_index()
            
        # Exit if only building the index
        if args.build_index_only and index_loaded or (not index_loaded and should_build_index):
            print("Global index built successfully. Exiting as requested.")
            sys.exit(0)
    
    try:
        # If specific query text is provided, run with it directly
        if args.query:
            if not args.table_id:
                print("Error: --table-id is required when using --query")
                sys.exit(1)
            
            print(f"Running with specific query: {args.query}")
            print(f"Processing table ID: {args.table_id}")
            pruned_df = prune_table(args.table_id, args.query, use_tapas=use_tapas, chunk_indexer=chunk_indexer)
            
            if not pruned_df.empty:
                entry = dataframe_to_json_entry(pruned_df, args.table_id)
                output_file = "pruned_table.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                print(f"Saved pruned table to {output_file}")
            sys.exit(0)
        
        # Find query files
        query_files = extract_query_files()
        
        if query_files:
            print(f"Found {len(query_files)} query files")
            
            # Determine which query files to process based on range arguments
            start_idx = args.query_start - 1  # Convert to 0-based
            end_idx = args.query_end if args.query_end is not None else len(query_files)
            end_idx = min(end_idx, len(query_files))
            
            if start_idx < 0 or start_idx >= len(query_files):
                print(f"Invalid query start index: {args.query_start}, must be between 1 and {len(query_files)}")
                start_idx = 0
                
            query_files_to_process = query_files[start_idx:end_idx]
            print(f"Processing {len(query_files_to_process)} queries from index {args.query_start} to {min(args.query_end if args.query_end else len(query_files), len(query_files))}")
            
            start_time = time.time()
            output_file = "pruned_tables_tapas.json" if use_tapas else "pruned_tables.json"
            
            with open(output_file, "w", encoding="utf-8") as f:
                # Process each query
                for query_idx, query_csv in enumerate(query_files_to_process):
                    print(f"\nProcessing query {start_idx + query_idx + 1}/{len(query_files)}: {query_csv}")
                    
                    try:
                        query_csv_df = pd.read_csv(query_csv)
                        question = query_csv_df['query'][0]
                        table_list = query_csv_df['top tables'].tolist()
                        target_table_id = query_csv_df['target table'][0] if 'target table' in query_csv_df.columns else None
                        goal_answer = query_csv_df['target answer'][0] if 'target answer' in query_csv_df.columns else None
                        
                        print(f"Query: {question}")
                        if target_table_id:
                            print(f"Target table: {target_table_id}")
                        if goal_answer:
                            print(f"Target answer: {goal_answer}")
                        
                        # Determine which tables to process
                        if args.specific_table is not None:
                            if 0 < args.specific_table <= len(table_list):
                                tables_to_process = [table_list[args.specific_table - 1]]  # Convert to 0-based
                                print(f"Processing only table {args.specific_table}: {tables_to_process[0]}")
                            else:
                                print(f"Invalid specific table index: {args.specific_table}, must be between 1 and {len(table_list)}")
                                tables_to_process = table_list[:args.max_tables] if args.max_tables else table_list
                        else:
                            tables_to_process = table_list[:args.max_tables] if args.max_tables else table_list
                            print(f"Processing {len(tables_to_process)} tables")
                        
                        # Process each table for this query
                        for i, table_id in enumerate(tables_to_process):
                            print(f"\nProcessing table {i+1}/{len(tables_to_process)}: {table_id}")
                            pruned_df = prune_table(table_id, question, use_tapas=use_tapas, chunk_indexer=chunk_indexer)
                            
                            if not pruned_df.empty:
                                entry = dataframe_to_json_entry(pruned_df, table_id)
                                if entry:
                                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                    print(f"Saved pruned table to {output_file}")
                    
                    except Exception as e:
                        print(f"Error processing query {query_csv}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue  # Continue with next query even if this one fails
            
            end_time = time.time()
            print(f"\nExecution time: {end_time - start_time:.2f} seconds")
        
        else:
            # No query files, use manual input
            print("No query files found. Please provide a query.")
            question = input("Enter query: ")
            table_id = input("Enter table ID to process: ")
            if table_id:
                pruned_df = prune_table(table_id, question, use_tapas=use_tapas, chunk_indexer=chunk_indexer)
                if not pruned_df.empty:
                    entry = dataframe_to_json_entry(pruned_df, table_id)
                    with open("pruned_tables_tapas.json" if use_tapas else "pruned_tables.json", "w", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
