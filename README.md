# Table Pruning with SBERT

This repository contains code for pruning tables using Sentence-BERT (SBERT) embeddings. The main script `pruning_sbert.py` allows you to prune tables based on semantic relevance to queries.

## Initial Setup

### Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, install the core dependencies directly:

```bash
pip install sentence-transformers faiss-cpu torch pandas numpy tqdm tabulate matplotlib colorama
```

For GPU support (recommended for large datasets):

```bash
pip install sentence-transformers faiss-gpu torch pandas numpy tqdm tabulate matplotlib colorama
```

### Installing PyTorch with CUDA Support

If you need specific PyTorch versions with CUDA support, use the provided `torch_update.txt` file:

```bash
pip install -r torch_update.txt
```

This will install PyTorch 1.10.2 with CUDA 11.3 support, which is optimal for this project.

### Directory Structure

Make sure you have the following directories:
- `faiss_indexes/` - For storing FAISS indexes
- `pruned_chunks/` - For storing pruned chunks output
- `pruning_results/` - For storing pruning evaluation results

You can create them if they don't exist:

```bash
mkdir -p faiss_indexes pruned_chunks pruning_results
```

## Running the Pruning Script

The `pruning_sbert.py` script provides multiple ways to run table pruning:

### Basic Usage

```bash
python pruning_sbert.py --tables tables.jsonl --query "your query here" --output pruned_chunks
```

### All Available Options

```bash
python pruning_sbert.py \
    --tables tables.jsonl \
    --query "your query here" \
    --output pruned_chunks \
    --model-name "all-MiniLM-L6-v2" \
    --top-k 5 \
    --min-score 0.5 \
    --pruning-method rows \
    --create-index \
    --overwrite
```

### Parameter Descriptions

- `--tables`: Path to the JSONL file containing tables (default: `tables.jsonl`)
- `--query`: The query to use for pruning tables
- `--output`: Directory to store pruned chunks (default: `pruned_chunks`)
- `--model-name`: Name of the Sentence-BERT model (default: `all-MiniLM-L6-v2`)
- `--top-k`: Number of top elements to keep (default: `5`)
- `--min-score`: Minimum similarity score threshold (default: `0.5`)
- `--pruning-method`: Method for pruning (`rows`, `columns`, `both`, or `cells`) (default: `rows`)
- `--create-index`: Flag to create new indexes even if they already exist
- `--overwrite`: Flag to overwrite existing pruned chunks

### Common Usage Patterns

#### 1. Pruning by Rows Only

```bash
python pruning_sbert.py --query "who is king charles" --pruning-method rows
```

#### 2. Pruning by Columns Only

```bash
python pruning_sbert.py --query "who is king charles" --pruning-method columns
```

#### 3. Pruning Both Rows and Columns

```bash
python pruning_sbert.py --query "who is king charles" --pruning-method both
```

#### 4. Cell-Level Pruning

```bash
python pruning_sbert.py --query "who is king charles" --pruning-method cells
```

#### 5. Adjusting Similarity Threshold

```bash
python pruning_sbert.py --query "who is king charles" --min-score 0.7
```

#### 6. Using a Different SBERT Model

```bash
python pruning_sbert.py --query "who is king charles" --model-name "all-mpnet-base-v2"
```

#### 7. Keeping More Top Results

```bash
python pruning_sbert.py --query "who is king charles" --top-k 10
```

#### 8. Force Recreating Indexes

```bash
python pruning_sbert.py --query "who is king charles" --create-index
```

## Visualizing Pruning Results

You can visualize the pruning results using the `visualize_pruned_tables.py` script:

```bash
python visualize_pruned_tables.py
```

Filtering by specific table:

```bash
python visualize_pruned_tables.py --table-name "Table_Name"
```

Filtering by query:

```bash
python visualize_pruned_tables.py --query "query_string"
```

## Examples

### Example 1: Basic Pruning

```bash
python pruning_sbert.py --query "who won the 2015 great british bake off"
```

### Example 2: Pruning with Higher Threshold

```bash
python pruning_sbert.py --query "gossip girl who is kristen bell" --min-score 0.7 --pruning-method both
```

### Example 3: Using Different Model with More Results

```bash
python pruning_sbert.py --query "where does the brazos river start" --model-name "all-mpnet-base-v2" --top-k 10
```

## Evaluating Pruning Performance

To evaluate pruning performance:

```bash
python compare_pruning_results.py --pruned-dir pruned_chunks --query "your query" --output pruning_comparison_results
```

