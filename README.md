# CSE 576: Topics in NLP
### TRIM QA - Table Retrieval and Information Minimization for Question Answering


---

## Abstract
Pretrained Large Language Models (LLMs), while powerful, often struggle with specific queries over structured data due to their reliance on broad, generalized knowledge. Traditional Retrieval Augmented Generation (RAG) frameworks help by retrieving external context but treat all retrieved chunks equally, limiting effectiveness for structured table querying. We propose TRIM (Table Retrieval and Information Minimization), a retrieval enhancement that filters out irrelevant or distracting content from retrieved tables, allowing the LLM to focus only on the rows and columns with most question-relevant information—leading to clearer, more accurate responses. We evaluate our method on the NQ-Tables dataset, a benchmark focused on natural language querying over textual tables from Wikipedia.

---

## Pipeline Overview

### 1. BM25 Retrieval
BM25 is a probabilistic retrieval framework based on TF-IDF. It ranks documents based on their relevance to a given query. For our project:
- **Tokenization:** We experimented with basic tokenization, NLTK, and advanced tokenization using GPT-4's Tiktoken.
- **Stopword Removal:** Improved precision by eliminating low-information words.
- **Hyperparameter Tuning:** Optimized parameters (k1=1.5, b=0.75) for better recall.
- **Results:** Recall improved from 84.67% (baseline) to 96.67% with advanced tokenization.

### 2. Pruning
Pruning reduces irrelevant context and improves the focus of downstream processing. Our approach:
- **Hierarchical Chunking:** Tables are split into row and column chunks with metadata (Table ID, Chunk ID, Row ID, Column ID).
- **Semantic Pruning:**
  - Used Sentence-BERT (SBERT) to embed queries and table chunks into a shared semantic vector space.
  - Applied cosine similarity to score relevance of rows and columns.
  - Dynamic thresholding for filtering:
    - Rows: `threshold = median(similarity scores) + 0.5 × standard deviation`
    - Columns: `threshold = median(similarity scores) - 0.2 × standard deviation`
  - Final pruned table is derived by intersecting retained rows and columns.

### 3. Recall and Reranking
- **Reranking Models:**
  - TAPAS: Joint BERT-style encoder for query and table. Computationally expensive and suppressed recall.
  - SBERT: Lightweight bi-encoder with cosine similarity for reranking. Faster and more accurate.
- **Results:**
  - SBERT recall: 96.67% at N=2500 (slightly better than BM25).
  - Post-pruning recall improved top-10% and top-20% rankings.

---

## Dataset
- **NQ-Tables Dataset:**
  - Extracted from Wikipedia pages in the Google Natural Questions (NQ) benchmark.
  - ≈160k tables from 19,885 articles.
  - JSON schema allows exploration of structural components (rows vs. columns).

---

## Installation

### Requirements
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
Alternatively, install core dependencies directly:
```bash
pip install sentence-transformers faiss-cpu torch pandas numpy tqdm tabulate matplotlib colorama
```
For GPU support:
```bash
pip install sentence-transformers faiss-gpu torch pandas numpy tqdm tabulate matplotlib colorama
```

### Directory Setup
Ensure the following directories exist:
- `faiss_indexes/` - For storing FAISS indexes.
- `pruned_chunks/` - For storing pruned chunks output.
- `pruning_results/` - For storing pruning evaluation results.

Create them if they don't exist:
```bash
mkdir -p faiss_indexes pruned_chunks pruning_results
```

---

## Pipeline Workflow

### Step 1: BM25 Retrieval
Run BM25 retrieval to fetch top-k relevant tables:
```bash
python bm25_retrieval.py --query "your query here" --top-k 5000
```

### Step 2: Pruning
Prune irrelevant rows and columns:
```bash
python pruning_sbert.py --tables tables.jsonl --query "your query here" --output pruned_chunks
```

### Step 3: Reranking
Rerank pruned tables using SBERT:
```bash
python sbert_rerank_postpruning.py --query "your query here" --output reranked_results
```

---

## Results and Analysis

### Key Observations
- **BM25:**
  - Advanced tokenization improved recall by 12%.
  - Stopword removal and hyperparameter tuning further enhanced results.
- **Pruning:**
  - Post-pruning recall outperformed pre-pruning for smaller k values.
  - Column pruning alone was less effective than combined row+column pruning.
- **Reranking:**
  - SBERT surpassed TAPAS in recall and efficiency.
  - Post-pruning reranking improved top-10% and top-20% rankings.

### Figures and Tables
- **Figure 1:** BM25 Recall Comparison (Basic vs. Advanced Tokenization).
- **Figure 2:** Recall Comparison (TAPAS vs. SBERT).
- **Figure 3:** Recall (BM25 vs. Pre-pruning vs. Post-pruning).
- **Figure 4:** Top-10% & Top-20% Rankings (BM25 vs. Pre-pruning vs. Post-pruning).
- **Table 1:** Recall Results (BM25, Pre-pruning, Post-pruning).

---

## Shortcomings and Future Work
- **Limitations:**
  - Relevance scoring relies on surface-level semantic proximity.
  - Struggles with multilingual content transliterated into English.
- **Future Directions:**
  - Develop schema-aware pruning inspired by CABINET architecture.
  - Extend pipeline to handle complex, compositional queries.

---

## References
- [BM25 Weighting Function](https://www.researchgate.net/publication/221037764_Okapi)
- [CABINET: Context Relevance-Based Noise Reduction](https://arxiv.org/pdf/2402.01155)
- [SBERT: Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [TAPAS: Table Parsing](https://arxiv.org/pdf/2004.02349)

---

## Appendix
- **Presentation:** [Google Drive](https://drive.google.com/drive/folders/1OFOgt6PrCW023hjH5jXdoefXUSPo8qjG)
- **GitHub Repository:** [TRIM-QA](https://github.com/NihaarikaAgarwal/TRIM-QA)

