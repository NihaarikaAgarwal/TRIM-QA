import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#********************# DATA LOADING AND PREPROCESSING #********************#

class DataProcessor:
    def __init__(self, train_file: str, tables_file: str, test_file: str):
        self.train_file = train_file
        self.tables_file = tables_file
        self.test_file = test_file
        self.tables_data = {}
        self.load_tables()
    
    def load_tables(self):
        """Load table content from tables.jsonl"""
        logger.info("Loading table data...")
        with open(self.tables_file, 'r') as f:
            for line in tqdm(f, desc="Loading tables"):
                table = json.loads(line.strip())
                self.tables_data[table['tableId']] = table
        logger.info(f"Loaded {len(self.tables_data)} tables")
    
    def get_table_content(self, table_id: str) -> str:
        """Extract table content as text"""
        if table_id not in self.tables_data:
            return ""
        
        table = self.tables_data[table_id]
        content_parts = []
        
        # Add table caption/title
        if 'documentTitle' in table:
            content_parts.append(table['documentTitle'])
        
        # Add table content
        if 'tableData' in table:
            for row in table['tableData']:
                if isinstance(row, list):
                    content_parts.append(" | ".join(str(cell) for cell in row))
        
        return " ".join(content_parts)
    
    def process_train_data(self) -> List[Dict]:
        """Process training data to extract hard negatives"""
        logger.info("Processing training data...")
        
        with open(self.train_file, 'r') as f:
            train_data = json.load(f)
        
        processed_samples = []
        skipped_count = 0
        
        for sample in tqdm(train_data, desc="Processing training samples"):
            query = sample['query']
            gold_table_id = sample['gold_table_id']
            top_tables = sample.get('top_tables', [])
            
            # Find gold table rank
            gold_rank = None
            for i, (table_id, score) in enumerate(top_tables):
                if table_id == gold_table_id:
                    gold_rank = i + 1
                    break
            
            if gold_rank is None or gold_rank <= 3:
                skipped_count += 1
                continue
            
            # Get hard negatives (tables ranked above gold table)
            hard_negatives = [table_id for table_id, _ in top_tables[:gold_rank-1]]
            
            # Take top 3 hard negatives
            selected_negatives = hard_negatives[:3]
            
            if len(selected_negatives) == 0:
                skipped_count += 1
                continue
            
            processed_samples.append({
                'query': query,
                'gold_table_id': gold_table_id,
                'negative_table_ids': selected_negatives,
                'original_gold_rank': gold_rank
            })
        
        logger.info(f"Processed {len(processed_samples)} samples, skipped {skipped_count}")
        return processed_samples

#********************# SPLADE EMBEDDING GENERATION #********************#

class SpladeEmbedder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Generate SPLADE embeddings for texts"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get sparse embeddings (assuming SPLADE output format)
                batch_embeddings = torch.log(1 + torch.relu(outputs.last_hidden_state))
                batch_embeddings = torch.max(batch_embeddings, dim=1)[0]  # Max pooling
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

#********************# NEURAL RANKING MODEL #********************#

class TableRankingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], output_dim: int = 1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, query_emb: torch.Tensor, table_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [batch_size, embedding_dim]
            table_emb: [batch_size, embedding_dim]
        Returns:
            scores: [batch_size, 1] - relevance scores
        """
        # Concatenate query and table embeddings
        combined = torch.cat([query_emb, table_emb], dim=1)
        
        print(f"Input shape: {combined.shape}")
        scores = self.network(combined)
        print(f"Output shape: {scores.shape}")
        
        return scores

#********************# DATASET CLASS #********************#

class TableRankingDataset(Dataset):
    def __init__(self, samples: List[Dict], query_embeddings: np.ndarray, 
                 table_embeddings: Dict[str, np.ndarray]):
        self.samples = samples
        self.query_embeddings = query_embeddings
        self.table_embeddings = table_embeddings
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        query_emb = torch.FloatTensor(self.query_embeddings[idx])
        
        # Get positive table embedding
        pos_table_emb = torch.FloatTensor(self.table_embeddings[sample['gold_table_id']])
        
        # Get negative table embedding (randomly select one from negatives)
        neg_table_id = np.random.choice(sample['negative_table_ids'])
        neg_table_emb = torch.FloatTensor(self.table_embeddings[neg_table_id])
        
        return {
            'query_emb': query_emb,
            'pos_table_emb': pos_table_emb,
            'neg_table_emb': neg_table_emb,
            'original_rank': sample['original_gold_rank']
        }

#********************# TRAINING COMPONENTS #********************#

class RankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor):
        # Calculate similarities
        pos_sim = self.cosine_sim(query_emb, pos_emb)
        neg_sim = self.cosine_sim(query_emb, neg_emb)
        
        # Margin ranking loss: minimize distance to positive, maximize distance to negative
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0).mean()
        
        return loss, {
            'pos_similarity': pos_sim.mean().item(),
            'neg_similarity': neg_sim.mean().item()
        }

class TableRankingTrainer:
    def __init__(self, model: TableRankingModel, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = RankingLoss()
        
        # Best hyperparameters (you can tune these)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'original_rank'}
            
            self.optimizer.zero_grad()
            
            # Forward pass for positive and negative samples
            pos_scores = self.model(batch['query_emb'], batch['pos_table_emb'])
            neg_scores = self.model(batch['query_emb'], batch['neg_table_emb'])
            
            # Calculate loss
            loss, metrics = self.criterion(
                batch['query_emb'], 
                batch['pos_table_emb'], 
                batch['neg_table_emb']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pos_sim += metrics['pos_similarity']
            total_neg_sim += metrics['neg_similarity']
        
        avg_loss = total_loss / len(train_loader)
        self.scheduler.step(avg_loss)
        
        return {
            'loss': avg_loss,
            'pos_similarity': total_pos_sim / len(train_loader),
            'neg_similarity': total_neg_sim / len(train_loader),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader, val_samples: List[Dict]) -> Dict:
        self.model.eval()
        
        ranking_improvements = []
        total_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                original_ranks = [sample['original_gold_rank'] for sample in val_samples[i*val_loader.batch_size:(i+1)*val_loader.batch_size]]
                
                batch_items = {k: v.to(self.device) for k, v in batch.items() if k != 'original_rank'}
                
                # Calculate loss
                loss, _ = self.criterion(
                    batch_items['query_emb'], 
                    batch_items['pos_table_emb'], 
                    batch_items['neg_table_emb']
                )
                total_loss += loss.item()
                
                # Calculate scores for ranking
                pos_scores = self.model(batch_items['query_emb'], batch_items['pos_table_emb'])
                
                # Store improvement metrics
                for j, orig_rank in enumerate(original_ranks):
                    ranking_improvements.append({
                        'original_rank': orig_rank,
                        'predicted_score': pos_scores[j].item()
                    })
        
        # Calculate ranking statistics
        avg_original_rank = np.mean([r['original_rank'] for r in ranking_improvements])
        
        return {
            'val_loss': total_loss / len(val_loader),
            'avg_original_rank': avg_original_rank,
            'num_samples': len(ranking_improvements)
        }

#********************# INFERENCE AND EVALUATION #********************#

class TableRankingInference:
    def __init__(self, model: TableRankingModel, embedder: SpladeEmbedder, 
                 processor: DataProcessor, device: torch.device):
        self.model = model
        self.embedder = embedder
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def evaluate_test_set(self, test_file: str) -> Dict:
        """Evaluate on test set and calculate ranking improvements"""
        
        # Load test data
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        results = []
        recall_metrics = {'@1': 0, '@5': 0, '@10': 0, '@20': 0}
        total_queries = 0
        
        for sample in tqdm(test_data, desc="Evaluating test set"):
            query = sample['query']
            gold_table_id = sample['gold_table_id']
            top_50_tables = sample['top_50_tables']  # List of (table_id, score) tuples
            
            # Generate query embedding
            query_emb = self.embedder.encode([query])[0]
            query_tensor = torch.FloatTensor(query_emb).unsqueeze(0).to(self.device)
            
            # Get table contents and generate embeddings
            table_contents = []
            table_ids = []
            
            for table_id, _ in top_50_tables:
                content = self.processor.get_table_content(table_id)
                table_contents.append(content)
                table_ids.append(table_id)
            
            # Generate table embeddings
            table_embeddings = self.embedder.encode(table_contents)
            
            # Calculate relevance scores using the model
            relevance_scores = []
            
            with torch.no_grad():
                for i, table_emb in enumerate(table_embeddings):
                    table_tensor = torch.FloatTensor(table_emb).unsqueeze(0).to(self.device)
                    score = self.model(query_tensor, table_tensor)
                    relevance_scores.append((table_ids[i], score.item()))
            
            # Sort by relevance score (descending)
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find new rank of gold table
            new_rank = None
            original_rank = None
            
            for i, (table_id, _) in enumerate(top_50_tables):
                if table_id == gold_table_id:
                    original_rank = i + 1
                    break
            
            for i, (table_id, _) in enumerate(relevance_scores):
                if table_id == gold_table_id:
                    new_rank = i + 1
                    break
            
            if original_rank and new_rank:
                results.append({
                    'query': query,
                    'gold_table_id': gold_table_id,
                    'original_rank': original_rank,
                    'new_rank': new_rank,
                    'improvement': original_rank - new_rank
                })
                
                # Calculate recall metrics
                total_queries += 1
                if new_rank <= 1:
                    recall_metrics['@1'] += 1
                if new_rank <= 5:
                    recall_metrics['@5'] += 1
                if new_rank <= 10:
                    recall_metrics['@10'] += 1
                if new_rank <= 20:
                    recall_metrics['@20'] += 1
        
        # Calculate final metrics
        for k in recall_metrics:
            recall_metrics[k] = (recall_metrics[k] / total_queries) * 100
        
        avg_improvement = np.mean([r['improvement'] for r in results])
        avg_new_rank = np.mean([r['new_rank'] for r in results])
        avg_original_rank = np.mean([r['original_rank'] for r in results])
        
        return {
            'results': results,
            'recall_metrics': recall_metrics,
            'avg_improvement': avg_improvement,
            'avg_new_rank': avg_new_rank,
            'avg_original_rank': avg_original_rank,
            'total_queries': total_queries
        }

#********************# MAIN EXECUTION PIPELINE #********************#

def main():
    # Initialize components
    processor = DataProcessor(
        train_file="/home/mvyas7/TRIM-QA/Summer-Research/CRAFT_NQ_TEST_1st_Stage_results.json",
        tables_file="/home/mvyas7/TRIM-QA/Summer-Research/nq_tables/tables/tables.jsonl",
        test_file="/home/mvyas7/TRIM-QA/Summer-Research/nq_final_retrieval_result.pkl"
    )
    
    embedder = SpladeEmbedder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process training data
    train_samples = processor.process_train_data()
    
    # Split into train/validation
    train_split, val_split = train_test_split(train_samples, test_size=0.3, random_state=42)
    logger.info(f"Train samples: {len(train_split)}, Validation samples: {len(val_split)}")
    
    # Generate embeddings for queries
    train_queries = [sample['query'] for sample in train_split]
    val_queries = [sample['query'] for sample in val_split]
    
    logger.info("Generating query embeddings...")
    train_query_embeddings = embedder.encode(train_queries)
    val_query_embeddings = embedder.encode(val_queries)
    
    print(f"Query embedding dimensions: {train_query_embeddings.shape}")
    
    # Generate embeddings for all unique tables
    all_table_ids = set()
    for sample in train_samples:
        all_table_ids.add(sample['gold_table_id'])
        all_table_ids.update(sample['negative_table_ids'])
    
    logger.info("Generating table embeddings...")
    table_contents = []
    table_ids_list = list(all_table_ids)
    
    for table_id in tqdm(table_ids_list, desc="Getting table contents"):
        content = processor.get_table_content(table_id)
        table_contents.append(content)
    
    table_embeddings_array = embedder.encode(table_contents)
    table_embeddings_dict = {
        table_id: embedding 
        for table_id, embedding in zip(table_ids_list, table_embeddings_array)
    }
    
    print(f"Table embedding dimensions: {table_embeddings_array.shape}")
    
    # Initialize model
    input_dim = train_query_embeddings.shape[1] * 2  # Query + Table embeddings concatenated
    model = TableRankingModel(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],  # Best hyperparameters
        output_dim=1
    ).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TableRankingDataset(train_split, train_query_embeddings, table_embeddings_dict)
    val_dataset = TableRankingDataset(val_split, val_query_embeddings, table_embeddings_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    trainer = TableRankingTrainer(model, device)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(50):  # Max epochs
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader, val_split)
        
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Train Pos Sim: {train_metrics['pos_similarity']:.4f}")
        logger.info(f"  Train Neg Sim: {train_metrics['neg_similarity']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Avg Original Rank: {val_metrics['avg_original_rank']:.2f}")
        logger.info(f"  Learning Rate: {train_metrics['lr']:.6f}")
        
        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_ranking_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            logger.info("Early stopping triggered")
            break
    
    # Load best model for inference
    model.load_state_dict(torch.load('best_ranking_model.pth'))
    
    # Inference on test set
    logger.info("Starting inference on test set...")
    inference = TableRankingInference(model, embedder, processor, device)
    test_results = inference.evaluate_test_set(processor.test_file)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total test queries: {test_results['total_queries']}")
    print(f"Average original rank: {test_results['avg_original_rank']:.2f}")
    print(f"Average new rank: {test_results['avg_new_rank']:.2f}")
    print(f"Average improvement: {test_results['avg_improvement']:.2f}")
    print("\nRecall Metrics:")
    print(f"  Recall@1:  {test_results['recall_metrics']['@1']:.2f}%")
    print(f"  Recall@5:  {test_results['recall_metrics']['@5']:.2f}%")
    print(f"  Recall@10: {test_results['recall_metrics']['@10']:.2f}%")
    print(f"  Recall@20: {test_results['recall_metrics']['@20']:.2f}%")
    
    # Show some example improvements
    improved_samples = [r for r in test_results['results'] if r['improvement'] > 0]
    improved_samples.sort(key=lambda x: x['improvement'], reverse=True)
    
    print(f"\nTop 5 Ranking Improvements:")
    for i, sample in enumerate(improved_samples[:5]):
        print(f"  {i+1}. {sample['original_rank']} → {sample['new_rank']} (improvement: +{sample['improvement']})")
        print(f"     Query: {sample['query'][:80]}...")

if __name__ == "__main__":
    main()