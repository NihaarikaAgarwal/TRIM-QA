import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import jsonlines

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
        
        # Sample s from standard normal
        s = torch.randn_like(sigma)
        
        # Reparameterization: z = mu + s * sigma --- latent variable
        z = mu + s * sigma
        
        # Normalize with sigmoid
        eta_uns = torch.sigmoid(z)
        return eta_uns, mu, sigma

# # Example: Simulate processing a batch of token embeddings.
# hidden_dim = 768  # example hidden dimension from an LLM
# urs_model = URSModule(hidden_dim)
# # Assume batch_embeddings is a tensor from your LLM's encoder
# batch_embeddings = torch.randn(32, hidden_dim)  # simulate 32 tokens
# eta_uns, mu, sigma = urs_model(batch_embeddings)
# print("Unsupervised Relevance Scores:", eta_uns.shape)


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
        eta_combined = eta_uns * eta_ws
        return eta_combined, mu, sigma
    
#******************Pruning Function******************#
# This function prunes the chunks based on the combined relevance scores.
# It filters out chunks with scores below a certain threshold.
#--------------------------------------------------------------------------#    
    
def prune_chunks(chunks, scores, threshold=0.7):
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

# Main execution block
if __name__ == "__main__":
    # Initialize models
    hidden_dim = 768  # BERT/RoBERTa hidden dimension
    model_name = "bert-base-uncased"  # or any other preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = EmbeddingModule(AutoModel.from_pretrained(model_name))
    combined_model = CombinedModule(hidden_dim)
    
    # Read the first table from tables.jsonl
    with jsonlines.open('NQ-Dataset/tables/tables.jsonl', 'r') as reader:
        for first_table in reader:
            break  # Get only the first table
    
    # Read chunks from chunks.json
    chunks = []
    with jsonlines.open('chunks.json', 'r') as reader:
        for chunk in reader:
            chunks.append(chunk)
    
    # Read questions and gold tables from test.jsonl
    test_items = []
    with jsonlines.open('test.jsonl', 'r') as reader:
        for item in reader:
            if 'questions' in item:
                for question in item['questions']:
                    test_items.append({
                        'question': question['originalText'],
                        'answer': question['answer']['answerTexts']
                    })
    
    # Process each question
    for item in test_items:
        question = item['question']
        print(f"\nProcessing question: {question}")
        print(f"Expected answer(s): {item['answer']}")
        
        # Tokenize and get embeddings for chunks and question
        chunk_embeddings = []
        for chunk in chunks:
            # Combine chunk content into a single string
            chunk_text = " ".join([str(cell) for cell in chunk['text']])
            inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1)
                chunk_embeddings.append(embeddings)
        
        # Get question embedding
        question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            question_embedding = embedding_model.model(**question_inputs).last_hidden_state.mean(dim=1)
        
        chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
        
        # Get combined relevance scores using both question and chunk embeddings
        scores, _, _ = combined_model(chunk_embeddings)
        scores = scores.squeeze().tolist()
        
        # Scale scores based on similarity with question
        similarity_scores = torch.nn.functional.cosine_similarity(
            chunk_embeddings, 
            question_embedding.expand(chunk_embeddings.size(0), -1)
        ).tolist()
        
        # Combine URS scores with question similarity
        final_scores = [(s + sim) / 2 for s, sim in zip(scores, similarity_scores)]
        
        # Prune chunks based on final scores
        pruned_chunks = prune_chunks(chunks, final_scores)
        
        # Save pruned chunks
        output_filename = f"pruned_chunks_{question[:30].replace(' ', '_')}.json"
        with open(output_filename, 'w') as f:
            json.dump(pruned_chunks, f, indent=2)
        
        print(f"Original chunks: {len(chunks)}, Pruned chunks: {len(pruned_chunks)}")


