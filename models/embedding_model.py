import torch
from transformers import AutoTokenizer, AutoModel

class BioBERTEmbedder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        """Load BioBERT model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def get_embedding(self, text):
        """Generate BioBERT embeddings for a given medical text."""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Extract last hidden state and average token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

# Example Usage
if __name__ == "__main__":
    embedder = BioBERTEmbedder()
    
    sample_text = "HER2-positive breast cancer is associated with poor prognosis."
    embedding_vector = embedder.get_embedding(sample_text)
    
    print("Embedding Shape:", embedding_vector.shape)  # Expected: (768,)
    print("Sample Embedding:", embedding_vector[:5])  # Print first 5 values for preview
