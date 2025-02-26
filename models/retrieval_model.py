import faiss
import numpy as np
import pickle
from embedding_model import BioBERTEmbedder

class FaissRetrieval:
    def __init__(self, embedding_dim=768, index_path="faiss_index.bin"):
        """Initialize FAISS index for similarity search."""
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        self.text_data = []  # Store corresponding text entries
        
        # Load existing index if available
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            print("FAISS index loaded successfully.")
        except FileNotFoundError:
            print("Creating a new FAISS index.")

    def add_embeddings(self, texts, embedder):
        """Convert texts into embeddings and store them in FAISS."""
        for text in texts:
            embedding = embedder.get_embedding(text)
            embedding = np.expand_dims(embedding, axis=0)  # Convert to 2D array
            self.index.add(embedding)
            self.text_data.append(text)
        
        # Save updated index
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

    def retrieve_similar_texts(self, query, embedder, top_k=5):
        """Retrieve top-k most relevant texts based on BioBERT embeddings."""
        query_embedding = embedder.get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = [(self.text_data[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

# Example Usage
if __name__ == "__main__":
    embedder = BioBERTEmbedder()
    retriever = FaissRetrieval()

    # Sample pathology knowledge texts
    pathology_texts = [
        "HER2-positive breast cancer is aggressive and requires targeted therapy.",
        "PD-L1 expression is a key biomarker in immunotherapy.",
        "KRAS mutations are commonly seen in colorectal cancer.",
        "TNM staging is critical for cancer prognosis."
    ]
    
    # Add embeddings to FAISS
    retriever.add_embeddings(pathology_texts, embedder)

    # Query retrieval
    query_text = "What are the key biomarkers in breast cancer?"
    retrieved_texts = retriever.retrieve_similar_texts(query_text, embedder)

    print("\n🔍 Top retrieved texts:")
    for text, score in retrieved_texts:
        print(f"- {text} (Score: {score:.4f})")
