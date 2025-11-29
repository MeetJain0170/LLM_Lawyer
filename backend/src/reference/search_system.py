"""
Legal Document Search and Retrieval System
Implements vector search over constitutional documents
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json
from pathlib import Path
import pickle

class DocumentEncoder(torch.nn.Module):
    """
    Simple encoder to create document embeddings
    Uses mean pooling over transformer outputs
    """
    def __init__(self, llm_model):
        super().__init__()
        self.llm = llm_model
        
    def forward(self, input_ids):
        """
        Encode documents to fixed-size vectors
        """
        # Get hidden states from LLM
        self.llm.eval()
        with torch.no_grad():
            # Forward through transformer blocks
            x = self.llm.token_embedding(input_ids)
            x = x * np.sqrt(self.llm.d_model)
            x = self.llm.pos_encoding(x)
            
            mask = self.llm.create_causal_mask(input_ids.size(1)).to(input_ids.device)
            
            for block in self.llm.transformer_blocks:
                x = block(x, mask)
                
            x = self.llm.ln_f(x)
            
        # Mean pooling over sequence dimension
        # Shape: (batch, seq_len, d_model) -> (batch, d_model)
        embeddings = x.mean(dim=1)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class LegalDocumentIndex:
    """
    Vector database for legal documents
    Uses cosine similarity for retrieval
    """
    def __init__(self, encoder, tokenizer, device='cpu'):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        
        # Storage
        self.documents = []  # Original documents
        self.embeddings = []  # Document embeddings
        self.metadata = []  # Document metadata
        
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the index
        
        documents: List of dicts with keys: 'text', 'title', 'source', etc.
        """
        print(f"Indexing {len(documents)} documents...")
        
        for doc in documents:
            # Tokenize
            token_ids = self.tokenizer.encode(doc['text'])
            
            # Truncate if needed
            max_len = 512
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                
            # Convert to tensor
            input_ids = torch.tensor([token_ids]).to(self.device)
            
            # Get embedding
            embedding = self.encoder(input_ids).cpu().numpy()
            
            # Store
            self.documents.append(doc['text'])
            self.embeddings.append(embedding)
            self.metadata.append({
                'title': doc.get('title', ''),
                'source': doc.get('source', ''),
                'type': doc.get('type', '')
            })
            
        # Convert embeddings to numpy array
        self.embeddings = np.vstack(self.embeddings)
        
        print(f"Index built with {len(self.documents)} documents")
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for most relevant documents
        
        Returns: List of (document_text, similarity_score, metadata)
        """
        # Encode query
        token_ids = self.tokenizer.encode(query)
        if len(token_ids) > 512:
            token_ids = token_ids[:512]
            
        input_ids = torch.tensor([token_ids]).to(self.device)
        query_embedding = self.encoder(input_ids).cpu().numpy()
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding.T).squeeze()
        
        # Get top k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((
                self.documents[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))
            
        return results
    
    def save(self, filepath: str):
        """Save index to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Index saved to {filepath}")
        
    def load(self, filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        
        print(f"Index loaded from {filepath}")

class RAGLegalAssistant:
    """
    Retrieval-Augmented Generation Legal Assistant
    Combines search with LLM generation
    """
    def __init__(self, llm_model, tokenizer, document_index, device='cpu'):
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.index = document_index
        self.device = device
        
        self.llm.eval()
        
    def answer_query(
        self,
        query: str,
        num_docs: int = 3,
        max_new_tokens: int = 200,
        temperature: float = 0.7
    ) -> Dict:
        """
        Answer a legal query using RAG
        
        Returns: Dict with 'answer', 'sources', 'retrieved_docs'
        """
        # Step 1: Retrieve relevant documents
        print(f"Searching for: {query}")
        retrieved_docs = self.index.search(query, k=num_docs)
        
        # Step 2: Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, (doc_text, score, metadata) in enumerate(retrieved_docs, 1):
            # Truncate long documents
            doc_text = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
            
            context_parts.append(f"[Document {i}]\n{doc_text}\n")
            sources.append({
                'title': metadata.get('title', 'Unknown'),
                'source': metadata.get('source', 'Unknown'),
                'relevance_score': score
            })
            
        context = "\n".join(context_parts)
        
        # Step 3: Build prompt
        prompt = f"""You are a legal expert on Indian Constitutional Law.

Retrieved Legal Documents:
{context}

Question: {query}

Based on the documents above, provide a clear and accurate answer:"""
        
        # Step 4: Generate answer
        print("Generating answer...")
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens]).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.llm.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50
            )
            
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        
        # Extract just the answer (after the prompt)
        answer = generated_text[len(prompt):]
        
        return {
            'answer': answer.strip(),
            'sources': sources,
            'retrieved_docs': [doc for doc, _, _ in retrieved_docs]
        }
    
    def interactive_mode(self):
        """
        Interactive Q&A session
        """
        print("\n" + "="*60)
        print("Indian Constitutional Law Assistant")
        print("="*60)
        print("Ask me anything about Indian Constitutional Law!")
        print("Type 'quit' to exit.\n")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
                
            try:
                result = self.answer_query(query)
                
                print("\n" + "-"*60)
                print("ANSWER:")
                print(result['answer'])
                
                print("\n" + "-"*60)
                print("SOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['title']}")
                    print(f"   Relevance: {source['relevance_score']:.2f}")
                    print(f"   Source: {source['source']}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    from legal_llm_model import LegalLLM
    from bpe_tokenizer import BPETokenizer
    
    # Load trained model and tokenizer
    print("Loading model...")
    tokenizer = BPETokenizer()
    tokenizer.load("legal_tokenizer.pkl")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LegalLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create encoder
    encoder = DocumentEncoder(model)
    
    # Create document index
    print("Building document index...")
    doc_index = LegalDocumentIndex(encoder, tokenizer, device)
    
    # Load legal documents
    documents = []
    with open("data/indian_law_dataset.jsonl", 'r') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
            
    doc_index.add_documents(documents)
    
    # Save index
    doc_index.save("legal_document_index.pkl")
    
    # Create RAG assistant
    assistant = RAGLegalAssistant(model, tokenizer, doc_index, device)
    
    # Start interactive mode
    assistant.interactive_mode()