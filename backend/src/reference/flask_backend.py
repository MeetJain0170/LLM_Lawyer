"""
Flask API Backend for Legal LLM
Serves the model via REST API
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import json
import time
from threading import Lock

# Import your components
from legal_llm_model import LegalLLM
from bpe_tokenizer import BPETokenizer
from search_system import DocumentEncoder, LegalDocumentIndex, RAGLegalAssistant

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model state
model_lock = Lock()
assistant = None
device = None

def load_model():
    """Load model and create assistant"""
    global assistant, device
    
    print("Loading model...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("legal_tokenizer.pkl")
    
    # Load model
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
    model.eval()
    
    # Load document index
    encoder = DocumentEncoder(model)
    doc_index = LegalDocumentIndex(encoder, tokenizer, device)
    doc_index.load("legal_document_index.pkl")
    
    # Create assistant
    assistant = RAGLegalAssistant(model, tokenizer, doc_index, device)
    
    print("Model loaded successfully!")

# Load model on startup
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': assistant is not None,
        'device': str(device)
    })

@app.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint
    
    Request body:
    {
        "question": "What is Article 21?",
        "num_docs": 3,
        "max_tokens": 200,
        "temperature": 0.7
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request'}), 400
            
        question = data['question']
        num_docs = data.get('num_docs', 3)
        max_tokens = data.get('max_tokens', 200)
        temperature = data.get('temperature', 0.7)
        
        # Validate parameters
        if not question.strip():
            return jsonify({'error': 'Empty question'}), 400
            
        if num_docs < 1 or num_docs > 10:
            return jsonify({'error': 'num_docs must be between 1 and 10'}), 400
            
        if max_tokens < 10 or max_tokens > 500:
            return jsonify({'error': 'max_tokens must be between 10 and 500'}), 400
            
        # Thread-safe model inference
        with model_lock:
            start_time = time.time()
            
            result = assistant.answer_query(
                query=question,
                num_docs=num_docs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            inference_time = time.time() - start_time
            
        # Add metadata
        result['inference_time_seconds'] = round(inference_time, 2)
        result['question'] = question
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query_stream', methods=['POST'])
def query_stream():
    """
    Streaming endpoint for real-time token generation
    """
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'error': 'Empty question'}), 400
            
        def generate():
            """Generator function for streaming"""
            with model_lock:
                # Search for documents
                retrieved_docs = assistant.index.search(question, k=3)
                
                # Build context
                context_parts = []
                for i, (doc_text, score, metadata) in enumerate(retrieved_docs, 1):
                    doc_text = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
                    context_parts.append(f"[Document {i}]\n{doc_text}\n")
                    
                context = "\n".join(context_parts)
                
                # Build prompt
                prompt = f"""You are a legal expert on Indian Constitutional Law.

Retrieved Legal Documents:
{context}

Question: {question}

Based on the documents above, provide a clear and accurate answer:"""
                
                # Tokenize
                prompt_tokens = assistant.tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens]).to(assistant.device)
                
                # Generate token by token
                assistant.llm.eval()
                
                with torch.no_grad():
                    for _ in range(200):  # Max tokens
                        # Get logits
                        logits, _ = assistant.llm(input_ids)
                        next_token_logits = logits[:, -1, :] / 0.7
                        
                        # Sample
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Decode token
                        token_text = assistant.tokenizer.decode([next_token.item()])
                        
                        # Stream token
                        yield f"data: {json.dumps({'token': token_text})}\n\n"
                        
                        # Append to sequence
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        
                        # Stop if EOS
                        if next_token.item() == assistant.tokenizer.vocab.get(assistant.tokenizer.EOS_TOKEN):
                            break
                            
                # Send sources
                sources = [
                    {
                        'title': meta.get('title', 'Unknown'),
                        'relevance_score': float(score)
                    }
                    for _, score, meta in retrieved_docs
                ]
                
                yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """
    Document search endpoint (without generation)
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not query.strip():
            return jsonify({'error': 'Empty query'}), 400
            
        with model_lock:
            results = assistant.index.search(query, k=k)
            
        documents = [
            {
                'text': text[:500] + "..." if len(text) > 500 else text,
                'relevance_score': float(score),
                'metadata': metadata
            }
            for text, score, metadata in results
        ]
        
        return jsonify({'documents': documents})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    try:
        num_params = sum(p.numel() for p in assistant.llm.parameters())
        num_docs = len(assistant.index.documents)
        
        return jsonify({
            'model_parameters': num_params,
            'indexed_documents': num_docs,
            'vocab_size': len(assistant.tokenizer.vocab),
            'model_architecture': {
                'd_model': assistant.llm.d_model,
                'num_layers': len(assistant.llm.transformer_blocks),
                'max_seq_len': assistant.llm.max_seq_len
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False in production
        threaded=True
    )