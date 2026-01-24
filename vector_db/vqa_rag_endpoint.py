"""
VQA RAG Endpoint - Flask API for Vector-Based VQA

This endpoint uses multimodal vector search to answer questions:
1. Receives image + question
2. Searches for similar examples in vector database using ViLT embeddings
3. Returns answer based on similar training examples

Can be used alongside or as alternative to the neural network model.
Uses ViLT (Vision-and-Language Transformer) for embeddings.
"""

import os
import io
import base64
from flask import Blueprint, request, jsonify
from PIL import Image
import numpy as np
from datetime import datetime

# Import vector search components
from vector_db.vilt_embeddings import ViLTEmbeddings
from vector_db.cosmos_vector_db import CosmosVectorDB
from vector_db.query_similar import VQAVectorSearch


# Create Blueprint for RAG endpoints
rag_blueprint = Blueprint('rag', __name__, url_prefix='/rag')

# Global instances (initialized on first request)
_vector_searcher = None


def get_vector_searcher() -> VQAVectorSearch:
    """Get or initialize the vector searcher (singleton)."""
    global _vector_searcher
    
    if _vector_searcher is None:
        connection_string = os.getenv("COSMOS_CONNECTION_STRING")
        
        if not connection_string:
            raise ValueError("COSMOS_CONNECTION_STRING environment variable not set")
        
        _vector_searcher = VQAVectorSearch(
            cosmos_connection_string=connection_string,
            model_name="dandelin/vilt-b32-finetuned-vqa"
        )
    
    return _vector_searcher


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    # Remove data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


@rag_blueprint.route('/predict', methods=['POST'])
def rag_predict():
    """
    RAG-based VQA prediction endpoint.
    
    Request body:
    {
        "image": "<base64 encoded image>",
        "question": "What color is the car?",
        "voting_strategy": "weighted",  // optional: top1, majority, weighted
        "k": 5,  // optional: number of similar examples
        "include_similar": false  // optional: include similar examples in response
    }
    
    Response:
    {
        "answer": "red",
        "confidence": 0.85,
        "method": "weighted",
        "processing_time_ms": 150,
        "similar_examples": [...]  // if include_similar=true
    }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data.get("image"):
            return jsonify({"error": "No image provided"}), 400
        if not data.get("question"):
            return jsonify({"error": "No question provided"}), 400
        
        start_time = datetime.now()
        
        # Decode image
        image = decode_image(data["image"])
        question = data["question"]
        
        # Get parameters
        voting_strategy = data.get("voting_strategy", "weighted")
        k = data.get("k", 5)
        include_similar = data.get("include_similar", False)
        
        # Get vector searcher
        searcher = get_vector_searcher()
        
        # Get similar examples
        similar = searcher.find_similar(image, question, k=k)
        
        # Get answer
        answer_result = searcher.get_answer(
            image=image,
            question=question,
            voting_strategy=voting_strategy
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = {
            "answer": answer_result["answer"],
            "confidence": round(answer_result["confidence"], 4),
            "method": answer_result["method"],
            "processing_time_ms": round(processing_time, 2)
        }
        
        if include_similar:
            response["similar_examples"] = [
                {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "question_type": ex.get("question_type"),
                    "similarity_score": round(ex.get("similarity_score", 0), 4)
                }
                for ex in similar
            ]
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_blueprint.route('/similar', methods=['POST'])
def find_similar():
    """
    Find similar VQA examples without returning an answer.
    
    Useful for exploring the training set and debugging.
    
    Request body:
    {
        "image": "<base64 encoded image>",
        "question": "What is the man doing?",
        "k": 10,
        "filter_question_type": "action"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data.get("image") or not data.get("question"):
            return jsonify({"error": "Image and question required"}), 400
        
        image = decode_image(data["image"])
        question = data["question"]
        k = data.get("k", 10)
        filter_type = data.get("filter_question_type")
        
        searcher = get_vector_searcher()
        
        similar = searcher.find_similar(
            image=image,
            question=question,
            k=k,
            filter_question_type=filter_type
        )
        
        return jsonify({
            "query_question": question,
            "num_results": len(similar),
            "similar_examples": [
                {
                    "image_id": ex.get("image_id"),
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "question_type": ex.get("question_type"),
                    "similarity_score": round(ex.get("similarity_score", 0), 4)
                }
                for ex in similar
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_blueprint.route('/hybrid', methods=['POST'])
def hybrid_predict():
    """
    Hybrid prediction combining neural network and RAG.
    
    Uses both the trained model and vector search, then combines results.
    
    Request body:
    {
        "image": "<base64 encoded image>",
        "question": "What color is the car?",
        "model_weight": 0.7,  // Weight for neural network prediction
        "rag_weight": 0.3    // Weight for RAG prediction
    }
    """
    try:
        data = request.get_json()
        
        if not data.get("image") or not data.get("question"):
            return jsonify({"error": "Image and question required"}), 400
        
        image = decode_image(data["image"])
        question = data["question"]
        
        model_weight = data.get("model_weight", 0.7)
        rag_weight = data.get("rag_weight", 0.3)
        
        # Get RAG prediction
        searcher = get_vector_searcher()
        rag_result = searcher.get_answer(image, question, voting_strategy="weighted")
        
        # Get model prediction (would need to import from main.py)
        # For now, we just return RAG result
        # TODO: Integrate with neural network model
        
        return jsonify({
            "rag_answer": rag_result["answer"],
            "rag_confidence": round(rag_result["confidence"], 4),
            "model_weight": model_weight,
            "rag_weight": rag_weight,
            "note": "Hybrid mode - integrate with neural model for full functionality"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_blueprint.route('/health', methods=['GET'])
def rag_health():
    """Check RAG system health."""
    try:
        searcher = get_vector_searcher()
        doc_count = searcher.vector_db.get_document_count()
        
        return jsonify({
            "status": "healthy",
            "indexed_documents": doc_count,
            "model": "dandelin/vilt-b32-finetuned-vqa",
            "embedding_dim": 768
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# Cleanup on shutdown
def cleanup():
    """Clean up resources on shutdown."""
    global _vector_searcher
    if _vector_searcher:
        _vector_searcher.close()
        _vector_searcher = None


# Register cleanup
import atexit
atexit.register(cleanup)
