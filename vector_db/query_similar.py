"""
Query Similar VQA Examples from Vector Database

Given a new image and question, find the most similar examples
from the VQAv2 training set and return their answers.

Uses ViLT (Vision-and-Language Transformer) for multimodal embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from PIL import Image
import os

from vilt_embeddings import ViLTEmbeddings
from cosmos_vector_db import CosmosVectorDB


class VQAVectorSearch:
    """
    Search for similar VQA examples using multimodal embeddings.
    
    This implements a RAG-like system for VQA:
    1. Encode the new image + question using ViLT
    2. Search for similar examples in the vector database
    3. Return the answers from similar examples
    """
    
    def __init__(
        self,
        cosmos_connection_string: str = None,
        model_name: str = "dandelin/vilt-b32-finetuned-vqa",
        database_name: str = "vqa_vectors",
        collection_name: str = "embeddings"
    ):
        """
        Initialize the vector search system.
        
        Args:
            cosmos_connection_string: Cosmos DB connection string
            model_name: ViLT model to use
            database_name: Cosmos DB database name
            collection_name: Collection name
        """
        # Initialize ViLT encoder
        print("Initializing VQA Vector Search with ViLT...")
        self.vilt_encoder = ViLTEmbeddings(model_name=model_name)
        
        # Initialize database connection
        self.vector_db = CosmosVectorDB(
            connection_string=cosmos_connection_string,
            database_name=database_name,
            collection_name=collection_name
        )
        
        print("✓ VQA Vector Search ready")
    
    def find_similar(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        k: int = 5,
        filter_question_type: str = None
    ) -> List[Dict]:
        """
        Find similar VQA examples.
        
        Args:
            image: Image path, PIL Image, or numpy array
            question: Question text
            k: Number of similar examples to return
            filter_question_type: Optional filter by question type
            
        Returns:
            List of similar examples with answers and similarity scores
        """
        # Generate embedding for query using ViLT
        query_embedding = self.vilt_encoder.encode_image_question(
            image=image,
            question=question
        )
        
        # Build filter if specified
        filter_query = None
        if filter_question_type:
            filter_query = {"question_type": filter_question_type}
        
        # Search vector database
        results = self.vector_db.vector_search(
            query_vector=query_embedding,
            k=k,
            index_name="vqa_vector_index",
            filter_query=filter_query
        )
        
        return results
    
    def get_answer(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        voting_strategy: str = "weighted"
    ) -> Dict:
        """
        Get the best answer for a new question using RAG.
        
        Args:
            image: Image to query
            question: Question to answer
            voting_strategy: How to combine similar answers:
                - "top1": Return answer from most similar example
                - "majority": Return most common answer among top-k
                - "weighted": Weight answers by similarity score
                
        Returns:
            Dict with predicted answer and confidence
        """
        # Get similar examples
        similar = self.find_similar(image, question, k=5)
        
        if not similar:
            return {
                "answer": "unknown",
                "confidence": 0.0,
                "method": "no_matches"
            }
        
        if voting_strategy == "top1":
            # Simply return the most similar answer
            best = similar[0]
            return {
                "answer": best["answer"],
                "confidence": best.get("similarity_score", 0.0),
                "method": "top1",
                "similar_question": best["question"]
            }
        
        elif voting_strategy == "majority":
            # Majority voting among top-k
            from collections import Counter
            answers = [r["answer"] for r in similar]
            counter = Counter(answers)
            most_common = counter.most_common(1)[0]
            
            return {
                "answer": most_common[0],
                "confidence": most_common[1] / len(answers),
                "method": "majority",
                "vote_count": most_common[1]
            }
        
        else:  # weighted
            # Weight answers by similarity score
            answer_scores = {}
            
            for result in similar:
                answer = result["answer"]
                score = result.get("similarity_score", 1.0)
                
                if answer not in answer_scores:
                    answer_scores[answer] = 0.0
                answer_scores[answer] += score
            
            # Get best weighted answer
            best_answer = max(answer_scores.items(), key=lambda x: x[1])
            total_score = sum(answer_scores.values())
            
            return {
                "answer": best_answer[0],
                "confidence": best_answer[1] / total_score if total_score > 0 else 0.0,
                "method": "weighted",
                "answer_scores": answer_scores
            }
    
    def explain_answer(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        k: int = 3
    ) -> Dict:
        """
        Get answer with detailed explanation of similar examples.
        
        Useful for debugging and understanding the retrieval.
        """
        similar = self.find_similar(image, question, k=k)
        answer = self.get_answer(image, question, voting_strategy="weighted")
        
        return {
            "predicted_answer": answer["answer"],
            "confidence": answer["confidence"],
            "similar_examples": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "question_type": r.get("question_type"),
                    "similarity": r.get("similarity_score", 0.0)
                }
                for r in similar
            ]
        }
    
    def close(self):
        """Close database connection."""
        self.vector_db.close()


def demo():
    """Demo the vector search system."""
    print("=" * 60)
    print("VQA Vector Search Demo")
    print("=" * 60)
    
    # This would require a populated database
    connection_string = os.getenv("COSMOS_CONNECTION_STRING")
    
    if not connection_string:
        print("Set COSMOS_CONNECTION_STRING to run demo")
        print("\nExample usage:")
        print("""
from query_similar import VQAVectorSearch

# Initialize
searcher = VQAVectorSearch(cosmos_connection_string="...")

# Find similar examples
results = searcher.find_similar(
    image="path/to/image.jpg",
    question="What color is the car?"
)

for r in results:
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Similarity: {r['similarity_score']:.3f}")
    print()

# Get answer with RAG
answer = searcher.get_answer(
    image="path/to/image.jpg",
    question="What color is the car?",
    voting_strategy="weighted"
)
print(f"Predicted: {answer['answer']} (confidence: {answer['confidence']:.2f})")
""")
        return
    
    searcher = VQAVectorSearch(cosmos_connection_string=connection_string)
    
    # Example query (would need a real image)
    print("\nReady to query! Use searcher.find_similar() or searcher.get_answer()")
    
    searcher.close()


if __name__ == "__main__":
    demo()
