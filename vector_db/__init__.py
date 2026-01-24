"""
VQA Vector DB Package

Multimodal vector search for Visual Question Answering using:
- ViLT (Vision-and-Language Transformer) for multimodal embeddings
- Azure Cosmos DB for MongoDB vCore for vector storage
- VQAv2 dataset as knowledge base

ViLT is specifically designed for vision-language tasks like VQA,
unlike CLIP which is trained on image-caption pairs.
"""

from .vilt_embeddings import ViLTEmbeddings
from .cosmos_vector_db import CosmosVectorDB, AzureAISearchVectorDB
from .query_similar import VQAVectorSearch

__all__ = [
    "ViLTEmbeddings",
    "CosmosVectorDB", 
    "AzureAISearchVectorDB",
    "VQAVectorSearch"
]
