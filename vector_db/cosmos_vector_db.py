"""
Azure Cosmos DB for MongoDB vCore - Vector Database Operations

Handles:
- Connection to Cosmos DB
- Vector index creation (IVF or HNSW)
- Document insertion with embeddings
- Vector similarity search
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import numpy as np
from typing import List, Dict, Any, Optional
import os


class CosmosVectorDB:
    """Azure Cosmos DB for MongoDB vCore with vector search."""
    
    def __init__(
        self,
        connection_string: str = None,
        database_name: str = "vqa_vectors",
        collection_name: str = "predictions"
    ):
        """
        Initialize Cosmos DB connection.
        
        Args:
            connection_string: MongoDB connection string for Cosmos DB vCore
            database_name: Database name
            collection_name: Collection name
        """
        self.connection_string = connection_string or os.getenv("COSMOS_CONNECTION_STRING")
        
        if not self.connection_string:
            raise ValueError(
                "Connection string required. Set COSMOS_CONNECTION_STRING env var or pass directly."
            )
        
        # Connect to Cosmos DB
        print("Connecting to Cosmos DB for MongoDB vCore...")
        self.client = MongoClient(self.connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        print(f"✓ Connected to database: {database_name}, collection: {collection_name}")
    
    def create_vector_index(
        self,
        index_name: str = "vqa_vector_index",
        vector_field: str = "embedding",
        dimensions: int = 768,
        similarity: str = "COS",  # COS, L2, or IP
        index_type: str = "hnsw",  # hnsw (faster queries) or ivf (faster build)
        num_lists: int = 100  # For IVF index
    ):
        """
        Create a vector search index.
        
        Args:
            index_name: Name of the index
            vector_field: Field containing the vector
            dimensions: Vector dimensions (768 for ViLT)
            similarity: Similarity metric (COS, L2, IP)
            index_type: "hnsw" (faster queries, recommended) or "ivf" (faster build)
            num_lists: Number of clusters for IVF
        """
        # Check if index exists
        existing_indexes = list(self.collection.list_search_indexes())
        if any(idx.get("name") == index_name for idx in existing_indexes):
            print(f"Index '{index_name}' already exists")
            return
        
        # Define vector index
        if index_type == "ivf":
            index_definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": dimensions,
                        "similarity": similarity,
                        "numLists": num_lists
                    }
                ]
            }
        else:  # hnsw
            index_definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": dimensions,
                        "similarity": similarity,
                        "m": 16,  # Max connections per node
                        "efConstruction": 64  # Build-time search width
                    }
                ]
            }
        
        # Create the index
        search_index_model = SearchIndexModel(
            definition=index_definition,
            name=index_name
        )
        
        self.collection.create_search_index(search_index_model)
        print(f"✓ Created {index_type.upper()} vector index: {index_name}")
        print(f"  Dimensions: {dimensions}, Similarity: {similarity}")
    
    def insert_vqa_document(
        self,
        image_id: str,
        question_id: str,
        question: str,
        answer: str,
        embedding: np.ndarray,
        image_path: str = None,
        question_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Insert a single VQA document with embedding.
        
        Args:
            image_id: COCO image ID
            question_id: VQA question ID
            question: Question text
            answer: Answer text
            embedding: Vector embedding (numpy array)
            image_path: Path to image file
            question_type: Type of question (what, where, how many, etc.)
            metadata: Additional metadata
            
        Returns:
            Inserted document ID
        """
        document = {
            "image_id": image_id,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "embedding": embedding.tolist(),  # Convert numpy to list
            "image_path": image_path,
            "question_type": question_type,
            **(metadata or {})
        }
        
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def bulk_insert(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Bulk insert VQA documents.
        
        Args:
            documents: List of documents with embeddings
            batch_size: Batch size for insertion
            
        Returns:
            Number of inserted documents
        """
        total_inserted = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Convert numpy arrays to lists
            for doc in batch:
                if isinstance(doc.get("embedding"), np.ndarray):
                    doc["embedding"] = doc["embedding"].tolist()
            
            result = self.collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            
            print(f"Inserted {total_inserted}/{len(documents)} documents")
        
        return total_inserted
    
    def vector_search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        index_name: str = "vqa_vector_index",
        vector_field: str = "embedding",
        filter_query: Dict = None,
        include_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding (numpy array)
            k: Number of results to return
            index_name: Name of the vector index
            vector_field: Field containing vectors
            filter_query: Optional MongoDB filter
            include_score: Include similarity score
            
        Returns:
            List of matching documents with scores
        """
        # Build aggregation pipeline
        vector_search_stage = {
            "$search": {
                "cosmosSearch": {
                    "vector": query_vector.tolist(),
                    "path": vector_field,
                    "k": k
                },
                "returnStoredSource": True
            }
        }
        
        # Add filter if provided
        if filter_query:
            vector_search_stage["$search"]["cosmosSearch"]["filter"] = filter_query
        
        pipeline = [vector_search_stage]
        
        # Add score
        if include_score:
            pipeline.append({
                "$addFields": {
                    "similarity_score": {"$meta": "searchScore"}
                }
            })
        
        # Project out the embedding (too large to return)
        pipeline.append({
            "$project": {
                "embedding": 0
            }
        })
        
        results = list(self.collection.aggregate(pipeline))
        return results
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        text_query: str,
        k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and text search.
        
        Args:
            query_vector: Query embedding
            text_query: Text to search in questions
            k: Number of results
            vector_weight: Weight for vector similarity
            text_weight: Weight for text match
            
        Returns:
            Combined search results
        """
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_vector.tolist(),
                        "path": "embedding",
                        "k": k * 2  # Get more candidates for reranking
                    }
                }
            },
            {
                "$addFields": {
                    "vector_score": {"$meta": "searchScore"}
                }
            },
            # Text match scoring
            {
                "$addFields": {
                    "text_score": {
                        "$cond": {
                            "if": {
                                "$regexMatch": {
                                    "input": {"$toLower": "$question"},
                                    "regex": text_query.lower()
                                }
                            },
                            "then": 1.0,
                            "else": 0.0
                        }
                    }
                }
            },
            # Combined score
            {
                "$addFields": {
                    "combined_score": {
                        "$add": [
                            {"$multiply": ["$vector_score", vector_weight]},
                            {"$multiply": ["$text_score", text_weight]}
                        ]
                    }
                }
            },
            {"$sort": {"combined_score": -1}},
            {"$limit": k},
            {"$project": {"embedding": 0}}
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def get_document_count(self) -> int:
        """Get total document count."""
        return self.collection.count_documents({})
    
    def delete_all(self):
        """Delete all documents (use with caution!)."""
        result = self.collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents")
    
    def close(self):
        """Close the connection."""
        self.client.close()
        print("✓ Connection closed")


# Alternative: Azure AI Search (if Cosmos DB not available)
class AzureAISearchVectorDB:
    """
    Azure AI Search with vector capabilities.
    Alternative to Cosmos DB for more advanced search features.
    """
    
    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        index_name: str = "vqa-vectors"
    ):
        """
        Initialize Azure AI Search.
        
        Args:
            endpoint: Azure AI Search endpoint
            api_key: Admin API key
            index_name: Index name
        """
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        
        credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(self.endpoint, credential)
        self.search_client = SearchClient(self.endpoint, self.index_name, credential)
        
        print(f"✓ Connected to Azure AI Search: {self.endpoint}")
    
    def create_index(self, dimensions: int = 512):
        """Create search index with vector field."""
        from azure.search.documents.indexes.models import (
            SearchIndex,
            SimpleField,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            VectorSearch,
            HnswAlgorithmConfiguration,
            VectorSearchProfile
        )
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="image_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="question_id", type=SearchFieldDataType.String),
            SearchableField(name="question", type=SearchFieldDataType.String),
            SearchableField(name="answer", type=SearchFieldDataType.String),
            SimpleField(name="question_type", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name="vqa-vector-profile"
            )
        ]
        
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="vqa-hnsw")
            ],
            profiles=[
                VectorSearchProfile(
                    name="vqa-vector-profile",
                    algorithm_configuration_name="vqa-hnsw"
                )
            ]
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        self.index_client.create_or_update_index(index)
        print(f"✓ Created index: {self.index_name}")
    
    def vector_search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_expr: str = None
    ) -> List[Dict]:
        """Vector similarity search."""
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_vector.tolist(),
            k_nearest_neighbors=k,
            fields="embedding"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=["id", "image_id", "question", "answer", "question_type"]
        )
        
        return [dict(r) for r in results]


if __name__ == "__main__":
    print("Testing Cosmos DB Vector operations...")
    print("Set COSMOS_CONNECTION_STRING environment variable to test")
