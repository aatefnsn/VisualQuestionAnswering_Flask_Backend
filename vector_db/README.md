# VQA Multimodal Vector Search with Azure ML & Cosmos DB

A complete Visual Question Answering (VQA) system using multimodal embeddings, vector search, and Azure cloud services.

## 🎯 Overview

This system uses **ViLT** (Vision-and-Language Transformer) to generate embeddings for image-question pairs from the VQAv2 dataset, stores them in **Azure Cosmos DB for MongoDB vCore** with vector indexing, and enables semantic search to find answers for new questions.

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   VQAv2     │───▶│    ViLT     │───▶│   Cosmos DB    │  │
│  │  Dataset    │    │   Model     │    │  Vector Index  │  │
│  │ (~148k QA)  │    │ (GPU/T4)    │    │   (768-dim)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                ↓             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  New Image  │───▶│    ViLT     │───▶│  Vector Search │  │
│  │ + Question  │    │   Embed     │    │  → Top-K Match │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                ↓             │
│                                         ┌─────────────────┐  │
│                                         │  Return Answer  │  │
│                                         │  (Majority Vote)│  │
│                                         └─────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Azure Subscription** with access to:
  - Azure Cosmos DB for MongoDB vCore
  - Azure Machine Learning (for GPU compute)
- **VQAv2 Dataset** access
- **Python 3.8+**

## 🚀 Quick Start

### 1. Set Up Azure Cosmos DB

Follow [AZURE_COSMOS_SETUP.md](AZURE_COSMOS_SETUP.md) to create your vector database.

### 2. Set Up Azure ML Compute

1. Go to [Azure ML Studio](https://ml.azure.com)
2. Create a new workspace (or use existing)
3. Create a compute instance:
   - Size: **Standard_NC6s_v3** (1x V100 GPU) or **Standard_NC4as_T4_v3** (1x T4 GPU)
   - The T4 is more cost-effective for this workload
4. Open JupyterLab on the compute instance

### 3. Upload and Run Notebook

1. Upload `VQA_Vector_Search_AzureML.ipynb` to your Azure ML workspace
2. Update the configuration in Cell 3:
   ```python
   COSMOS_CONNECTION_STRING = "your-connection-string"
   ```
3. Run cells sequentially

## 📓 Notebook Phases

| Phase | Description | Cells |
|-------|-------------|-------|
| **Phase 0** | Setup & Configuration | 1-4 |
| **Phase 1** | Download VQAv2 Dataset | 5-7 |
| **Phase 2** | Data Preparation | 8-11 |
| **Phase 3** | Cosmos DB Setup | 12-13 |
| **Phase 4** | ViLT Embedding Generation | 14-17 |
| **Phase 5** | Upload to Cosmos DB | 18-20 |
| **Phase 6** | Query System | 21-24 |
| **Phase 7** | Evaluation | 25-28 |
| **Phase 8** | CoAttention Comparison | 29 |
| **Phase 9** | Utilities & Cleanup | 30-31 |

## 🔧 Configuration Options

```python
# In Cell 3 of the notebook

# Dataset settings
USE_VALIDATION_SET = True   # True = smaller (~214k), False = training (~443k)
SAMPLE_FRACTION = 0.33      # Use 1/3 of dataset (faster processing)

# Processing settings
BATCH_SIZE = 32             # Reduce if GPU OOM (16 for 8GB GPU)
CHECKPOINT_EVERY = 1000     # Save progress frequency

# Model settings
EMBEDDING_DIM = 768         # ViLT hidden size (don't change)
MAX_QUESTION_LENGTH = 40    # Question truncation length
```

## 📊 Expected Results

| Metric | Expected Range |
|--------|----------------|
| Overall Accuracy | 35-45% |
| Yes/No Accuracy | 60-70% |
| Count Accuracy | 25-35% |
| Color Accuracy | 45-55% |
| Processing Time | ~2-4 hours (148k samples on T4) |

## 💰 Cost Estimates

| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| Azure ML Compute (T4) | NC4as_T4_v3 (4 hrs) | ~$5-10 |
| Azure ML Compute (V100) | NC6s_v3 (2 hrs) | ~$15-20 |
| Cosmos DB vCore M25 | Development tier | ~$100 |
| Cosmos DB vCore M40 | Production tier | ~$200 |
| Storage (embeddings) | ~5-10 GB | ~$1 |

**Tips to reduce costs:**
- Use T4 GPU instead of V100
- Stop compute when not in use
- Use M25 tier for development
- Process smaller sample fraction first

## 🔍 Usage Examples

### Query System

```python
# After running the notebook, use the query system:

result = query_system.query(
    image_path="path/to/image.jpg",
    question="What color is the car?",
    k=5  # Number of similar examples to find
)

print(f"Answer: {result['predicted_answer']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### Filter by Question Type

```python
result = query_system.query(
    image_path="path/to/image.jpg",
    question="How many people are there?",
    k=5,
    filter_type="count"  # Only search count-type questions
)
```

### Batch Evaluation

```python
eval_results = evaluate_vqa_accuracy(
    query_system, 
    test_samples=samples[:500],
    k=5
)
print(f"Accuracy: {eval_results['overall_accuracy']:.1%}")
```

## 🛠️ Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8
```

### "Connection timed out" (Cosmos DB)
1. Check firewall allows Azure ML IP
2. Enable "Allow Azure services" in Cosmos DB networking
3. Verify connection string is correct

### "No GPU detected"
1. Ensure you created GPU compute instance
2. Check compute is running
3. Verify PyTorch CUDA: `torch.cuda.is_available()`

### "Vector index not found"
The index is created automatically on first vector insert. If you get search errors:
```python
# Recreate index
create_vector_index(cosmos_collection, dimensions=768)
```

### Slow embedding generation
- Use GPU compute (not CPU)
- Increase batch size (if memory allows)
- Use smaller sample fraction for testing

## 📁 Files in This Package

```
vector_db/
├── README.md                         # This file
├── AZURE_COSMOS_SETUP.md             # Cosmos DB setup guide
├── VQA_Vector_Search_AzureML.ipynb   # Main notebook (builds vectors on Azure ML)
├── vilt_embeddings.py                # ViLT encoder (768-dim embeddings)
├── cosmos_vector_db.py               # Cosmos DB operations
├── query_similar.py                  # Query system
├── vqa_rag_endpoint.py               # Flask API endpoints
└── __init__.py                       # Package exports
```

## 🔄 CoAttention Comparison

The notebook includes a ready-to-use comparison framework (Cell 29). To compare with your trained CoAttention model:

1. Export your CoAttention model
2. Uncomment Cell 29
3. Update the import and model loading code
4. Run comparison

```python
# Example comparison output:
# ViLT Vector Search Accuracy: 42.3%
# CoAttention Model Accuracy: 58.7%
```

## 🔗 References

- [ViLT Paper](https://arxiv.org/abs/2102.03334)
- [VQAv2 Dataset](https://visualqa.org/)
- [Azure Cosmos DB Vector Search](https://learn.microsoft.com/azure/cosmos-db/mongodb/vcore/vector-search)
- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
