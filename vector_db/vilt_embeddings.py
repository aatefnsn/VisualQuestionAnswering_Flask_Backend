"""
ViLT Embeddings Generator for VQA Multimodal Search

Generates embeddings for image-question pairs using ViLT
(Vision-and-Language Transformer) from Hugging Face Transformers.

ViLT is specifically designed for vision-language tasks and is
better suited for VQA than CLIP, which is trained on image-caption pairs.

Reference: https://arxiv.org/abs/2102.03334
"""

import torch
from transformers import ViltProcessor, ViltModel
from PIL import Image
import numpy as np
from typing import Union, List
import os


class ViLTEmbeddings:
    """Generate ViLT embeddings for image-question pairs (VQA-finetuned)."""
    
    def __init__(
        self, 
        model_name: str = "dandelin/vilt-b32-finetuned-vqa",
        device: str = None,
        max_length: int = 40
    ):
        """
        Initialize ViLT model.
        
        Args:
            model_name: ViLT model variant. Default is VQA-finetuned version.
            device: "cuda" or "cpu". Auto-detected if None.
            max_length: Maximum question length (tokens)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        print(f"Loading ViLT model '{model_name}' on {self.device}...")
        
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ViLT hidden size is 768
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"✓ ViLT loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_image_question(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        question: str
    ) -> np.ndarray:
        """
        Encode image and question together to a multimodal embedding.
        
        ViLT processes image and text jointly, unlike CLIP which
        encodes them separately. This is better for VQA tasks.
        
        Args:
            image: File path, PIL Image, or numpy array
            question: Question text
            
        Returns:
            Normalized embedding vector (768-dim)
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Process image and question together
        inputs = self.processor(
            image, 
            question, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token of last hidden state)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.flatten()
    
    def encode_batch(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str]
    ) -> np.ndarray:
        """
        Encode multiple image-question pairs efficiently.
        
        Args:
            images: List of image paths or PIL Images
            questions: List of questions (same length as images)
            
        Returns:
            Array of embeddings (N x 768)
        """
        assert len(images) == len(questions), "Images and questions must have same length"
        
        # Load all images
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        
        # Process batch
        inputs = self.processor(
            pil_images,
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        return embeddings
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2))


# Test the embeddings
if __name__ == "__main__":
    print("Testing ViLT Embeddings...")
    
    vilt_encoder = ViLTEmbeddings(model_name="dandelin/vilt-b32-mlm")
    
    # Test with a dummy image (white image)
    test_image = Image.new("RGB", (224, 224), color="white")
    
    # Test encoding
    q1 = "What color is the background?"
    q2 = "What is in the image?"
    
    e1 = vilt_encoder.encode_image_question(test_image, q1)
    e2 = vilt_encoder.encode_image_question(test_image, q2)
    
    print(f"Embedding shape: {e1.shape}")
    print(f"Embedding norm: {np.linalg.norm(e1):.4f}")
    print(f"Similarity between questions: {vilt_encoder.compute_similarity(e1, e2):.3f}")
    
    print("\n✓ ViLT embeddings working correctly!")
