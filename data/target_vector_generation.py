import torch
import numpy as np
from typing import List, Tuple, Optional

class TargetVectorGenerator:
    def __init__(
            self, 
            nudity_vector: torch.Tensor, 
            scaling_factor: float,
            safe_embeddings: torch.Tensor
        ) -> None:
        """
        Initialize the target vector generator.
        
        Args:
            nudity_vector: A tensor representing the nudity vector of shape [embedding_dim].
            scaling_factor: Scaling factor (sg) for nudity subtraction as described in the paper.
            safe_embeddings: A tensor of shape [num_safe, embedding_dim] containing safe embeddings.
        """
        self.nudity_vector = nudity_vector
        self.scaling_factor = scaling_factor
        self.safe_embeddings = safe_embeddings
        
        # Normalize nudity vector for easier scaling
        self.normalized_nudity = self.nudity_vector / torch.norm(self.nudity_vector)
        
        # Precompute safe embeddings normalization for faster similarity calculation
        self.safe_norm = self.safe_embeddings / torch.norm(self.safe_embeddings, dim=1, keepdim=True)
        
        print(f"TargetVectorGenerator initialized with {len(safe_embeddings)} safe embeddings")
        print(f"Nudity vector shape: {nudity_vector.shape}")
        print(f"Scaling factor: {scaling_factor}")

    def generate_target_vector(self, unsafe_embedding: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        For a given unsafe embedding, search among safe_embeddings to select the safe vector
        with the lowest cosine similarity and then subtract the nudity direction.
        
        Args:
            unsafe_embedding: Tensor of shape [embedding_dim] representing an unsafe embedding.
        
        Returns:
            Tuple of (target_safe_embedding, selected_index) where:
                - target_safe_embedding: Tensor of shape [embedding_dim] representing the target vector
                - selected_index: Index of the selected safe embedding
        """
        # Normalize the unsafe embedding for cosine similarity calculation
        unsafe_norm = unsafe_embedding / torch.norm(unsafe_embedding)
        
        # Calculate cosine similarity between unsafe_embedding and all safe_embeddings
        # Implementation of Equation (1) in the paper
        similarities = torch.matmul(self.safe_norm, unsafe_norm)
        
        # Find the safe embedding with minimum similarity
        min_similarity_idx = torch.argmin(similarities).item()
        selected_safe_vector = self.safe_embeddings[min_similarity_idx]
        
        # Subtract the nudity direction to create target vector
        # Implementation of Equation (2) in the paper
        target_vector = selected_safe_vector - self.scaling_factor * self.normalized_nudity
        
        return target_vector, min_similarity_idx
        
    def generate_all_target_vectors(self, unsafe_embeddings: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate target vectors for all unsafe embeddings.
        
        Args:
            unsafe_embeddings: Tensor of shape [num_unsafe, embedding_dim] containing unsafe embeddings.
            
        Returns:
            Tuple of (target_vectors, selected_indices) where:
                - target_vectors: Tensor of shape [num_unsafe, embedding_dim] containing target vectors
                - selected_indices: List of indices of selected safe embeddings
        """
        target_vectors = []
        selected_indices = []
        
        for i in range(len(unsafe_embeddings)):
            target_vector, selected_idx = self.generate_target_vector(unsafe_embeddings[i])
            target_vectors.append(target_vector)
            selected_indices.append(selected_idx)
            
            # Log progress
            if (i + 1) % 500 == 0 or i == 0 or i == len(unsafe_embeddings) - 1:
                print(f"Generated target vector for unsafe embedding {i+1}/{len(unsafe_embeddings)}")
                
        return torch.stack(target_vectors), selected_indices

    def analyze_similarity_distribution(self, target_vectors: torch.Tensor, unsafe_embeddings: torch.Tensor) -> None:
        """
        Analyze the cosine similarity distribution between vectors to evaluate the effectiveness
        of the target vector generation.
        
        Args:
            target_vectors: Tensor of shape [num_vectors, embedding_dim] containing target vectors.
            unsafe_embeddings: Tensor of shape [num_vectors, embedding_dim] containing unsafe embeddings.
        """
        # Normalize vectors for cosine similarity
        target_norm = target_vectors / torch.norm(target_vectors, dim=1, keepdim=True)
        unsafe_norm = unsafe_embeddings / torch.norm(unsafe_embeddings, dim=1, keepdim=True)
        nudity_norm = self.normalized_nudity
        
        # Compute cosine similarities
        target_unsafe_sim = torch.sum(target_norm * unsafe_norm, dim=1)
        target_nudity_sim = torch.matmul(target_norm, nudity_norm)
        unsafe_nudity_sim = torch.matmul(unsafe_norm, nudity_norm)
        
        # Calculate statistics
        print("\nSimilarity Analysis:")
        print(f"Target-Unsafe similarity (mean): {target_unsafe_sim.mean().item():.4f}")
        print(f"Target-Unsafe similarity (min): {target_unsafe_sim.min().item():.4f}")
        print(f"Target-Unsafe similarity (max): {target_unsafe_sim.max().item():.4f}")
        print(f"Target-Nudity similarity (mean): {target_nudity_sim.mean().item():.4f}")
        print(f"Unsafe-Nudity similarity (mean): {unsafe_nudity_sim.mean().item():.4f}")
        
        # Count negative similarities with nudity (indicating direction away from nudity)
        negative_nudity_count = torch.sum(target_nudity_sim < 0).item()
        print(f"Target vectors with negative similarity to nudity: {negative_nudity_count}/{len(target_vectors)} ({negative_nudity_count/len(target_vectors)*100:.2f}%)")
