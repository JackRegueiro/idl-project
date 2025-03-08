# test target vector generation here. 

import os
import sys
import torch
from pathlib import Path

# Add the project root to sys.path so we can import from other directories
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.target_vector_generation import TargetVectorGenerator

def main():
    """Test the TargetVectorGenerator with a small example dataset."""
    print("Testing TargetVectorGenerator implementation...")
    
    # Create synthetic embeddings for testing
    embedding_dim = 5  # Small dimension for easier visualization
    num_samples = 10
    
    # Create random embeddings and normalize them
    safe_embeddings = torch.randn(num_samples, embedding_dim)
    safe_embeddings = safe_embeddings / torch.norm(safe_embeddings, dim=1, keepdim=True)
    
    unsafe_embeddings = torch.randn(num_samples, embedding_dim)
    unsafe_embeddings = unsafe_embeddings / torch.norm(unsafe_embeddings, dim=1, keepdim=True)
    
    # Create a nudity vector that has positive correlation with unsafe and negative with safe
    nudity_vector = 0.7 * unsafe_embeddings[0] - 0.3 * safe_embeddings[0]
    nudity_vector = nudity_vector / torch.norm(nudity_vector)
    
    # Create the generator
    scaling_factor = 2.0  # Small scaling factor for this test
    generator = TargetVectorGenerator(
        nudity_vector=nudity_vector,
        scaling_factor=scaling_factor,
        safe_embeddings=safe_embeddings
    )
    
    # Test single vector generation
    print("\nTesting single vector generation:")
    target_vector, selected_idx = generator.generate_target_vector(unsafe_embeddings[0])
    
    print(f"Selected safe embedding index: {selected_idx}")
    print(f"Original safe embedding: {safe_embeddings[selected_idx]}")
    print(f"Target vector: {target_vector}")
    
    # Verify the calculations
    expected_target = safe_embeddings[selected_idx] - scaling_factor * (nudity_vector / torch.norm(nudity_vector))
    print(f"Expected target vector: {expected_target}")
    
    similarity = torch.sum(target_vector * expected_target) / (torch.norm(target_vector) * torch.norm(expected_target))
    print(f"Similarity between actual and expected target vectors: {similarity.item():.6f} (should be close to 1.0)")
    
    # Test generating all vectors
    print("\nTesting batch vector generation:")
    target_vectors, selected_indices = generator.generate_all_target_vectors(unsafe_embeddings)
    
    print(f"Generated {len(target_vectors)} target vectors")
    print(f"Selected safe embedding indices: {selected_indices}")
    
    # Analyze the results
    generator.analyze_similarity_distribution(target_vectors, unsafe_embeddings)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 