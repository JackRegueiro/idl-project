import os
import sys
import argparse
import torch
from pathlib import Path

# Add the project root to sys.path so we can import from other directories
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.target_vector_generation import TargetVectorGenerator
from data.dataset import CoProDataset

def main():
    parser = argparse.ArgumentParser(description="Generate target vectors for the DES algorithm")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings",
                       help="Directory containing the saved embeddings")
    parser.add_argument("--scaling_factor", type=float, default=200.0,
                       help="Scaling factor (sg) for nudity subtraction (default: 200.0, as mentioned in the paper)")
    parser.add_argument("--output_dir", type=str, default="./target_vectors",
                       help="Directory to save the target vectors")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to the CoPro dataset (required to update the dataset with target vectors)")
    parser.add_argument("--category", type=str, default="sexual",
                       help="Category to filter (default: sexual)")
    parser.add_argument("--update_dataset", action="store_true",
                       help="Update the dataset with the generated target vectors")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.abspath(os.path.join(script_dir, args.embeddings_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_dir}...")
    
    safe_embeddings_path = os.path.join(embeddings_dir, "safe_embeddings.pt")
    unsafe_embeddings_path = os.path.join(embeddings_dir, "unsafe_embeddings.pt")
    nudity_embedding_path = os.path.join(embeddings_dir, "nudity_embedding.pt")
    
    if not all(os.path.exists(p) for p in [safe_embeddings_path, unsafe_embeddings_path, nudity_embedding_path]):
        print("Error: One or more embedding files not found. Make sure to run load_clip_encoder.py first.")
        sys.exit(1)
        
    safe_embeddings = torch.load(safe_embeddings_path)
    unsafe_embeddings = torch.load(unsafe_embeddings_path)
    nudity_embedding = torch.load(nudity_embedding_path)
    
    print(f"Loaded safe embeddings: {safe_embeddings.shape}")
    print(f"Loaded unsafe embeddings: {unsafe_embeddings.shape}")
    print(f"Loaded nudity embedding: {nudity_embedding.shape}")
    
    # Initialize target vector generator
    print(f"\nInitializing TargetVectorGenerator with scaling factor = {args.scaling_factor}...")
    target_generator = TargetVectorGenerator(
        nudity_vector=nudity_embedding,
        scaling_factor=args.scaling_factor,
        safe_embeddings=safe_embeddings
    )
    
    # Generate target vectors
    print("\nGenerating target vectors...")
    target_vectors, selected_indices = target_generator.generate_all_target_vectors(unsafe_embeddings)
    
    # Analyze similarity distribution
    target_generator.analyze_similarity_distribution(target_vectors, unsafe_embeddings)
    
    # Save target vectors
    target_vectors_path = os.path.join(output_dir, "target_vectors.pt")
    selected_indices_path = os.path.join(output_dir, "selected_indices.pt")
    
    print(f"\nSaving target vectors to {target_vectors_path}...")
    torch.save(target_vectors, target_vectors_path)
    torch.save(selected_indices, selected_indices_path)
    
    # Print some examples of the selected safe prompts
    if args.update_dataset and args.data_path:
        # Find the CoPro dataset
        data_path = args.data_path
        if not os.path.exists(data_path):
            possible_paths = [
                os.path.join(script_dir, data_path),
                os.path.join(project_root, data_path),
                os.path.join(os.path.dirname(project_root), data_path)
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
                    
        if not os.path.exists(data_path):
            print(f"Warning: Could not find CoPro dataset at {data_path}. Skipping dataset update.")
        else:
            print(f"\nUpdating CoProDataset with target vectors...")
            dataset = CoProDataset(data_path, args.category)
            dataset.update_target_vectors(target_vectors.tolist())
            
            # Print examples
            print("\nExample Target Vector Pairs:")
            for i in range(min(5, len(selected_indices))):
                unsafe_idx = i
                safe_idx = selected_indices[i]
                target_vector = target_vectors[i]
                
                # Get prompt pairs
                _, unsafe_prompt, _ = dataset[unsafe_idx]
                _, _, safe_prompt = dataset[safe_idx]
                
                print(f"\nExample {i+1}:")
                print(f"Unsafe Prompt: {unsafe_prompt}")
                print(f"Selected Safe Prompt: {safe_prompt}")
                print(f"Target Vector Norm: {torch.norm(target_vector).item():.4f}")
                
    print("\nTarget vector generation completed successfully!")

if __name__ == "__main__":
    main() 