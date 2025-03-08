import os
import torch
import sys
import argparse
from pathlib import Path

# Add the project root to sys.path so we can import from other directories
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.text_encoder import TextEncoder
from data.dataset import CoProDataset

def find_copro_dataset():
    """
    Try various paths to locate the CoPro dataset.
    
    Returns:
        The path to the CoPro dataset directory or None if not found.
    """
    # List of possible relative paths to try
    possible_paths = [
        "CoPro Dataset",
        "../CoPro Dataset",
        "../../CoPro Dataset",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CoPro Dataset"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "CoPro Dataset"),
    ]
    
    # Try each path
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"Checking for CoPro dataset at: {abs_path}")
        
        # Check for directory
        if os.path.isdir(abs_path):
            print(f"Found CoPro dataset directory at: {abs_path}")
            return abs_path
            
        # Check for JSON file directly
        json_path = os.path.join(os.path.dirname(abs_path), "CoPro_v1.0.json")
        if os.path.isfile(json_path):
            print(f"Found CoPro dataset file at: {json_path}")
            return os.path.dirname(json_path)
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Load CLIP encoder and extract embeddings for CoPro dataset")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to the CoPro dataset directory or JSON file (relative to this script)")
    parser.add_argument("--category", type=str, default="sexual", 
                        help="Category to filter ('sexual', 'self-harm', etc.)")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", 
                        help="CLIP model name to use")
    parser.add_argument("--save_dir", type=str, default="./embeddings", 
                        help="Directory to save embeddings")
    parser.add_argument("--nudity_prompt", type=str, default="nudity", 
                        help="Prompt to use for generating the nudity embedding")
    parser.add_argument("--unconditioned_prompt", type=str, default="", 
                        help="Empty prompt for unconditioned embedding")
    args = parser.parse_args()

    # Use the provided data path or try to find the dataset
    data_path = args.data_path
    if data_path is None:
        data_path = find_copro_dataset()
        if data_path is None:
            print("Error: Could not find CoPro dataset. Please provide a valid path using --data_path.")
            sys.exit(1)
    else:
        # Convert relative path to absolute path based on this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(script_dir, args.data_path))
        print(f"Using specified CoPro dataset path: {data_path}")

    # Convert relative save_dir path to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Load the dataset
        print(f"Loading CoPro dataset from: {data_path}")
        dataset = CoProDataset(data_path, args.category)
        
        # Load the CLIP text encoder
        print(f"Loading CLIP text encoder: {args.model_name}...")
        text_encoder = TextEncoder(args.model_name)
        text_encoder.eval()  # Set to evaluation mode
        
        # Get all prompts
        safe_prompts = dataset.get_all_safe_prompts()
        unsafe_prompts = dataset.get_all_unsafe_prompts()
        
        print(f"Extracting embeddings for {len(safe_prompts)} safe prompts...")
        print(f"Extracting embeddings for {len(unsafe_prompts)} unsafe prompts...")
        
        # Extract embeddings in batches to avoid OOM
        batch_size = 64
        
        # Process safe prompts
        safe_embeddings = []
        for i in range(0, len(safe_prompts), batch_size):
            batch = safe_prompts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = text_encoder(batch)
            safe_embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i+len(batch)}/{len(safe_prompts)} safe prompts")
                
        safe_embeddings = torch.cat(safe_embeddings, dim=0)
        print(f"Safe embeddings shape: {safe_embeddings.shape}")
        
        # Process unsafe prompts
        unsafe_embeddings = []
        for i in range(0, len(unsafe_prompts), batch_size):
            batch = unsafe_prompts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = text_encoder(batch)
            unsafe_embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i+len(batch)}/{len(unsafe_prompts)} unsafe prompts")
                
        unsafe_embeddings = torch.cat(unsafe_embeddings, dim=0)
        print(f"Unsafe embeddings shape: {unsafe_embeddings.shape}")
        
        # Generate nudity embedding
        print(f"Generating nudity embedding using prompt: '{args.nudity_prompt}'")
        with torch.no_grad():
            nudity_embedding = text_encoder.encode(args.nudity_prompt)
        print(f"Nudity embedding shape: {nudity_embedding.shape}")
        
        # Generate unconditioned embedding (empty prompt)
        print(f"Generating unconditioned embedding using empty prompt")
        with torch.no_grad():
            unconditioned_embedding = text_encoder.encode(args.unconditioned_prompt)
        print(f"Unconditioned embedding shape: {unconditioned_embedding.shape}")
        
        # Save embeddings
        print(f"Saving embeddings to {save_dir}...")
        torch.save(safe_embeddings, os.path.join(save_dir, "safe_embeddings.pt"))
        torch.save(unsafe_embeddings, os.path.join(save_dir, "unsafe_embeddings.pt"))
        torch.save(nudity_embedding, os.path.join(save_dir, "nudity_embedding.pt"))
        torch.save(unconditioned_embedding, os.path.join(save_dir, "unconditioned_embedding.pt"))
        
        print("Done!")
        
        # Print some stats and analysis
        with torch.no_grad():
            # Compute average cosine similarity between safe and unsafe embeddings
            safe_norm = safe_embeddings / safe_embeddings.norm(dim=1, keepdim=True)
            unsafe_norm = unsafe_embeddings / unsafe_embeddings.norm(dim=1, keepdim=True)
            nudity_norm = nudity_embedding / nudity_embedding.norm()
            uncond_norm = unconditioned_embedding / unconditioned_embedding.norm()
            
            # Average similarity between safe prompts and nudity
            safe_nudity_sim = (safe_norm @ nudity_norm).mean().item()
            
            # Average similarity between unsafe prompts and nudity
            unsafe_nudity_sim = (unsafe_norm @ nudity_norm).mean().item()
            
            # Average similarity between safe and unsafe prompts
            # This uses the diagonal of the similarity matrix, which are pairs from the dataset
            safe_unsafe_sim = torch.sum(safe_norm * unsafe_norm, dim=1).mean().item()
            
            print("\nEmbedding Analysis:")
            print(f"Average similarity between safe prompts and nudity: {safe_nudity_sim:.4f}")
            print(f"Average similarity between unsafe prompts and nudity: {unsafe_nudity_sim:.4f}")
            print(f"Average similarity between safe and unsafe prompt pairs: {safe_unsafe_sim:.4f}")
            print(f"Similarity between nudity and unconditioned: {(nudity_norm @ uncond_norm).item():.4f}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CoPro dataset is available at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 