import os
import json
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any

from models.text_encoder import TextEncoder

class CoProDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        text_encoder,
        nudity_prompt = "nudity",
        scaling_factor = 200.0,
        category = "sexual"
    ) -> None:
        """
        Dataset class for loading and processing the CoPro dataset.
        
        It loads the raw data, filters by category, computes embeddings for safe and unsafe prompts,
        generates the target safe vectors, and stores the results.
        
        Args:
            data_path: Path to the CoPro dataset JSON file or directory containing the file.
            text_encoder: An *original* pre-trained text encoder instance used for pre-calculating fixed embeddings.
            nudity_prompt: The prompt to compute the nudity vector.
            scaling_factor: Scaling factor (sg) for nudity subtraction.
            category: Category to filter (default "sexual").
        """ 
        self.data_path = data_path
        self.category = category
        self.scaling_factor = scaling_factor
        self.nudity_prompt = nudity_prompt
        
        # Load raw data
        self.raw_data = self._load_raw_data()
        if not self.raw_data:
            raise ValueError(f"No data found for category '{self.category}' in {self.data_path}")
        
        # Extract safe and unsafe prompts from the raw data
        self.safe_prompts: List[str] = [item["safe_prompt"] for item in self.raw_data]
        self.unsafe_prompts: List[str] = [item["unsafe_prompt"] for item in self.raw_data]
        
        print(f"Loaded {len(self.raw_data)} '{self.category}' prompt pairs from CoPro dataset.")
        
        # Compute embeddings using the provided *original* text encoder.
        # Ensure the passed encoder is in eval mode and doesn't track gradients for precomputation
        original_encoder = text_encoder
        original_encoder.eval() # Ensure it's in eval mode
        device = next(original_encoder.parameters()).device # Get device from encoder

        with torch.no_grad():
            # Use the passed original_encoder directly
            self.original_safe_embeddings = original_encoder(self.safe_prompts).cpu()  # [N, D] - Store on CPU
            self.original_unsafe_embeddings = original_encoder(self.unsafe_prompts).cpu() # [N, D] - Store on CPU (NEW)
            self.original_nudity_vector = original_encoder.encode(self.nudity_prompt).cpu()  # [D] - Store on CPU

        # Normalize the original nudity vector
        self.normalized_original_nudity = self.original_nudity_vector / torch.norm(self.original_nudity_vector)
        
        # Precompute normalized original safe embeddings for cosine similarity
        self.original_safe_norm = self.original_safe_embeddings / torch.norm(self.original_safe_embeddings, dim=1, keepdim=True)
        
        # Generate target safe vectors for each unsafe prompt
        self.data: List[Tuple[torch.Tensor, str, str, torch.Tensor, torch.Tensor]] = [] # Added original_unsafe_embedding
        for i in range(len(self.unsafe_prompts)):
            unsafe_emb = self.original_unsafe_embeddings[i]  # Use the stored original unsafe embedding
            # Normalize unsafe embedding
            unsafe_norm = unsafe_emb / torch.norm(unsafe_emb)
            # Compute cosine similarities with all original safe embeddings
            similarities = torch.matmul(self.original_safe_norm, unsafe_norm)
            # Select original safe embedding with minimum cosine similarity
            min_idx = torch.argmin(similarities).item()
            selected_safe_emb = self.original_safe_embeddings[min_idx]
            # Create target vector: subtract scaled original nudity direction from the selected original safe embedding
            target_vector = selected_safe_emb - self.scaling_factor * self.normalized_original_nudity
            # Store the tuple: (target safe vector, unsafe prompt, safe prompt, original safe embedding, original unsafe embedding)
            self.data.append((target_vector, self.unsafe_prompts[i], self.safe_prompts[i], self.original_safe_embeddings[i], self.original_unsafe_embeddings[i]))
            
        print("Target safe vectors generated for all prompt pairs.")
        # Clear potentially large tensor no longer needed after init
        # del original_unsafe_embeddings # Keep this one now
        del self.original_safe_norm

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load and filter the CoPro dataset based on the specified category.
        
        Returns:
            List of dictionaries containing unsafe_prompt, safe_prompt, concept, and category.
        
        Raises:
            FileNotFoundError: If the CoPro dataset file cannot be found.
            json.JSONDecodeError: If the CoPro dataset file is not valid JSON.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CoPro dataset file not found at: {self.data_path}")
            
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
            # Extract training data
            train_data = data.get("ID_train_data", [])
            
            # Filter by category
            filtered_data = [item for item in train_data if item.get("category") == self.category]
            
            if not filtered_data:
                print(f"Warning: No data found for category '{self.category}' in CoPro dataset")
            
            return filtered_data
            
        except json.JSONDecodeError as e:
            print(f"Error decoding CoPro dataset JSON: {e}")
            raise
        except Exception as e:
            print(f"Error loading CoPro dataset: {e}")
            raise

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple (target_safe_vector, unsafe_prompt, safe_prompt, original_safe_embedding, original_unsafe_embedding)
        for the given index.
        """
        return self.data[idx]
