import json
import os
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional

class CoProDataset(Dataset):
    def __init__(self, data_path: str, category: str = "sexual") -> None:
        """
        Dataset class for loading and processing the CoPro dataset.
        
        The CoPro dataset contains pairs of unsafe and safe prompts for various categories.
        As mentioned in the DES paper, we use the sexual category with 6,911 safe-unsafe prompt pairs.
        
        Args:
            data_path: Path to the CoPro dataset JSON file or directory containing the file.
            category: Category to filter ('sexual', 'self-harm', etc.). Default is 'sexual'.
        """
        # Handle both directory and file paths
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "CoPro_v1.0.json")
        
        self.data_path = data_path
        self.category = category
        
        # Load raw data from data_path
        self.raw_data = self._load_raw_data()
        
        # For now, we'll just store the raw data
        # The target vectors will be generated later when TextEncoder is available
        self.data: List[Tuple[Optional[torch.Tensor], str, str]] = [
            (None, item["unsafe_prompt"], item["safe_prompt"]) for item in self.raw_data
        ]
        
        print(f"Loaded {len(self.data)} {category} safe-unsafe prompt pairs from CoPro dataset")

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

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], str, str]:
        """
        Returns a tuple (target_safe_vector, unsafe_prompt, safe_prompt) for the given index.
        
        Note: target_safe_vector will be None until we implement TargetVectorGenerator
        and call update_target_vectors().
        
        Args:
            idx: Index of the data point to retrieve.
            
        Returns:
            Tuple containing (target_safe_vector, unsafe_prompt, safe_prompt).
        """
        target_safe_vector, unsafe_prompt, safe_prompt = self.data[idx]
        return target_safe_vector, unsafe_prompt, safe_prompt
        
    def update_target_vectors(self, target_vectors: List[torch.Tensor]) -> None:
        """
        Update the dataset with pre-computed target vectors.
        
        This method should be called after generating target vectors using the
        TargetVectorGenerator from the safe and unsafe prompts.
        
        Args:
            target_vectors: List of target vectors corresponding to each data point.
            
        Raises:
            ValueError: If the number of target vectors does not match the dataset size.
        """
        if len(target_vectors) != len(self.data):
            raise ValueError(f"Number of target vectors ({len(target_vectors)}) does not match dataset size ({len(self.data)})")
            
        self.data = [(target_vectors[i], self.data[i][1], self.data[i][2]) for i in range(len(self.data))]
        
    def get_all_safe_prompts(self) -> List[str]:
        """
        Returns a list of all safe prompts in the dataset.
        
        Returns:
            List of all safe prompts.
        """
        return [item[2] for item in self.data]
        
    def get_all_unsafe_prompts(self) -> List[str]:
        """
        Returns a list of all unsafe prompts in the dataset.
        
        Returns:
            List of all unsafe prompts.
        """
        return [item[1] for item in self.data]
