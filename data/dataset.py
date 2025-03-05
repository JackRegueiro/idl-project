import torch
from torch.utils.data import Dataset
from typing import Tuple

class CoProDataset(Dataset):
    def __init__(self, data_path: str) -> None: # TODO: Add any necessary params
        """
        Dataset class for loading and processing the CoPro dataset.
        :param data_path: Path to the CoPro dataset.
        :param transform: Optional data transformation to apply.
        """
        self.data_path = data_path
        # TODO
        # 1. Load raw data from data_path
        #   self.raw_data = ... <-- List[Tuple[str, str]] -- [(unsafe prompt, safe prompt)]
        # 2. Pre-process data
        #   Make use of the target_vector_generation.py
        #   self.data = ... <-- List[Tuple[torch.Tensor, str, str]] -- [(target safe vector, unsafe prompt, safe prompt)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Returns a tuple (target safe vector, unsafe prompt, safe prompt) for the given index.
        """
        # TODO
        pass
