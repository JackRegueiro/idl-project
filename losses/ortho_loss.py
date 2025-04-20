import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Loss (L_ortho).
    Penalizes alignment between current unsafe embeddings and predefined
    general harmful directions.
    """
    def __init__(self) -> None:
        super(OrthogonalityLoss, self).__init__()

    def forward(
        self,
        current_unsafe_embeddings: torch.Tensor, # Shape [B, D]
        harm_directions: List[torch.Tensor]      # List of Tensors [D]
    ) -> torch.Tensor:
        """
        Calculates the sum of absolute cosine similarities between unsafe embeddings
        and harmful directions.

        Args:
            current_unsafe_embeddings: Batch of current unsafe embeddings.
            harm_directions: List of pre-defined harmful direction vectors.

        Returns:
            Scalar loss value (mean over batch and directions).
        """
        if not harm_directions:
            return torch.tensor(0.0, device=current_unsafe_embeddings.device)

        total_ortho_loss = torch.tensor(0.0, device=current_unsafe_embeddings.device)
        num_directions = len(harm_directions)
        batch_size = current_unsafe_embeddings.shape[0]

        for harm_dir in harm_directions:
            # Ensure harm_dir is on the correct device and shape [1, D] for broadcasting
            h_dir = harm_dir.to(current_unsafe_embeddings.device).unsqueeze(0)
            # Cosine similarity between batch [B, D] and direction [1, D] -> [B]
            cosine_sim = F.cosine_similarity(current_unsafe_embeddings, h_dir, dim=1)
            total_ortho_loss += torch.sum(torch.abs(cosine_sim)) # Sum over batch

        # Average over all batches and all directions
        return total_ortho_loss / (batch_size * num_directions) 