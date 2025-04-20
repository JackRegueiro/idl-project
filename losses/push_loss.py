import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class PushAwayLoss(nn.Module):
    """
    Push Away Loss (L_push).
    Encourages current unsafe embeddings to move away from their original positions
    and optionally away from known harmful concept centers.
    """
    def __init__(self) -> None:
        super(PushAwayLoss, self).__init__()

    def forward(
        self,
        current_unsafe_embeddings: torch.Tensor,    # Shape [B, D]
        original_unsafe_embeddings: torch.Tensor,   # Shape [B, D]
        harmful_concept_embeddings: Optional[List[torch.Tensor]] = None # List of Tensors [D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates cosine similarity between current and original unsafe embeddings (to be minimized),
        and optionally between current unsafe embeddings and harmful concepts (to be minimized).

        Args:
            current_unsafe_embeddings: Batch of current unsafe embeddings.
            original_unsafe_embeddings: Batch of corresponding original unsafe embeddings.
            harmful_concept_embeddings: Optional list of fixed harmful concept embeddings.

        Returns:
            Tuple containing:
                - loss_push_orig: Mean cosine similarity with original positions.
                - loss_push_harm: Mean cosine similarity with harmful concepts (0.0 if none provided).
        """
        # L_push_orig: Minimize similarity with original position
        # Cosine similarity returns values between -1 and 1. Minimizing means pushing towards -1.
        # The paper says "minimized", which implies we want cos -> -1.
        # However, often in practice, "minimizing similarity" means maximizing distance,
        # which might mean minimizing (1 + cos)/2 or maximizing (1 - cos)/2.
        # Let's return the raw cosine similarity, and the user can decide how to use it (minimize it directly).
        loss_push_orig = F.cosine_similarity(
            current_unsafe_embeddings, original_unsafe_embeddings, dim=1
        )
        mean_loss_push_orig = torch.mean(loss_push_orig)

        # L_push_harm: Minimize similarity with harmful concepts
        mean_loss_push_harm = torch.tensor(0.0, device=current_unsafe_embeddings.device)
        if harmful_concept_embeddings:
            num_harm_concepts = len(harmful_concept_embeddings)
            batch_size = current_unsafe_embeddings.shape[0]
            total_harm_sim = torch.tensor(0.0, device=current_unsafe_embeddings.device)

            for harm_emb in harmful_concept_embeddings:
                h_emb = harm_emb.to(current_unsafe_embeddings.device).unsqueeze(0) # Shape [1, D]
                cosine_sim = F.cosine_similarity(current_unsafe_embeddings, h_emb, dim=1) # Shape [B]
                total_harm_sim += torch.sum(cosine_sim) # Sum over batch

            mean_loss_push_harm = total_harm_sim / (batch_size * num_harm_concepts) # Average

        return mean_loss_push_orig, mean_loss_push_harm 