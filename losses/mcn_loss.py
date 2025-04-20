import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class MultiConceptNENLoss(nn.Module):
    """
    Multi-Concept Nudity Embedding Neutralization (MCN).
    Generalizes NEN to neutralize multiple harmful concepts by aligning
    their embeddings with the unconditioned embedding.
    """
    def __init__(self) -> None:
        super(MultiConceptNENLoss, self).__init__()

    def forward(
        self,
        current_harmful_embeddings: Dict[str, torch.Tensor], # Embeddings keyed by concept name
        uncond_embedding: torch.Tensor,
        concept_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Calculates the weighted sum of cosine dissimilarities between
        harmful concept embeddings and the unconditioned embedding.

        Args:
            current_harmful_embeddings: Dict where keys are concept names (str)
                                        and values are their current embeddings (Tensor [D]).
            uncond_embedding: The fixed unconditioned embedding (Tensor [D]).
            concept_weights: Optional dict mapping concept names to weights (float).
                             Defaults to uniform weight 1.0 if None.

        Returns:
            Scalar loss value.
        """
        if not current_harmful_embeddings:
            return torch.tensor(0.0, device=uncond_embedding.device)

        total_mcn_loss = torch.tensor(0.0, device=uncond_embedding.device)
        num_concepts = len(current_harmful_embeddings)
        u = uncond_embedding.unsqueeze(0) # Shape [1, D]

        for concept, harm_emb in current_harmful_embeddings.items():
            weight = 1.0
            if concept_weights and concept in concept_weights:
                weight = concept_weights[concept]

            h = harm_emb.unsqueeze(0) # Shape [1, D]
            cosine_sim = F.cosine_similarity(h, u, dim=1)
            loss_concept = 1.0 - cosine_sim
            total_mcn_loss += weight * loss_concept

        # Return mean loss if weights aren't specified, otherwise weighted sum
        if concept_weights is None:
             return total_mcn_loss / num_concepts
        else:
             # The user might want sum or mean of weighted losses, sum is simpler here
             return total_mcn_loss.squeeze() # Squeeze potential [1] tensor 