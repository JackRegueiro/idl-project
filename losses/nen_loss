import torch
import torch.nn as nn
import torch.nn.functional as F

class NENLoss(nn.Module):
    """
    Nudity Embedding Neutralization (NEN).
    Aligns the nudity embedding with the unconditioned embedding,
    neutralizing the nudity concept.
    """
    def __init__(self) -> None:
        super(NENLoss, self).__init__()

    def forward(
        self,
        nudity_embedding: torch.Tensor,
        uncond_embedding: torch.Tensor
    ) -> torch.Tensor:
        # ensure shape [1, D] for cosine_similarity
        n = nudity_embedding.unsqueeze(0)
        u = uncond_embedding.unsqueeze(0)
        cosine_n = F.cosine_similarity(n, u, dim=1)
        return torch.mean(1.0 - cosine_n)
