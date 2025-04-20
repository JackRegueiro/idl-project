import torch
import torch.nn as nn
import torch.nn.functional as F

class SEPLoss(nn.Module):
    def __init__(self, scale_g:float) -> None:
        super(SEPLoss, self).__init__()
        self.scale_g = scale_g

    def forward(
        self,
        current_safe_embeddings: torch.Tensor,
        original_safe_embeddings: torch.Tensor,
        nudity_vector: torch.Tensor
    ) -> torch.Tensor:
        # Basic alignment loss
        cosine_basic = F.cosine_similarity(
            current_safe_embeddings, original_safe_embeddings, dim=1
        )
        loss_basic = torch.mean(1.0 - cosine_basic)

        # PALA-adjusted alignment
        nudity_dir = nudity_vector / (nudity_vector.norm(dim=-1, keepdim=True) + 1e-8)
        adjusted_safe = current_safe_embeddings + self.scale_g * nudity_dir.unsqueeze(0)
        cosine_pala = F.cosine_similarity(
            adjusted_safe, original_safe_embeddings, dim=1
        )
        loss_pala = torch.mean(1.0 - cosine_pala)

        return loss_basic + loss_pala