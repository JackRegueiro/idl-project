import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginSEPLoss(nn.Module):
    """
    Margin-based Safe Embedding Preservation (SEP) Loss.
    Only penalizes if cosine similarity drops below a margin.
    """
    def __init__(self, margin_s: float, scale_g: float) -> None:
        super(MarginSEPLoss, self).__init__()
        self.margin_s = margin_s
        self.scale_g = scale_g

    def forward(
        self,
        current_safe_embeddings: torch.Tensor,
        original_safe_embeddings: torch.Tensor,
        original_nudity_vector: torch.Tensor # Use original for consistency w/ paper Eq.4
    ) -> torch.Tensor:
        """
        Calculates SEP loss using a margin.

        Args:
            current_safe_embeddings: Batch of current safe embeddings [B, D].
            original_safe_embeddings: Batch of original safe embeddings [B, D].
            original_nudity_vector: The original nudity vector [D].

        Returns:
            Scalar loss value.
        """
        # Basic alignment loss with margin
        cosine_basic = F.cosine_similarity(
            current_safe_embeddings, original_safe_embeddings, dim=1
        )
        # Loss = max(0, margin - similarity) -> minimize this pushes similarity >= margin
        loss_basic = torch.mean(F.relu(self.margin_s - cosine_basic))

        # PALA-adjusted alignment with margin
        # Note: Paper uses current nudity embedding (ẽ_n) in Eq 4 for PALA adjustment (˜e′s,i = ˜es,i + sg * ẽn / ||ẽn||)
        # However, the original SEP loss implementation used the *original* nudity vector.
        # Let's stick to the original implementation's choice for minimal change from previous state.
        nudity_dir = original_nudity_vector / (original_nudity_vector.norm(dim=-1, keepdim=True) + 1e-8)
        # Ensure nudity_dir is on the correct device
        nudity_dir = nudity_dir.to(current_safe_embeddings.device)
        adjusted_safe = current_safe_embeddings + self.scale_g * nudity_dir.unsqueeze(0) # Use original nudity vector

        cosine_pala = F.cosine_similarity(
            adjusted_safe, original_safe_embeddings, dim=1
        )
        loss_pala = torch.mean(F.relu(self.margin_s - cosine_pala))

        return loss_basic + loss_pala 