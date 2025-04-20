import torch
import torch.nn as nn

# Helper function for Gaussian Kernel MMD
def _gaussian_kernel(x, y, sigma_sq):
    """Gaussian kernel for MMD."""
    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    y_norm = torch.sum(y * y, dim=1, keepdim=True)
    xy = torch.mm(x, y.t())
    dist_sq = x_norm + y_norm.t() - 2 * xy
    # Clamp minimum distance to avoid numerical instability with exp
    dist_sq = torch.clamp(dist_sq, min=1e-12)
    gamma = 1.0 / (2 * sigma_sq)
    kernel_val = torch.exp(-gamma * dist_sq)
    return kernel_val

def _mmd_gaussian(source: torch.Tensor, target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute MMD with Gaussian kernel."""
    sigma_sq = sigma * sigma
    source_kernel = _gaussian_kernel(source, source, sigma_sq)
    target_kernel = _gaussian_kernel(target, target, sigma_sq)
    source_target_kernel = _gaussian_kernel(source, target, sigma_sq)

    mmd_val = source_kernel.mean() + target_kernel.mean() - 2 * source_target_kernel.mean()
    # Ensure non-negative result due to potential floating point inaccuracies
    return torch.clamp(mmd_val, min=0.0)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss.
    Encourages the distribution of current unsafe embeddings to match the
    distribution of original safe embeddings.
    """
    def __init__(self, kernel_sigma: float = 1.0) -> None:
        super(MMDLoss, self).__init__()
        self.sigma = kernel_sigma

    def forward(
        self,
        current_unsafe_embeddings: torch.Tensor, # Shape [B, D]
        original_safe_embeddings: torch.Tensor   # Shape [B, D]
    ) -> torch.Tensor:
        """
        Calculates the MMD between the two sets of embeddings using a Gaussian kernel.

        Args:
            current_unsafe_embeddings: Batch of current unsafe embeddings.
            original_safe_embeddings: Batch of original safe embeddings.

        Returns:
            Scalar MMD loss value.
        """
        # Ensure inputs are on the same device
        device = current_unsafe_embeddings.device
        target_embeddings = original_safe_embeddings.to(device)

        mmd_value = _mmd_gaussian(current_unsafe_embeddings, target_embeddings, self.sigma)
        return mmd_value 