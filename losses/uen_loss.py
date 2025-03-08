import torch
import torch.nn as nn
import torch.nn.functional as F

class UENLoss(nn.Module):
    def __init__(self) -> None:
        """
        Implements the Unsafe Embedding Neutralization (UEN) loss.
        """
        super(UENLoss, self).__init__()

    def forward(
        self, unsafe_embeddings: torch.Tensor, target_safe_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the UEN loss which minimizes cosine similarity between unsafe embeddings
        and their corresponding target safe embeddings.
        
        :param unsafe_embeddings: Tensor of shape [batch_size, embedding_dim] representing the current unsafe embeddings.
        :param target_safe_embeddings: Tensor of shape [batch_size, embedding_dim] representing the target safe embeddings.
        :return: Scalar loss value computed as the mean over the batch.
        """
        # See Equation 3 of https://arxiv.org/abs/2501.18877
        cosine_sim = F.cosine_similarity(unsafe_embeddings, target_safe_embeddings, dim=1)
        loss = torch.mean(1 - cosine_sim)
        return loss
