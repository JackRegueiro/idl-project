import torch
import torch.nn as nn

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
        
        :param unsafe_embeddings: Tensor of shape [batch_size, embedding_dim].
        :param target_safe_embeddings: Tensor of shape [batch_size, embedding_dim].
        :return: Scalar loss value.
        """
        # TODO
        # Implement Equation 3 in the paper
        pass
