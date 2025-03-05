import torch

class TargetVectorGenerator:
    def __init__(
            self, 
            nudity_vector: torch.Tensor, 
            scaling_factor: float,
            safe_embeddings: torch.Tensor
        ) -> None:
        """
        :param scaling_factor: Scaling factor for nudity subtraction.
        :param nudity_vector: A tensor representing the nudity vector of shape [embedding_dim].
        :param safe_embeddings: A tensor of shape [num_safe, embedding_dim] containing safe embeddings.
        """
        # TODO
        # Initialize params
        pass

    def generate_target_vector(self, unsafe_embedding: torch.Tensor) -> torch.Tensor:
        """
        For a given unsafe embedding, search among safe_embeddings to select the safe vector
        with the lowest cosine similarity and then subtract the nudity direction.
        
        :param unsafe_embedding: Tensor of shape [embedding_dim].
        :return: Target safe embedding tensor.
        """
        # TODO
        # Implement equation (1) and (2) in the paper
        pass
