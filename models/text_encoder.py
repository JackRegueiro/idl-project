import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14") -> None:
        """
        Wrapper for the pre-trained CLIP ViT-L/14 text encoder.
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text prompt into an embedding.
        :param text: The input text prompt.
        :return: A tensor representing the text embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        # Use a simple pooling (e.g., mean over tokens)
        # TODO: VERIFY THIS
        # Not sure this is exactly how the encoding works but it seems reasonable
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze(0)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text prompts into a batch of embeddings.
        :param texts: List of input prompts.
        :return: A tensor of shape [batch_size, embedding_dim].
        """
        embeddings = [self.encode(text) for text in texts]
        return torch.stack(embeddings, dim=0)
