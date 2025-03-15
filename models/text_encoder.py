import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Union

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14") -> None:
        """
        Wrapper for the pre-trained CLIP ViT-L/14 text encoder.
        
        Stable Diffusion uses a modified CLIP text encoder to convert 
        text prompts into embeddings. This class provides an interface
        for extracting these embeddings.
        
        Args:
            model_name: The name of the pre-trained CLIP model.
                        For Stable Diffusion v1.4/v1.5, use "openai/clip-vit-large-patch14"
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = True
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"TextEncoder initialized on device: {self.device}")

        self.config = self.model.config
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text prompt into an embedding.
        
        This method follows the CLIP text encoding process used in Stable Diffusion.
        It extracts the last hidden state of the [EOS] token, which serves as
        the text embedding.
        
        Args:
            text: The input text prompt.
        
        Returns:
            A tensor representing the text embedding of shape [embedding_dim].
        """
        # Move inputs to the same device as the model
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        eos_token_idx = inputs['attention_mask'].sum(dim=1) - 1
        batch_size = eos_token_idx.shape[0]
        # Extract the embedding for the [EOS] token for each sample
        embedding = torch.stack([last_hidden_state[i, eos_token_idx[i]] for i in range(batch_size)])
        # Remove the extra batch dimension if batch_size == 1
        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        return embedding

    def forward(self, texts: Union[List[str], str]) -> torch.Tensor:
        """
        Encode text prompts into embeddings.
        
        Args:
            texts: A single text prompt or a list of text prompts.
        
        Returns:
            A tensor of shape [batch_size, embedding_dim].
        """
        if isinstance(texts, str):
            return self.encode(texts).unsqueeze(0)
        
        embeddings = torch.stack([self.encode(text) for text in texts])
        return embeddings
