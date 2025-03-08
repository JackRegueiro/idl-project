import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Any, Dict
from data.dataset import CoProDataset
from data.target_vector_generation import TargetVectorGenerator
from models.text_encoder import TextEncoder
from losses.uen_loss import UENLoss

def train(config: Dict[str, Any]) -> None:
    """
    Training loop for the simplified DES model using only the UEN loss.
    
    :param config: Dictionary containing training hyperparameters and paths.
    """
    # See Algorithm 2 of https://arxiv.org/abs/2501.18877
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CoProDataset(data_path=config["data_path"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    text_encoder = TextEncoder().to(device)
    text_encoder.train()

    optimizer = AdamW(
        text_encoder.parameters(),
        lr=config["learning_rate"]
    )

    uen_loss_fn = UENLoss().to(device)

    for epoch in range(config["epochs"]):
        for batch_idx, batch in enumerate(dataloader):
            target_safe_vectors, unsafe_prompts, safe_prompts = batch
            target_safe_vectors = target_safe_vectors.to(device)

            unsafe_embeddings = text_encoder(list(unsafe_prompts)).to(device)

            loss = uen_loss_fn(unsafe_embeddings, target_safe_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(text_encoder.state_dict(), config["model_save_path"])
