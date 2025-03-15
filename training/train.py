import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Any, Dict
from data.dataset import CoProDataset
from models.text_encoder import TextEncoder
from losses.uen_loss import UENLoss
from tqdm import tqdm

def train(config: Dict[str, Any]) -> None:
    """
    Training loop for the simplified DES model using only the UEN loss.
    
    :param config: Dictionary containing training hyperparameters and paths.
    """
    # See Algorithm 2 of https://arxiv.org/abs/2501.18877
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    text_encoder = TextEncoder().to(device)
    text_encoder.train()

    dataset = CoProDataset(
        data_path=config["data_path"],
        text_encoder=text_encoder
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    optimizer = AdamW(
        text_encoder.parameters(),
        lr=config["learning_rate"]
    )

    uen_loss_fn = UENLoss().to(device)

    for epoch in range(config["epochs"]):
        print(f"\nStarting Epoch {epoch + 1}/{config['epochs']}")
        epoch_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            target_safe_vectors, unsafe_prompts, safe_prompts = batch
            target_safe_vectors = target_safe_vectors.to(device)

            unsafe_embeddings = text_encoder(list(unsafe_prompts)).to(device)

            loss = uen_loss_fn(unsafe_embeddings, target_safe_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    torch.save(text_encoder.state_dict(), config["model_save_path"])
    print(f"Training completed. Model saved to {config['model_save_path']}")
