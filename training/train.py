import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Any, Dict
from data.dataset import CoProDataset
from models.text_encoder import TextEncoder
from losses.uen_loss import UENLoss
from losses.sep_loss import SEPLoss
from losses.nen_loss import NENLoss
from tqdm import tqdm

def train(config: Dict[str, Any]) -> None:
    """
    Training loop for the simplified DES model using only the UEN loss.
    :param config: Dictionary containing training hyperparameters and paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize text encoder
    text_encoder = TextEncoder().to(device)
    text_encoder.train()

    # Prepare dataset and dataloader
    dataset = CoProDataset(
        data_path=config["data_path"],
        original_encoder=text_encoder  # if CoProDataset precomputes via frozen encoder
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    optimizer = AdamW(text_encoder.parameters(), lr=config["learning_rate"])

    # Loss functions
    uen_loss_fn = UENLoss().to(device)
    sep_loss_fn = SEPLoss(scale_g=config.get("scale_g", 200.0)).to(device)
    nen_loss_fn = NENLoss().to(device)

    lambda_w = config.get("lambda", 0.3)

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # unpack
            target_safe_vecs, unsafe_prompts, safe_prompts, original_safe_vecs = batch
            target_safe_vecs = target_safe_vecs.to(device)
            original_safe_vecs = original_safe_vecs.to(device)

            # current embeddings
            unsafe_emb = text_encoder(list(unsafe_prompts))        # [B, D]
            safe_emb = text_encoder(list(safe_prompts))            # [B, D]

            # nudity and unconditioned embeddings
            nudity_emb = text_encoder(["nudity"])[0]             # [D]
            uncond_emb = text_encoder([""])[0]                   # [D]

            # compute each loss
            lu = uen_loss_fn(unsafe_emb, target_safe_vecs)
            ls = sep_loss_fn(safe_emb, original_safe_vecs, nudity_emb)
            ln = nen_loss_fn(nudity_emb, uncond_emb)

            # total loss: Eq.7
            loss = lambda_w * ls + (1.0 - lambda_w) * (lu + ln)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']} - Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(text_encoder.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")