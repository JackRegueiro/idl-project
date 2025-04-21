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
    Training loop for the DES model using UEN, SEP, and NEN losses.

    :param config: Dictionary containing training hyperparameters and paths.
    """
    # See Algorithm 2 of https://arxiv.org/abs/2501.18877
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Instantiate the text encoder to be trained
    text_encoder = TextEncoder().to(device)
    text_encoder.train() # Set the encoder to training mode

    # Instantiate the *original* text encoder for precomputation (if needed here)
    # Or ensure the one passed to Dataset is the original one.
    # For NEN loss, we need the original unconditioned embedding.
    print("Calculating original unconditioned embedding...")
    with torch.no_grad():
        original_encoder_temp = TextEncoder().to(device)
        original_encoder_temp.eval()
        original_uncond_embedding = original_encoder_temp.encode(
            config.get("uncond_prompt", "") # Use empty string if not provided
        ).detach().to(device)
        # We also need the original encoder to pass to the Dataset for its precomputations
        dataset_original_encoder = original_encoder_temp # Use the same instance
    print("Original unconditioned embedding calculated.")


    dataset = CoProDataset(
        data_path=config["data_path"],
        text_encoder=dataset_original_encoder, # Pass the original encoder
        nudity_prompt=config.get("nudity_prompt", "nudity"),
        scaling_factor=config["scaling_factor"]
        # category is default "sexual"
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    # Original encoder no longer needed directly in train function after dataset init
    del original_encoder_temp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    optimizer = AdamW(
        text_encoder.parameters(),
        lr=config["learning_rate"]
    )

    # Instantiate loss functions
    uen_loss_fn = UENLoss().to(device)
    sep_loss_fn = SEPLoss(scale_g=config["scaling_factor"]).to(device) # Pass scaling factor
    nen_loss_fn = NENLoss().to(device)

    # Get hyperparameters from config
    lambda_weight = config.get("lambda_weight", 0.3) # Default to 0.5 if not specified
    nudity_prompt = config.get("nudity_prompt", "nudity")
    print(f"Using lambda_weight: {lambda_weight}, nudity_prompt: '{nudity_prompt}'")

    # Get the original nudity vector from the dataset (move to device once)
    original_nudity_vector_gpu = dataset.original_nudity_vector.to(device)


    for epoch in range(config["epochs"]):
        print(f"\nStarting Epoch {epoch + 1}/{config['epochs']}")
        epoch_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Unpack data from dataset (now includes original_safe_embeddings)
            target_safe_vectors, unsafe_prompts, safe_prompts, original_safe_embeddings = batch
            target_safe_vectors = target_safe_vectors.to(device)
            original_safe_embeddings = original_safe_embeddings.to(device) # Move original safe embeds to GPU

            # --- Calculate current embeddings using the *training* encoder ---
            # 1. Current unsafe embeddings (for UEN)
            current_unsafe_embeddings = text_encoder(list(unsafe_prompts)) # Already on device

            # 2. Current safe embeddings (for SEP)
            current_safe_embeddings = text_encoder(list(safe_prompts)) # Already on device

            # 3. Current nudity embedding (for SEP PALA adjustment and NEN)
            # Note: SEP uses original nudity vector, NEN uses current nudity vector
            current_nudity_embedding = text_encoder.encode(nudity_prompt) # Already on device

            # --- Calculate individual losses ---
            # L_u (UEN Loss)
            uen_loss = uen_loss_fn(current_unsafe_embeddings, target_safe_vectors)

            # L_s (SEP Loss) - uses current safe, original safe, original nudity
            sep_loss = sep_loss_fn(
                current_safe_embeddings,
                original_safe_embeddings,
                original_nudity_vector_gpu # Use the pre-loaded original nudity vector
            )

            # L_n (NEN Loss) - uses current nudity, original unconditioned
            nen_loss = nen_loss_fn(current_nudity_embedding, original_uncond_embedding)

            # --- Combine losses (Equation 7) ---
            total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + nen_loss)

            # --- Backpropagation ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() # Accumulate total loss

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Total Loss: {avg_epoch_loss:.4f}")

    torch.save(text_encoder.state_dict(), config["model_save_path"])
    print(f"Training completed. Model saved to {config['model_save_path']}")
