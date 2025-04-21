import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Any, Dict, List
from data.dataset import CoProDataset
from models.text_encoder import TextEncoder
from losses.uen_loss import UENLoss
from losses.sep_loss import SEPLoss
from losses.nen_loss import NENLoss
from losses.mcn_loss import MultiConceptNENLoss
from losses.ortho_loss import OrthogonalityLoss
from losses.push_loss import PushAwayLoss
from losses.margin_losses import MarginSEPLoss
from losses.mmd_loss import MMDLoss
from tqdm import tqdm
import os

def train(config: Dict[str, Any]) -> None:
    """
    Training loop for the DES model using UEN, SEP, and NEN losses,
    with commented-out extensions.

    :param config: Dictionary containing training hyperparameters and paths.
    """
    # See Algorithm 2 of https://arxiv.org/abs/2501.18877
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Instantiate the text encoder to be trained
    text_encoder = TextEncoder().to(device)
    text_encoder.train() # Set the encoder to training mode

    # Instantiate the *original* text encoder for precomputation
    print("Calculating original fixed embeddings...")
    with torch.no_grad():
        original_encoder_temp = TextEncoder().to(device)
        original_encoder_temp.eval()
        # Original Unconditioned Embedding (for NEN, MCN)
        original_uncond_embedding = original_encoder_temp.encode(
            config.get("uncond_prompt", "")
        ).detach().to(device)

        # --- Precompute Embeddings for Extensions (Optional) ---
        # Harmful Concepts (for MCN, PushLoss) - Load from config
        harmful_concepts_list = config.get("extensions", {}).get("harmful_concepts", [])
        original_harmful_embeddings = {}
        if harmful_concepts_list:
            print(f"Calculating original embeddings for harmful concepts: {harmful_concepts_list}")
            for concept in harmful_concepts_list:
                 original_harmful_embeddings[concept] = original_encoder_temp.encode(concept).detach().cpu() # Store on CPU

        # Harmful Directions (for OrthoLoss) - Assume loaded/defined somehow, maybe from config path
        # Example: harm_directions_paths = config.get("extensions", {}).get("harm_directions_paths", [])
        # harm_directions = [torch.load(p).to(device) for p in harm_directions_paths]
        # For now, let's use an empty list as placeholder
        harm_directions: List[torch.Tensor] = []
        print("Harmful directions for OrthoLoss (placeholder):", "Not Loaded" if not harm_directions else f"{len(harm_directions)} loaded")
        # --- End Precompute Embeddings for Extensions ---

        # We also need the original encoder to pass to the Dataset for its precomputations
        dataset_original_encoder = original_encoder_temp # Use the same instance
    print("Original fixed embeddings calculated.")


    dataset = CoProDataset(
        data_path=config["data_path"],
        text_encoder=dataset_original_encoder, # Pass the original encoder
        nudity_prompt=config.get("nudity_prompt", "nudity"),
        scaling_factor=config["scaling_factor"]
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    # Original encoder no longer needed directly in train function after dataset init
    del original_encoder_temp
    # Move harmful embeddings to GPU once if they exist
    original_harmful_embeddings_gpu = {k: v.to(device) for k, v in original_harmful_embeddings.items()}
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

    # --- Instantiate Base Loss Functions ---
    uen_loss_fn = UENLoss().to(device)
    sep_loss_fn = SEPLoss(scale_g=config["scaling_factor"]).to(device)
    nen_loss_fn = NENLoss().to(device)
    # --- End Instantiate Base Loss Functions ---

    # --- Instantiate Extension Loss Functions (using config where needed) ---
    ext_config = config.get("extensions", {})
    mcn_loss_fn = MultiConceptNENLoss().to(device)
    ortho_loss_fn = OrthogonalityLoss().to(device)
    push_loss_fn = PushAwayLoss().to(device)
    margin_sep_loss_fn = MarginSEPLoss(
        margin_s=ext_config.get("margin_s", 0.9), # Example default margin
        scale_g=config["scaling_factor"]
    ).to(device)
    mmd_loss_fn = MMDLoss(kernel_sigma=ext_config.get("mmd_sigma", 1.0)).to(device)
    # --- End Instantiate Extension Loss Functions ---


    # Get hyperparameters from config
    lambda_weight = config.get("lambda_weight", 0.3)
    nudity_prompt = config.get("nudity_prompt", "nudity")
    print(f"Using lambda_weight: {lambda_weight}, nudity_prompt: '{nudity_prompt}'")

    # Get the original nudity vector from the dataset (move to device once)
    original_nudity_vector_gpu = dataset.original_nudity_vector.to(device)


    for epoch in range(config["epochs"]):
        print(f"\nStarting Epoch {epoch + 1}/{config['epochs']}")
        epoch_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Unpack data from dataset (now includes original_unsafe_embeddings)
            target_safe_vectors, unsafe_prompts, safe_prompts, original_safe_embeddings, original_unsafe_embeddings = batch
            target_safe_vectors = target_safe_vectors.to(device)
            original_safe_embeddings = original_safe_embeddings.to(device)
            original_unsafe_embeddings = original_unsafe_embeddings.to(device) # Move original unsafe embeds to GPU

            # --- Calculate current embeddings using the *training* encoder ---
            current_unsafe_embeddings = text_encoder(list(unsafe_prompts))
            current_safe_embeddings = text_encoder(list(safe_prompts))
            current_nudity_embedding = text_encoder.encode(nudity_prompt)
            # Current Harmful Concept Embeddings (for MCN, PushLoss)
            current_harmful_embeddings = {}
            if harmful_concepts_list:
                 with torch.no_grad(): # Usually don't need gradients through these for the loss calc itself
                     for concept in harmful_concepts_list:
                         current_harmful_embeddings[concept] = text_encoder.encode(concept)


            # --- Calculate Base Losses ---
            uen_loss = uen_loss_fn(current_unsafe_embeddings, target_safe_vectors)
            sep_loss = sep_loss_fn(
                current_safe_embeddings,
                original_safe_embeddings,
                original_nudity_vector_gpu
            )
            nen_loss = nen_loss_fn(current_nudity_embedding, original_uncond_embedding)
            # --- End Calculate Base Losses ---

            # --- Calculate Extension Losses ---
            # MCN Loss (replaces NEN)
            mcn_loss = mcn_loss_fn(
                current_harmful_embeddings=current_harmful_embeddings,
                uncond_embedding=original_uncond_embedding,
                concept_weights=ext_config.get("mcn_weights", None)
            )
            # Ortho Loss
            ortho_loss = ortho_loss_fn(
                current_unsafe_embeddings=current_unsafe_embeddings,
                harm_directions=harm_directions # Assumes harm_directions loaded earlier
            )
            # Push Away Loss
            push_loss_orig, push_loss_harm = push_loss_fn(
                current_unsafe_embeddings=current_unsafe_embeddings,
                original_unsafe_embeddings=original_unsafe_embeddings,
                # Pass original harmful embeddings (fixed targets) if using harm push
                harmful_concept_embeddings=list(original_harmful_embeddings_gpu.values()) if ext_config.get("use_push_harm", False) else None
            )
            # Margin SEP Loss (replaces SEP)
            sep_loss_margin = margin_sep_loss_fn(
                 current_safe_embeddings,
                 original_safe_embeddings,
                 original_nudity_vector_gpu
            )
            # MMD Loss
            mmd_loss = mmd_loss_fn(
                current_unsafe_embeddings=current_unsafe_embeddings,
                original_safe_embeddings=original_safe_embeddings # Compare unsafe dist to safe dist
            )
            # --- End Calculate Extension Losses ---


            # --- Combine losses (Equation 7 - Base) ---
            # total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + nen_loss)

            total_loss = lambda_weight * sep_loss \
                         + (1 - lambda_weight) * (uen_loss + nen_loss + mcn_loss) \
                         + ext_config.get("gamma", 0.1) * ortho_loss \
                         + ext_config.get("delta1", 0.1) * push_loss_orig \
                         + ext_config.get("delta2", 0.1) * push_loss_harm \
                         + sep_loss_margin \
                         + ext_config.get("mu", 0.1) * mmd_loss

            # --- Commented-out Examples for Extensions ---
            # gamma = ext_config.get("gamma", 0.1) # Example weight for ortho
            # delta1 = ext_config.get("delta1", 0.1) # Example weight for push_orig
            # delta2 = ext_config.get("delta2", 0.1) # Example weight for push_harm
            # mu = ext_config.get("mu", 0.1) # Example weight for mmd

            # Ex 1: Replace NEN with MCN
            # total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + mcn_loss)

            # Ex 2: Add Ortho Loss
            # total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + nen_loss) + gamma * ortho_loss

            # Ex 3: Add Push Away Loss (minimizing similarity means adding the loss term if weights are positive)
            # total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + nen_loss) + delta1 * push_loss_orig + delta2 * push_loss_harm

            # Ex 4: Replace SEP with Margin SEP
            # total_loss = lambda_weight * sep_loss_margin + (1 - lambda_weight) * (uen_loss + nen_loss)

            # Ex 5: Add MMD Loss
            # total_loss = lambda_weight * sep_loss + (1 - lambda_weight) * (uen_loss + nen_loss) + mu * mmd_loss
            # --- End Commented-out Examples ---


            # --- Backpropagation ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Total Loss: {avg_epoch_loss:.4f}")

    torch.save(text_encoder.state_dict(), config["model_save_path"])
    print(f"Training completed. Model saved to {config['model_save_path']}")
