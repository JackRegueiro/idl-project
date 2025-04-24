import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Any, Dict, List, Optional, Tuple
import argparse
import yaml
import json # For parsing potential dict arguments
import os
from tqdm import tqdm

# Assume these imports are correct relative to the project structure
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

# --- Argument Parsing ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train the DES Text Encoder model.")

    # --- Core Training Parameters ---
    parser.add_argument('--config_path', type=str, default=None,
                        help="Path to a YAML configuration file to load base settings.")
    parser.add_argument('--data_path', type=str, default="./CoPro Dataset/CoPro_v1.0.json",
                        help="Path to the CoPro dataset JSON file.")
    parser.add_argument('--model_save_path', type=str, default="./trained_text_encoder.pth",
                        help="Path where the trained model state dict will be saved.")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Number of training epochs.")
    parser.add_argument('--scaling_factor', type=float, default=200.0,
                        help="Scaling factor (sg) for nudity vector subtraction in dataset and SEPLoss.")
    parser.add_argument('--lambda_weight', type=float, default=0.3,
                        help="Weight for the SEP/MarginSEP loss component (lambda in Eq. 7).")
    parser.add_argument('--nudity_prompt', type=str, default="nudity",
                        help="Prompt used to calculate the nudity vector.")
    parser.add_argument('--uncond_prompt', type=str, default="",
                        help="Unconditional prompt for NEN/MCN loss calculation.")

    # --- Loss Activation Flags ---
    parser.add_argument('--use_mcn', action='store_true',
                        help="Use MultiConceptNENLoss instead of NENLoss.")
    parser.add_argument('--use_ortho', action='store_true',
                        help="Enable Orthogonality Loss.")
    parser.add_argument('--use_push', action='store_true',
                        help="Enable PushAway Loss (original unsafe vs current unsafe).")
    parser.add_argument('--use_push_harm', action='store_true',
                        help="Enable PushAway Loss against harmful concepts (requires --harmful_concepts).")
    parser.add_argument('--use_margin_sep', action='store_true',
                        help="Use MarginSEPLoss instead of SEPLoss.")
    parser.add_argument('--use_mmd', action='store_true',
                        help="Enable MMD Loss.")

    # --- Loss Weights & Parameters ---
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Weight for Orthogonality Loss.")
    parser.add_argument('--delta1', type=float, default=0.1,
                        help="Weight for PushAway Loss (original unsafe).")
    parser.add_argument('--delta2', type=float, default=0.1,
                        help="Weight for PushAway Loss (harmful concepts).")
    parser.add_argument('--mu', type=float, default=0.1,
                        help="Weight for MMD Loss.")
    parser.add_argument('--margin_s', type=float, default=0.9,
                        help="Margin 's' for MarginSEPLoss.")
    parser.add_argument('--mmd_sigma', type=float, default=1.0,
                        help="Kernel sigma for MMDLoss.")

    # --- Extension Specific Inputs ---
    parser.add_argument('--harmful_concepts', type=str, nargs='+', default=["violence", "hate speech"],
                        help="List of harmful concepts for MCN/PushHarm losses.")
    # Optional weights for MCN - expecting a JSON string, or rely on config file
    parser.add_argument('--mcn_weights', type=str, default=None,
                        help='Optional weights for MCN as a JSON string (e.g., \'{"violence": 1.0, "hate speech": 1.2}\').')
    parser.add_argument('--harm_directions_paths', type=str, nargs='*', default=[],
                        help="Paths to saved .pt files containing harmful directions for Ortho loss.")

    return parser.parse_args()

# --- Configuration Loading ---

def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads configuration from file and merges with command-line arguments."""
    config = {}
    # Load base config from file if provided
    if args.config_path:
        try:
            with open(args.config_path, 'r') as f:
                if args.config_path.endswith(".yaml") or args.config_path.endswith(".yml"):
                    config = yaml.safe_load(f)
                elif args.config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    print(f"Warning: Unknown config file format for {args.config_path}. Attempting YAML load.")
                    config = yaml.safe_load(f)
            print(f"Loaded base configuration from: {args.config_path}")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {args.config_path}. Using defaults and CLI args.")
        except Exception as e:
            print(f"Error loading config file {args.config_path}: {e}. Using defaults and CLI args.")

    # Override with command-line arguments
    cli_args = vars(args)
    config.update(cli_args) # CLI arguments take precedence

    # Parse MCN weights if provided as JSON string
    if isinstance(config.get('mcn_weights'), str):
        try:
            config['mcn_weights'] = json.loads(config['mcn_weights'])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse --mcn_weights JSON string: {config['mcn_weights']}. Using None.")
            config['mcn_weights'] = None

    # Remove helper args
    config.pop('config_path', None)

    # Basic validation/defaults for nested structures if needed
    if 'extensions' not in config:
         config['extensions'] = {} # Ensure extensions dict exists if not loaded

    # Transfer relevant CLI args into extensions if not loaded from file
    # (This assumes the config file structure matches the argparse names)
    ext_keys_weights = ['gamma', 'delta1', 'delta2', 'mu', 'margin_s', 'mmd_sigma']
    ext_keys_flags = ['use_mcn', 'use_ortho', 'use_push', 'use_push_harm', 'use_margin_sep', 'use_mmd']
    ext_keys_other = ['harmful_concepts', 'mcn_weights', 'harm_directions_paths']

    for key in ext_keys_weights + ext_keys_flags + ext_keys_other:
        if key in config:
            config['extensions'][key] = config.pop(key) # Move to extensions sub-dict

    return config


# --- Training Function ---

def train(config: Dict[str, Any]) -> None:
    """
    Training loop for the DES model using UEN, SEP, and NEN losses,
    with commented-out extensions.

    :param config: Dictionary containing training hyperparameters and paths.
    """
    # See Algorithm 2 of https://arxiv.org/abs/2501.18877
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print("Effective configuration:")
    # Print config nicely (optional, but helpful for debugging)
    import pprint
    pprint.pprint(config)


    # --- Instantiate Models ---
    text_encoder = TextEncoder().to(device)
    text_encoder.train()
    original_encoder_temp = TextEncoder().to(device)
    original_encoder_temp.eval()

    # --- Precompute Fixed Embeddings (Conditional) ---
    print("Calculating original fixed embeddings...")
    original_uncond_embedding = None
    original_harmful_embeddings = {}
    harm_directions: List[torch.Tensor] = []
    ext_config = config.get("extensions", {})

    with torch.no_grad():
        # Unconditional embedding (needed for NEN or MCN)
        if not ext_config.get("use_mcn", False) or ext_config.get("use_mcn", False): # Check if either NEN (default) or MCN is used
            original_uncond_embedding = original_encoder_temp.encode(
                config.get("uncond_prompt", "")
            ).detach().to(device)

        # Harmful concepts (needed for MCN or PushHarm)
        harmful_concepts_list = ext_config.get("harmful_concepts", [])
        if (ext_config.get("use_mcn", False) or ext_config.get("use_push_harm", False)) and harmful_concepts_list:
            print(f"Calculating original embeddings for harmful concepts: {harmful_concepts_list}")
            for concept in harmful_concepts_list:
                 original_harmful_embeddings[concept] = original_encoder_temp.encode(concept).detach().cpu() # Store on CPU

        # Harmful directions (needed for OrthoLoss)
        harm_directions_paths = ext_config.get("harm_directions_paths", [])
        if ext_config.get("use_ortho", False) and harm_directions_paths:
            print(f"Loading harmful directions for OrthoLoss from: {harm_directions_paths}")
            try:
                harm_directions = [torch.load(p).to(device) for p in harm_directions_paths]
                print(f"Loaded {len(harm_directions)} harmful directions.")
            except Exception as e:
                print(f"Warning: Failed to load harmful directions: {e}. Ortho loss may not function.")
                harm_directions = []
        elif ext_config.get("use_ortho", False):
             print("Warning: Ortho loss enabled (--use_ortho) but no --harm_directions_paths provided.")


    # Pass the original encoder to the dataset for its own precomputations
    dataset_original_encoder = original_encoder_temp
    print("Original fixed embeddings calculated.")

    # --- Dataset and Dataloader ---
    dataset = CoProDataset(
        data_path=config["data_path"],
        text_encoder=dataset_original_encoder,
        nudity_prompt=config.get("nudity_prompt", "nudity"),
        scaling_factor=config["scaling_factor"]
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    del original_encoder_temp # No longer needed after dataset init and precompute
    original_harmful_embeddings_gpu = {k: v.to(device) for k, v in original_harmful_embeddings.items()}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    # --- Optimizer ---
    optimizer = AdamW(
        text_encoder.parameters(),
        lr=config["learning_rate"]
    )

    # --- Instantiate Loss Functions (Conditional) ---
    uen_loss_fn = UENLoss().to(device)
    sep_loss_fn: Optional[SEPLoss] = None
    nen_loss_fn: Optional[NENLoss] = None
    mcn_loss_fn: Optional[MultiConceptNENLoss] = None
    ortho_loss_fn: Optional[OrthogonalityLoss] = None
    push_loss_fn: Optional[PushAwayLoss] = None
    margin_sep_loss_fn: Optional[MarginSEPLoss] = None
    mmd_loss_fn: Optional[MMDLoss] = None

    # Base Losses (one of SEP/MarginSEP and one of NEN/MCN will always be active unless lambda=0 or 1)
    if not ext_config.get("use_margin_sep", False):
        sep_loss_fn = SEPLoss(scale_g=config["scaling_factor"]).to(device)
    else:
        margin_sep_loss_fn = MarginSEPLoss(
            margin_s=ext_config.get("margin_s", 0.9),
            scale_g=config["scaling_factor"]
        ).to(device)

    if not ext_config.get("use_mcn", False):
        nen_loss_fn = NENLoss().to(device)
    else:
        mcn_loss_fn = MultiConceptNENLoss().to(device)

    # Optional Extension Losses
    if ext_config.get("use_ortho", False):
        ortho_loss_fn = OrthogonalityLoss().to(device)
    if ext_config.get("use_push", False) or ext_config.get("use_push_harm", False):
        push_loss_fn = PushAwayLoss().to(device) # Instantiate if either push type is used
    if ext_config.get("use_mmd", False):
        mmd_loss_fn = MMDLoss(kernel_sigma=ext_config.get("mmd_sigma", 1.0)).to(device)

    # --- Get Original Nudity Vector ---
    original_nudity_vector_gpu = dataset.original_nudity_vector.to(device)
    lambda_weight = config["lambda_weight"] # Base weight for SEP vs UEN/NEN

    # --- Training Loop ---
    print(f"\n--- Starting Training ---")
    for epoch in range(config["epochs"]):
        print(f"\nStarting Epoch {epoch + 1}/{config['epochs']}")
        epoch_loss = 0.0
        num_batches = len(dataloader)
        pbar = tqdm(dataloader, total=num_batches)
        for batch_idx, batch in enumerate(pbar):
            # Unpack data
            target_safe_vectors, unsafe_prompts, safe_prompts, original_safe_embeddings, original_unsafe_embeddings = batch
            target_safe_vectors = target_safe_vectors.to(device)
            original_safe_embeddings = original_safe_embeddings.to(device)
            original_unsafe_embeddings = original_unsafe_embeddings.to(device)

            # --- Calculate current embeddings using the *training* encoder ---
            current_unsafe_embeddings = text_encoder(list(unsafe_prompts))
            current_safe_embeddings = text_encoder(list(safe_prompts))
            current_nudity_embedding = text_encoder.encode(config.get("nudity_prompt", "nudity"))
            current_harmful_embeddings = {}
            if (ext_config.get("use_mcn", False) or ext_config.get("use_push_harm", False)) and harmful_concepts_list:
                 with torch.no_grad(): # Usually don't need gradients through these for loss calc itself
                     for concept in harmful_concepts_list:
                         current_harmful_embeddings[concept] = text_encoder.encode(concept)

            # --- Calculate Individual Losses (Conditionally) ---
            uen_loss = uen_loss_fn(current_unsafe_embeddings, target_safe_vectors)
            sep_loss = 0.0
            nen_loss = 0.0
            mcn_loss = 0.0
            ortho_loss = 0.0
            push_loss_orig = 0.0
            push_loss_harm = 0.0
            mmd_loss = 0.0

            # Calculate SEP or MarginSEP
            if margin_sep_loss_fn:
                sep_loss = margin_sep_loss_fn(current_safe_embeddings, original_safe_embeddings, original_nudity_vector_gpu)
            elif sep_loss_fn:
                sep_loss = sep_loss_fn(current_safe_embeddings, original_safe_embeddings, original_nudity_vector_gpu)

            # Calculate NEN or MCN
            if mcn_loss_fn:
                 mcn_loss = mcn_loss_fn(
                     current_harmful_embeddings=current_harmful_embeddings,
                     uncond_embedding=original_uncond_embedding, # Assumes precomputed
                     concept_weights=ext_config.get("mcn_weights", None)
                 )
            elif nen_loss_fn:
                 if original_uncond_embedding is None:
                      raise ValueError("Original unconditional embedding not computed, needed for NEN loss.")
                 nen_loss = nen_loss_fn(current_nudity_embedding, original_uncond_embedding)


            # Calculate Optional Losses
            if ortho_loss_fn and harm_directions: # Only calculate if enabled AND directions loaded
                ortho_loss = ortho_loss_fn(current_unsafe_embeddings, harm_directions)

            if push_loss_fn:
                temp_push_orig, temp_push_harm = push_loss_fn(
                    current_unsafe_embeddings=current_unsafe_embeddings,
                    original_unsafe_embeddings=original_unsafe_embeddings,
                    harmful_concept_embeddings=list(original_harmful_embeddings_gpu.values()) if ext_config.get("use_push_harm", False) else None
                )
                # Only assign if corresponding flag is set
                if ext_config.get("use_push", False):
                    push_loss_orig = temp_push_orig
                if ext_config.get("use_push_harm", False):
                    push_loss_harm = temp_push_harm # This might be 0 if harmful_concept_embeddings was None

            if mmd_loss_fn:
                 mmd_loss = mmd_loss_fn(current_unsafe_embeddings, original_safe_embeddings)

            # --- Combine Losses Dynamically ---
            # Base structure: lambda * SEP + (1-lambda) * (UEN + NEN)
            # Modifications: Replace SEP with MarginSEP, NEN with MCN, add optional terms

            # Start with UEN component
            total_loss = (1 - lambda_weight) * uen_loss

            # Add SEP/MarginSEP component
            total_loss += lambda_weight * sep_loss # sep_loss holds value from MarginSEP if used

            # Add NEN/MCN component
            if mcn_loss_fn:
                total_loss += (1 - lambda_weight) * mcn_loss
            else:
                 total_loss += (1 - lambda_weight) * nen_loss # Add NEN if MCN is not used


            # Add optional weighted extension losses
            if ext_config.get("use_ortho", False):
                 total_loss += ext_config.get("gamma", 0.1) * ortho_loss
            if ext_config.get("use_push", False):
                 total_loss += ext_config.get("delta1", 0.1) * push_loss_orig
            if ext_config.get("use_push_harm", False):
                 total_loss += ext_config.get("delta2", 0.1) * push_loss_harm
            if ext_config.get("use_mmd", False):
                 total_loss += ext_config.get("mu", 0.1) * mmd_loss


            # --- Backpropagation ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_description(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Loss: {total_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Total Loss: {avg_epoch_loss:.4f}")

    # --- Save Model ---
    save_path = config["model_save_path"]
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        torch.save(text_encoder.state_dict(), save_path)
        print(f"Training completed. Model saved to {save_path}")
    except Exception as e:
         print(f"Error saving model to {save_path}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args)
    train(config)
