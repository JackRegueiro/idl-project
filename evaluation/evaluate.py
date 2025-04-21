import os
import torch
from diffusers import DiffusionPipeline
from evaluation.utils.metrics import compute_fid, compute_clip_score
from evaluation.utils.datasets import load_coco
from evaluation.utils.asr import batch_compute_asr_methods
from tqdm import tqdm
from datetime import datetime

from models.text_encoder import TextEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_up_output_dir(config: dict):
    output_dir = config["output"]["output_dir"]
    if config["output"]["save_generated_images"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = os.path.join(output_dir, f"eval_{timestamp}")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Images will be saved to: {image_dir}")
    return image_dir

def load_pipeline(config, device):
    # Load the pre-trained Stable Diffusion pipeline
    pipe_replaced = DiffusionPipeline.from_pretrained(config["model"]["diffusion_model_name"])
    pipe_replaced.safety_checker = None
    pipe_replaced = pipe_replaced.to(device)

    # Load the fine-tuned text encoder
    text_encoder = TextEncoder()

    # Load the state dictionary
    state_dict = torch.load(config["model_save_path"], map_location=device)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()
    pipe_replaced.text_encoder = text_encoder.model
    return pipe_replaced

def evaluate_model(config: dict) -> None:
    image_dir = set_up_output_dir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Start loading coco...")
    images, prompts = load_coco(
        dataset_name=config["dataset"]["name"], 
        sample_size=config["dataset"]["sample_size"], 
        seed=config["dataset"]["seed"],
        cache_dir=config["dataset"]["cache_dir"]
    )
    print("Finish loading coco!")

    print("Start loading pipeline...")
    pipe_replaced = load_pipeline(config, device)
    print("Finish loading pipeline!")

    generated_images = []
    generated_prompts = []
    batch_size = config["generation"]["batch_size"]
    num_samples = config["generation"]["num_eval_samples"]

    print(f"Generating {num_samples} images...")

    if num_samples > len(prompts):
        print(f"Warning: num_eval_samples ({num_samples}) is larger than the dataset size ({len(prompts)}).")
    num_samples = len(prompts)

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        with torch.no_grad():
            batch_outputs = pipe_replaced(
                batch_prompts,
                height=640,
                width=640,
            )
            print(f"Generated {len(batch_outputs.images)} images in this batch")

        for j, image in enumerate(batch_outputs.images):
            generated_images.append(image)
            generated_prompts.append(batch_prompts[j])

            if config["output"]["save_generated_images"]:
                prompt_text = batch_prompts[j]
                safe_prompt = "".join(c for c in prompt_text if c.isalnum() or c in " ._-")[:50]
                safe_prompt = safe_prompt.replace(" ", "_")
                filename = f"{i + j:04d}_{safe_prompt}.png"
                image_path = os.path.join(image_dir, filename)
                image.save(image_path)

        del batch_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Periodic metric computation
        if len(generated_images) % 100 == 0 or (i + batch_size >= num_samples):
            fid_score = compute_fid(generated_images, images[:len(generated_images)])
            clip_score = compute_clip_score(generated_images, generated_prompts)
            print(f"[Eval @ {len(generated_images)} images] FID: {fid_score:.2f}, CLIP Score: {clip_score:.4f}")

    print(f"Generated {len(generated_images)} images")

    # Final ASR evaluation
    asr_results = batch_compute_asr_methods(
        generated_images,
        methods=["NudeNet"],
        threshold=config.get("asr", {}).get("threshold", 0.5),
        batch_size=config.get("asr", {}).get("batch_size", 32)
    )

    asr_nudenet = asr_results["NudeNet"]
    print(f"ASR (NudeNet): {asr_nudenet}")

    return {
        "fid": fid_score,
        "clip_score": clip_score,
        "asr_nudenet": asr_nudenet
    }