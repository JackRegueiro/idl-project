import os
import torch
from diffusers import DiffusionPipeline
from evaluation.utils.metrics import compute_fid, compute_clip_score
from evaluation.utils.datasets import load_coco
from evaluation.utils.asr import batch_compute_asr, batch_compute_asr_methods
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

def load_pipeline(config, device):
    # Load the pre-trained Stable Diffusion pipeline
    pipe_replaced = DiffusionPipeline.from_pretrained(config["model"]["diffusion_model_name"])
    pipe_replaced.safety_checker = None
    pipe_replaced = pipe_replaced.to(device)

    # Load the fine-tuned text encoder
    text_encoder = TextEncoder()

    # Load the state dictionary
    state_dict = torch.load(config["model_save_path"])
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()
    
    pipe_replaced.text_encoder = text_encoder
    return pipe_replaced

def evaluate_model(config: dict) -> None:
    # Create output directory if it doesn't exist
    image_dir = set_up_output_dir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the coco dataset
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
    
    # Define parameters for generation
    generated_images = []
    batch_size = config["generation"]["batch_size"]
    num_samples = config["generation"]["num_eval_samples"]
    
    print(f"Generating {num_samples} images...")

    if num_samples > len(prompts):
        print(f"Warning: num_eval_samples ({num_samples}) is larger than the dataset size ({len(prompts)}).")
    num_samples = len(prompts)
    
    # Generate images in batches with progress bar
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        # Generate images with Stable Diffusion
        with torch.no_grad():
            batch_outputs = pipe_replaced(
                batch_prompts,
                height=640,  # Set custom height (default is 512)
                width=640,   # Set custom width (default is 512)
            )
            print(f"Generated {len(batch_outputs.images)} images in this batch")

        # Process and save images
        for j, image in enumerate(batch_outputs.images):
            generated_images.append(image)

            # Save the image if configured
            if config["output"]["save_generated_images"]:
                # Create a filename with index and truncated prompt
                prompt_text = batch_prompts[j]
                # Clean up prompt for filename (limit length and remove invalid chars)
                safe_prompt = "".join(c for c in prompt_text if c.isalnum() or c in " ._-")[:50]
                safe_prompt = safe_prompt.replace(" ", "_")

                # Save the image with index and prompt
                filename = f"{i + j:04d}_{safe_prompt}.png"
                image_path = os.path.join(image_dir, filename)
                image.save(image_path)

        # Free up memory
        del batch_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Generated {len(generated_images)} images")

    # Compute evaluation metrics
    fid_score = compute_fid(generated_images, images)
    print(f"FID: {fid_score}")

    clip_score = compute_clip_score(generated_images, prompts)
    print(f"CLIP Score: {clip_score}")

    asr_results = batch_compute_asr_methods(
        generated_images,
        methods=["NudeNet"],
        threshold=config.get("asr", {}).get("threshold", 0.5),
        batch_size=config.get("asr", {}).get("batch_size", 32)
    )

    asr_nudenet = asr_results["NudeNet"]
    print(f"ASR (NudeNet): {asr_nudenet}")
    
    # Return the evaluation results for potential logging or further processing
    return {
        "fid": fid_score,
        "clip_score": clip_score,
        "asr_nudenet": asr_nudenet
    }


# if __name__ == "__main__":
#     # config dict for local testing
#     config = {
#         "dataset": {
#             "name": "sayakpaul/coco-30-val-2014",
#             "sample_size": 10000,  # Smaller sample for faster testing
#             "seed": 42, # for data sampling
#             "cache_dir": os.path.join(BASE_DIR, "data", "coco")
#         },
#         "model": {
#             "diffusion_model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5"
#         },
#         "generation": {
#             "batch_size": 2,
#             "num_inference_steps": 20,  # Fewer steps for faster generation
#             "guidance_scale": 7.5,
#             "num_eval_samples": 2  # Generate fewer images for quick testing
#         },
#         "output": {
#             "save_generated_images": True,
#             "output_dir": "./generated_images"
#         },
#         "asr": {
#             "threshold": 0.5,
#             "batch_size": 2
#         }
#     }
#     print(os.path.join(BASE_DIR, "data", "coco"))
#     # evaluate_model(config)