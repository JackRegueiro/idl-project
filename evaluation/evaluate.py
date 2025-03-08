import os
import torch
from diffusers import DiffusionPipeline
from evaluation.utils.metrics import compute_fid, compute_clip_score
from evaluation.utils.datasets import load_coco
from evaluation.utils.asr import batch_compute_asr  # Import the ASR functionality
from tqdm import tqdm
from datetime import datetime

from models.text_encoder import TextEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_model(config: dict) -> None:
    # Create output directory if it doesn't exist
    output_dir = config["output"]["output_dir"]
    if config["output"]["save_generated_images"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = os.path.join(output_dir, f"eval_{timestamp}")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Images will be saved to: {image_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the coco dataset
    images, prompts = load_coco(
        dataset_name=config["dataset"]["name"], 
        sample_size=config["dataset"]["sample_size"], 
        seed=config["dataset"]["seed"],
        cache_dir=config["dataset"]["cache_dir"]
    )

    # Load the pre-trained Stable Diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(config["model"]["diffusion_model_name"])
    pipe = pipe.to(device)

    # Load the fine-tuned text encoder
    text_encoder = TextEncoder()

    # Load the state dictionary
    state_dict = torch.load(config["model_save_path"])
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()
    
    # Replace the pipeline's text encoder with your fine-tuned version
    # (This assumes compatibility between your text encoder and the pipeline's interface.)
    pipe.text_encoder = text_encoder
    
    
    # Define parameters for generation
    generated_images = []
    batch_size = config["generation"]["batch_size"]
    num_samples = config["generation"]["num_eval_samples"]
    
    print(f"Generating {num_samples} images...")
    
    # Generate images in batches with progress bar
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        
        # Generate images with Stable Diffusion
        with torch.no_grad():
            batch_outputs = pipe(
                batch_prompts,
                num_inference_steps=config["generation"]["num_inference_steps"],
                guidance_scale=config["generation"]["guidance_scale"]
            )
        
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
    clip_score = compute_clip_score(generated_images, prompts)

    # Compute ASR using both NudeNet and Q16
    asr_results = batch_compute_asr(
        generated_images, 
        methods=["NudeNet", "Q16"],
        threshold=config.get("asr", {}).get("threshold", 0.5),
        batch_size=config.get("asr", {}).get("batch_size", 32)
    )
    
    asr_nudenet = asr_results["NudeNet"]
    asr_q16 = asr_results["Q16"]
    
    print(f"FID: {fid_score}, CLIP Score: {clip_score}")
    print(f"ASR (NudeNet): {asr_nudenet}, ASR (Q16): {asr_q16}")
    
    # Return the evaluation results for potential logging or further processing
    return {
        "fid": fid_score,
        "clip_score": clip_score,
        "asr_nudenet": asr_nudenet,
        "asr_q16": asr_q16
    }


if __name__ == "__main__":
    # config dict for local testing
    config = {
        "dataset": {
            "name": "sayakpaul/coco-30-val-2014",
            "sample_size": 2,  # Smaller sample for faster testing
            "seed": 42, # for data sampling
            "cache_dir": os.path.join(BASE_DIR, "data", "coco")
        },
        "model": {
            "diffusion_model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5"
        },
        "generation": {
            "batch_size": 2,
            "num_inference_steps": 20,  # Fewer steps for faster generation
            "guidance_scale": 7.5,
            "num_eval_samples": 2  # Generate fewer images for quick testing
        },
        "output": {
            "save_generated_images": True,
            "output_dir": "./generated_images"
        },
        "asr": {
            "threshold": 0.5,
            "batch_size": 2
        }
    }
    print(os.path.join(BASE_DIR, "data", "coco"))
    # evaluate_model(config)