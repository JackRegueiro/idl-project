from typing import List
import torch

def compute_fid(generated_images: List[torch.Tensor], real_images_path: str) -> float:
    """
    Compute the Frechet Inception Distance (FID) score.
    """
    # TODO
    # NOTE: the params maybe should change, just threw them in as placeholders
    pass

def compute_clip_score(generated_images: List[torch.Tensor], prompts: List[str]) -> float:
    """
    Compute the CLIP score between generated images and prompts.
    """
    # TODO
    # NOTE: the params maybe should change, just threw them in as placeholders
    pass

def compute_asr(generated_images: List[torch.Tensor], method: str = "NudeNet") -> float:
    """
    Compute the Attack Success Rate (ASR) using the specified NSFW classifier.
    """
    # TODO
    # NOTE: the params maybe should change, just threw them in as placeholders
    pass
