import numpy as np
import torch
import clip
from PIL import Image
from typing import List
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3


class InceptionModel:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model = inception_v3(weights=True, transform_input=False).to(device)
        self.model.eval()
        # Remove the final classification layer
        self.model.fc = torch.nn.Identity()
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_features(self, images):
        """Get Inception features for a batch of PIL images"""
        features = []
        for image in images:
            # Process PIL image directly
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(image).squeeze().cpu().numpy()
            features.append(feature)
        
        return np.array(features)


def compute_fid(generated_images: List[Image.Image], coco_images: List[Image.Image]) -> float:
    """
    Compute Fréchet Inception Distance between generated images and COCO dataset images.
    
    Args:
        generated_images: List of PIL Image objects
        coco_images: List of PIL Image objects
        
    Returns:
        fid_score: The FID score (lower is better)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionModel(device)
    
    # Get features for generated images
    gen_features = model.get_features(generated_images)
    
    # Get features for COCO images
    real_features = model.get_features(coco_images)
    
    # Calculate mean and covariance
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu_gen - mu_real) ** 2.0)
    covmean = linalg.sqrtm(sigma_gen.dot(sigma_real))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_gen + sigma_real - 2.0 * covmean)
    
    return float(fid)


def compute_clip_score(generated_images: List[Image.Image], prompts: List[str]) -> float:
    """
    Compute CLIP score to measure text-image alignment.
    
    Args:
        generated_images: List of PIL Image objects
        prompts: List of text prompts
        
    Returns:
        clip_score: Average CLIP similarity score (higher is better)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Ensure inputs are the same length
    assert len(generated_images) == len(prompts), "Number of images and prompts must match"
    
    similarities = []
    
    # Process in smaller batches to avoid OOM
    batch_size = 32
    for i in range(0, len(generated_images), batch_size):
        batch_images = generated_images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        # Process images
        processed_images = [preprocess(img) for img in batch_images]
        image_input = torch.stack(processed_images).to(device)
        
        # Process text
        text_tokens = clip.tokenize(batch_prompts).to(device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            batch_similarities = (100.0 * (image_features * text_features).sum(dim=-1)).cpu().numpy()
            similarities.extend(batch_similarities.tolist())
    
    # Return average similarity score
    return float(np.mean(similarities))