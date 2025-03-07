from datasets import load_dataset
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import random


def load_coco(dataset_name="sayakpaul/coco-30-val-2014", sample_size=10000, seed=42):
    dataset_dict = load_dataset(dataset_name)
    dataset = dataset_dict["train"]

    if sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    coco_prompts = []
    coco_images = []

    for sample in dataset:
        coco_prompts.append(sample["caption"]) # string
        coco_images.append(sample["image"])  # PIL object

    return coco_images, coco_prompts


def generate_fake_images(num_images=10, save_dir='./fake_generated_images', size=(512, 512)):
    """
    Generate fake images for testing FID and CLIP score computation.
    
    Args:
        num_images: Number of fake images to generate
        save_dir: Directory to save the images
        size: Size of the images (width, height)
    
    Returns:
        List of PIL Image objects
    """
    os.makedirs(save_dir, exist_ok=True)
    images = []
    
    # Color palettes for variety
    backgrounds = [
        (255, 240, 240),  # Light red
        (240, 255, 240),  # Light green
        (240, 240, 255),  # Light blue
        (255, 255, 240),  # Light yellow
        (255, 240, 255),  # Light purple
        (240, 255, 255),  # Light cyan
    ]
    
    # Shape types to draw
    shape_types = ['circle', 'rectangle', 'line', 'text']
    
    print(f"Generating {num_images} fake images...")
    for i in tqdm(range(num_images)):
        # Create a new image with random background color
        bg_color = random.choice(backgrounds)
        img = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add 2-5 random shapes to the image
        num_shapes = random.randint(2, 5)
        for _ in range(num_shapes):
            shape_type = random.choice(shape_types)
            
            # Random color for this shape
            color = (
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200)
            )
            
            # Random position
            x1 = random.randint(0, size[0] - 1)
            y1 = random.randint(0, size[1] - 1)
            
            if shape_type == 'circle':
                radius = random.randint(20, 100)
                draw.ellipse((x1, y1, x1 + radius, y1 + radius), fill=color)
            
            elif shape_type == 'rectangle':
                width = random.randint(40, 200)
                height = random.randint(40, 200)
                draw.rectangle((x1, y1, x1 + width, y1 + height), fill=color)
            
            elif shape_type == 'line':
                x2 = random.randint(0, size[0] - 1)
                y2 = random.randint(0, size[1] - 1)
                width = random.randint(2, 8)
                draw.line((x1, y1, x2, y2), fill=color, width=width)
            
            elif shape_type == 'text':
                text = f"Image {i}"
                # Draw text (no font specified, using default)
                draw.text((x1, y1), text, fill=color)
        
        # Add noise for texture variation
        img_array = np.array(img)
        noise = np.random.randint(0, 15, img_array.shape, dtype=np.uint8)
        img_array = np.clip(img_array + noise - 7, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save the image
        img_path = os.path.join(save_dir, f"fake_image_{i:04d}.png")
        img.save(img_path)
        
        # Add to our list
        images.append(img)
    
    print(f"Generated {len(images)} fake images and saved them to {save_dir}")
    return images