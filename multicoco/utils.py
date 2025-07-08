"""
Utility functions for image processing and transformations.

This module provides utility functions for image preprocessing, dynamic image
processing, and transform creation for use with InternVL models.
"""

import logging
from typing import List, Tuple, Union, Optional

# ** Core libraries
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# ** Local imports
from .constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MIN_PATCHES,
    DEFAULT_MAX_PATCHES,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from .exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


def build_transform(input_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """
    Build image transformation pipeline for model input.
    
    Creates a standardized image transformation pipeline including:
    - RGB conversion
    - Bicubic resizing
    - Tensor conversion  
    - ImageNet normalization
    
    Args:
        input_size: Target image size (square images)
        
    Returns:
        Composed transformation pipeline
        
    Raises:
        ImageProcessingError: If transform creation fails
    """
    try:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        logger.debug(f"Created image transform for size {input_size}")
        return transform
        
    except Exception as e:
        raise ImageProcessingError(f"Failed to build image transform: {e}")


def find_closest_aspect_ratio(
    aspect_ratio: float, 
    target_ratios: List[Tuple[int, int]], 
    width: int, 
    height: int, 
    image_size: int
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from a list of target ratios.
    
    This function finds the target aspect ratio that best matches the input
    image's aspect ratio, considering both ratio difference and image area.
    
    Args:
        aspect_ratio: Input image aspect ratio (width/height)
        target_ratios: List of (width_ratio, height_ratio) tuples
        width: Original image width
        height: Original image height  
        image_size: Target patch size
        
    Returns:
        Best matching (width_ratio, height_ratio) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # If ratios are equally good, prefer the one that better utilizes the image area
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    
    logger.debug(f"Selected aspect ratio {best_ratio} for input ratio {aspect_ratio:.3f}")
    return best_ratio


def dynamic_preprocess(
    image: Image.Image, 
    min_num: int = DEFAULT_MIN_PATCHES,
    max_num: int = DEFAULT_MAX_PATCHES, 
    image_size: int = DEFAULT_IMAGE_SIZE,
    use_thumbnail: bool = False
) -> List[Image.Image]:
    """
    Dynamically preprocess image into multiple patches based on aspect ratio.
    
    This function splits an image into multiple patches based on its aspect ratio
    to better preserve spatial information while fitting the model's input requirements.
    
    Args:
        image: Input PIL Image
        min_num: Minimum number of patches
        max_num: Maximum number of patches
        image_size: Size of each patch
        use_thumbnail: Whether to add a thumbnail version
        
    Returns:
        List of processed image patches
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Generate target aspect ratios
        target_ratios = _generate_target_ratios(min_num, max_num)
        
        # Find the best aspect ratio
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = _split_image_into_patches(
            resized_img, target_width, target_height, image_size, blocks
        )
        
        # Add thumbnail if requested and we have multiple patches
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        logger.debug(f"Processed image into {len(processed_images)} patches "
                    f"(target ratio: {target_aspect_ratio})")
        
        return processed_images
        
    except Exception as e:
        raise ImageProcessingError(f"Failed to dynamically preprocess image: {e}")


def _generate_target_ratios(min_num: int, max_num: int) -> List[Tuple[int, int]]:
    """
    Generate valid target aspect ratios for dynamic preprocessing.
    
    Args:
        min_num: Minimum number of patches
        max_num: Maximum number of patches
        
    Returns:
        List of (width_ratio, height_ratio) tuples sorted by total patches
    """
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def _split_image_into_patches(
    resized_img: Image.Image,
    target_width: int,
    target_height: int, 
    image_size: int,
    blocks: int
) -> List[Image.Image]:
    """
    Split a resized image into equal-sized patches.
    
    Args:
        resized_img: Resized PIL Image
        target_width: Target width after resizing
        target_height: Target height after resizing
        image_size: Size of each patch
        blocks: Total number of blocks/patches
        
    Returns:
        List of image patches
    """
    processed_images = []
    
    width_patches = target_width // image_size
    height_patches = target_height // image_size
    
    for i in range(blocks):
        # Calculate patch coordinates
        col = i % width_patches
        row = i // width_patches
        
        left = col * image_size
        top = row * image_size
        right = left + image_size
        bottom = top + image_size
        
        box = (left, top, right, bottom)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    return processed_images


def load_image(
    image_file: str, 
    input_size: int = DEFAULT_IMAGE_SIZE,
    max_num: int = DEFAULT_MAX_PATCHES
) -> torch.Tensor:
    """
    Load and preprocess an image file for model input.
    
    This is a convenience function that loads an image file and applies
    the full preprocessing pipeline including dynamic preprocessing and
    tensor conversion.
    
    Args:
        image_file: Path to the image file
        input_size: Target size for image patches
        max_num: Maximum number of patches to generate
        
    Returns:
        Stacked tensor of processed image patches
        
    Raises:
        ImageProcessingError: If image loading or processing fails
    """
    try:
        # Load image
        image = Image.open(image_file).convert('RGB')
        
        # Create transform
        transform = build_transform(input_size=input_size)
        
        # Apply dynamic preprocessing
        images = dynamic_preprocess(
            image, 
            image_size=input_size, 
            use_thumbnail=True, 
            max_num=max_num
        )
        
        # Apply transforms and stack
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        logger.info(f"Loaded image {image_file} with shape {pixel_values.shape}")
        return pixel_values
        
    except Exception as e:
        raise ImageProcessingError(f"Failed to load image {image_file}: {e}")
