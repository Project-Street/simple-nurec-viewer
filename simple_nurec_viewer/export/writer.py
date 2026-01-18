"""
Output utilities for saving rendered frames.

This module provides functions for saving rendered images and managing
output directory structure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image


@dataclass
class ExportTask:
    """Configuration for export task.

    Attributes:
        usdz_path: Input USDZ file path
        output_dir: Output directory path
        cameras: Optional list of camera names to export (None = all)
        resolution_scale: Resolution scale factor (0, 1]
        device: Rendering device
        timestamp_start: Optional start frame index
        timestamp_end: Optional end frame index
        overwrite: Whether to overwrite existing files
        debug: Whether to enable debug output for coordinate transforms
    """
    usdz_path: Path
    output_dir: Path
    cameras: Optional[List[str]] = None
    resolution_scale: float = 1.0
    device: torch.device = torch.device("cuda")
    timestamp_start: Optional[int] = None
    timestamp_end: Optional[int] = None
    overwrite: bool = True
    debug: bool = False


def save_rendered_frame(
    image: np.ndarray,
    output_path: Path,
    quality: int = 95,
) -> None:
    """Save rendered frame to disk as JPEG.

    Args:
        image: Rendered RGB image [H, W, 3] in range [0, 1]
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Create PIL Image and save
    img = Image.fromarray(image_uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=quality, format="JPEG")


def create_output_directories(
    output_dir: Path,
    camera_names: List[str],
) -> dict:
    """Create output directory structure for cameras.

    Args:
        output_dir: Root output directory
        camera_names: List of camera logical names

    Returns:
        Dictionary mapping camera names to their output directories
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_dirs = {}
    for camera_name in camera_names:
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        camera_dirs[camera_name] = camera_dir

    return camera_dirs
