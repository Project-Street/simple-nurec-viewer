"""
Rendering utilities for camera frames.

This module provides functions to render Gaussian splatting frames
from specified camera viewpoints using the shared rendering module.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from simple_nurec_viewer.core.rendering import RenderContext, render_frame
from simple_nurec_viewer.core.viewer import GaussianSet


def render_camera_frame(
    gaussian_set: GaussianSet,
    sky_cubemap: Optional[object],
    viewmat: np.ndarray,
    K: np.ndarray,
    resolution: Tuple[int, int],
    device: torch.device,
    timestamp: Optional[float] = None,
    camera_model: str = "pinhole",
    ftheta_coeffs=None,
) -> np.ndarray:
    """Render a single camera frame.

    Args:
        gaussian_set: GaussianSet containing all Gaussians
        sky_cubemap: Optional sky cubemap
        viewmat: View matrix [4, 4]
        K: Camera intrinsics [3, 3]
        resolution: (width, height)
        device: Torch device
        timestamp: Optional timestamp for rigid body animation
        camera_model: Camera model type ("pinhole" or "ftheta")
        ftheta_coeffs: FThetaCameraDistortionParameters for ftheta cameras

    Returns:
        Rendered RGB image [H, W, 3]
    """
    # Create RenderContext
    ctx = RenderContext(
        gaussian_set=gaussian_set,
        sky_cubemap=sky_cubemap,
        device=device,
    )

    # Render frame using shared rendering function
    return render_frame(
        ctx, viewmat, K, resolution, timestamp=timestamp, camera_model=camera_model, ftheta_coeffs=ftheta_coeffs
    )


__all__ = ["render_camera_frame"]
