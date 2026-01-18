"""
Rendering utilities for camera frames.

This module provides functions to render Gaussian splatting frames
from specified camera viewpoints.
"""

from typing import Tuple, Optional

import numpy as np
import torch

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
    from simple_nurec_viewer.core.viewer import render_gaussians, generate_ray_directions

    width, height = resolution

    # Convert to torch tensors
    viewmat_t = torch.from_numpy(viewmat).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)

    # Collect Gaussians
    means, quats, scales, opacities, colors = gaussian_set.hybrid.collect(timestamp=timestamp, viewmat=viewmat_t)

    # Render Gaussians with distortion support
    rgb, alpha = render_gaussians(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat_t,
        K_t,
        width,
        height,
        device,
        render_mode="RGB",
        return_alpha=True,
        camera_model=camera_model,
        ftheta_coeffs=ftheta_coeffs,
    )  # rgb: [H, W, 3], alpha: [H, W]

    # Blend with sky
    if sky_cubemap is not None:
        # Compute ray directions
        c2w = np.linalg.inv(viewmat)
        c2w_t = torch.from_numpy(c2w).float().to(device)
        ray_d = generate_ray_directions(height, width, K_t, c2w_t)  # [H, W, 3]

        # Render sky
        sky_rgb = sky_cubemap.render(height, width, ray_d)  # [3, H, W]

        # Alpha blend
        alpha_expanded = alpha.unsqueeze(-1).clamp(0, 1)  # [H, W, 1]
        sky_rgb = sky_rgb.permute(1, 2, 0)  # [H, W, 3]
        final_image = rgb * alpha_expanded + sky_rgb * (1 - alpha_expanded)
    else:
        final_image = rgb

    return final_image.cpu().numpy()
