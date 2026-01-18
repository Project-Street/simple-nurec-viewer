"""
Shared rendering module for Gaussian splatting.

This module provides the single source of truth for rendering Gaussian splatting
frames, used by both viewer (real-time) and export (batch) modules.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from gsplat.rendering import rasterization


@dataclass
class RenderContext:
    """
    Context for rendering a frame.

    This dataclass encapsulates all the data needed to render a frame,
    providing a clean interface for both viewer and export modules.
    """

    gaussian_set: "GaussianSet"
    """The Gaussian set containing all 3D Gaussians"""

    sky_cubemap: Optional["SkyCubeMap"]
    """Optional sky cubemap for background rendering"""

    device: torch.device
    """Torch device for rendering"""

    @classmethod
    def from_nurec_data(cls, data: "NuRecData", device: torch.device) -> "RenderContext":
        """Create RenderContext from NuRecData."""
        return cls(
            gaussian_set=data.gaussian_set,
            sky_cubemap=data.sky_cubemap,
            device=device,
        )


@torch.no_grad()
def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    render_mode: str = "RGB+ED",
    return_alpha: bool = False,
    camera_model: str = "pinhole",
    ftheta_coeffs=None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a single group of Gaussians using gsplat.

    Args:
        means: Gaussian centers [N, 3]
        quats: Gaussian rotations (normalized quaternions) [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        colors: RGB colors [N, 3]
        viewmat: View matrix [4, 4]
        K: Camera intrinsic matrix [3, 3]
        width: Image width
        height: Image height
        device: Torch device
        render_mode: "RGB" or "RGB+ED" (RGB + alpha/depth)
        return_alpha: If True, return (rgb, alpha), otherwise return rgb only
        camera_model: Camera model type ("pinhole" or "ftheta")
        ftheta_coeffs: FThetaCameraDistortionParameters for ftheta cameras

    Returns:
        Rendered image [H, W, 3] or tuple of (rgb [H, W, 3], alpha [H, W])
    """

    backgrounds = torch.zeros(1, 3, device=device)

    # Determine if we need UT mode based on distortion
    use_ut = ftheta_coeffs is not None or camera_model != "pinhole"

    render_colors, render_alphas, meta = rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3] - RGB colors
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=None,  # No SH conversion, colors are already RGB
        render_mode=render_mode,
        backgrounds=backgrounds,
        radius_clip=3,  # Cull distant Gaussians for performance
        camera_model=camera_model,
        with_ut=use_ut,  # Enable UT based on distortion
        with_eval3d=True,  # Enable Eval3D
        packed=False,  # UT mode requires packed=False
        ftheta_coeffs=ftheta_coeffs,  # Pass distortion coefficients
    )

    rgb = render_colors[0]  # [H, W, C]
    alpha = render_alphas[0]  # [H, W, 1]
    # Squeeze alpha to [H, W] if needed
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)  # [H, W]

    if return_alpha:
        return rgb, alpha
    return rgb


@torch.no_grad()
def render_frame(
    ctx: RenderContext,
    viewmat: np.ndarray | torch.Tensor,
    K: np.ndarray | torch.Tensor,
    resolution: Tuple[int, int],
    timestamp: Optional[float] = None,
    camera_model: str = "pinhole",
    ftheta_coeffs=None,
) -> np.ndarray:
    """
    Render a single frame from the given camera viewpoint.

    This is the shared rendering function used by both viewer (real-time)
    and export (batch) rendering.

    Args:
        ctx: RenderContext containing Gaussian set and sky cubemap
        viewmat: View matrix [4, 4] (numpy or torch tensor)
        K: Camera intrinsics [3, 3] (numpy or torch tensor)
        resolution: (width, height) tuple
        timestamp: Optional timestamp for rigid body animation
        camera_model: Camera model type ("pinhole" or "ftheta")
        ftheta_coeffs: FThetaCameraDistortionParameters for ftheta cameras

    Returns:
        Rendered RGB image as numpy array [H, W, 3]
    """
    from ..scenes.sky import generate_ray_directions

    width, height = resolution
    device = ctx.device

    # Convert to torch tensors
    if isinstance(viewmat, np.ndarray):
        viewmat_t = torch.from_numpy(viewmat).float().to(device)
    else:
        viewmat_t = viewmat.float().to(device)

    if isinstance(K, np.ndarray):
        K_t = torch.from_numpy(K).float().to(device)
    else:
        K_t = K.float().to(device)

    # Collect Gaussians
    means, quats, scales, opacities, colors = ctx.gaussian_set.hybrid.collect(timestamp=timestamp, viewmat=viewmat_t)

    # Render Gaussians with alpha channel for blending
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
    if ctx.sky_cubemap is not None:
        # Compute ray directions
        c2w = torch.linalg.inv(viewmat_t)
        ray_d = generate_ray_directions(height, width, K_t, c2w)  # [H, W, 3]

        # Render sky
        sky_rgb = ctx.sky_cubemap.render(height, width, ray_d)  # [3, H, W]

        # Alpha blend
        alpha_expanded = alpha.unsqueeze(-1).clamp(0, 1)  # [H, W, 1]
        sky_rgb = sky_rgb.permute(1, 2, 0)  # [H, W, 3]
        final_image = rgb * alpha_expanded + sky_rgb * (1 - alpha_expanded)
    else:
        final_image = rgb

    return final_image.cpu().numpy()


__all__ = [
    "RenderContext",
    "render_gaussians",
    "render_frame",
]
