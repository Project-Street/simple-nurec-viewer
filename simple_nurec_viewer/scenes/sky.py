"""
Sky Cubemap Rendering Module

This module provides sky cubemap rendering functionality using nvdiffrast.
It's a simplified, inference-only version adapted from EasyDrive's SkyCubeMap.
"""

from typing import Optional

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F


class SkyCubeMap:
    """
    Sky cubemap renderer using nvdiffrast.

    This class renders sky colors from a cubemap texture by sampling
    it with ray directions using nvdiffrast's cube boundary mode.
    """

    def __init__(self, cubemap_texture: torch.Tensor, remap_indices: Optional[list[int]] = None):
        """
        Initialize the SkyCubeMap with a cubemap texture.

        Args:
            cubemap_texture: Cubemap texture with shape [6, H, W, 3] or [1, 6, H, W, 3]
                           - 6 faces in order: +X, -X, +Y, -Y, +Z, -Z
                           - H, W: resolution of each face
                           - 3: RGB channels
            remap_indices: Optional list of 6 indices to reorder cubemap faces.
                          If provided, cubemap[remap_indices[i]] will be used as face i.
                          Example: [1, 0, 2, 3, 4, 5] swaps +X and -X faces.
        """
        # Remove batch dimension if present: [1, 6, H, W, 3] -> [6, H, W, 3]
        if cubemap_texture.dim() == 5:
            cubemap_texture = cubemap_texture.squeeze(0)

        # Validate shape
        if cubemap_texture.dim() != 4 or cubemap_texture.shape[0] != 6:
            raise ValueError(f"Expected cubemap shape [6, H, W, 3], got {cubemap_texture.shape}")

        # Remap cubemap faces if indices are provided
        if remap_indices is not None:
            if len(remap_indices) != 6:
                raise ValueError(f"remap_indices must have 6 elements, got {len(remap_indices)}")
            cubemap_texture = cubemap_texture[remap_indices]

        self.cubemap = cubemap_texture
        self.device = cubemap_texture.device
        self.resolution = cubemap_texture.shape[1]  # Assuming square faces

    @torch.no_grad()
    def render(self, H: int, W: int, ray_d: torch.Tensor) -> torch.Tensor:
        """
        Render sky colors for given ray directions.

        Args:
            H: Image height
            W: Image width
            ray_d: Ray directions in world space [H, W, 3] (should be normalized)

        Returns:
            sky_colors: RGB colors with shape [3, H, W]
        """
        # Ensure ray_d is in the right format [H, W, 3]
        if ray_d.dim() != 3 or ray_d.shape[-1] != 3:
            raise ValueError(f"Expected ray_d shape [H, W, 3], got {ray_d.shape}")

        # Normalize ray directions to ensure correct cubemap sampling
        ray_d = F.normalize(ray_d, p=2, dim=-1)

        # Ensure tensors are contiguous for nvdiffrast
        cubemap_contiguous = self.cubemap.contiguous()
        ray_d_contiguous = ray_d.contiguous()

        # First rotate by -90 degrees around z-axis
        # [ 0,  1,  0]
        # [-1,  0,  0]
        # [ 0,  0,  1]
        ray_d_rotated = ray_d_contiguous.clone()
        temp_x = ray_d_contiguous[..., 0].clone()
        ray_d_rotated[..., 0] = ray_d_contiguous[..., 1]  # x = y
        ray_d_rotated[..., 1] = -temp_x  # y = -x

        # Then rotate by -90 degrees around x-axis
        # [1,  0,  0]
        # [0,  0,  1]
        # [0, -1,  0]
        temp_y = ray_d_rotated[..., 1].clone()
        ray_d_rotated[..., 1] = ray_d_rotated[..., 2]  # y = z
        ray_d_rotated[..., 2] = -temp_y  # z = -y

        # Sample cubemap using nvdiffrast with cube boundary mode
        # dr.texture expects: [batch, faces, H, W, C] and ray directions [batch, ..., 3]
        sky_color = dr.texture(
            cubemap_contiguous[None, ...],  # Add batch dim: [1, 6, H, W, 3]
            ray_d_rotated[None, ...],  # Add batch dim: [1, H, W, 3]
            filter_mode="linear",
            boundary_mode="cube",
        )

        # Remove batch dimension and permute to [3, H, W]
        sky_color = sky_color[0].permute(2, 0, 1)  # [3, H, W]
        return sky_color


def generate_ray_directions(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Generate ray directions for all pixels in world space.

    Args:
        H: Image height
        W: Image width
        K: Camera intrinsic matrix [3, 3]
        c2w: Camera-to-world transformation matrix [4, 4]

    Returns:
        ray_d: Normalized ray directions in world space [H, W, 3]
    """
    device = K.device

    # Create pixel grid (0 to W-1, 0 to H-1)
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device), indexing="xy"
    )  # i: [W,], j: [H,] -> meshgrid -> [H, W] each

    # Extract focal lengths and principal point from intrinsic matrix
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1 ]]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Convert pixel coordinates to camera coordinates
    # Note: in camera coordinates, x points right, y points down, z points forward
    # We need to flip y because image coordinates have y pointing down
    x = (i - cx) / fx  # [H, W]
    y = (j - cy) / fy  # [H, W]

    # Stack into directions [H, W, 3]
    # Camera rays point in positive z direction
    rays_camera = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    # Normalize camera-space rays
    rays_camera = F.normalize(rays_camera, p=2, dim=-1)

    # Transform to world space using rotation from c2w
    # c2w is [4, 4], rotation is the upper-left 3x3
    R = c2w[:3, :3]  # [3, 3]
    rays_world = torch.einsum("ij,hwj->hwi", R, rays_camera)  # [H, W, 3]

    # Normalize world-space rays (should already be normalized, but ensure)
    rays_world = F.normalize(rays_world, p=2, dim=-1)

    return rays_world  # [H, W, 3]
