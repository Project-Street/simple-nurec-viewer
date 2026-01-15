#!/usr/bin/env python3
"""
Gaussian Splatting Data Management

This module provides classes for managing Gaussian Splatting data from NuRec checkpoints,
supporting both background and road Gaussians with proper feature handling.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics


class GaussianGroup:
    """Represents a single group of Gaussians (e.g., background or road).

    This class stores the raw Gaussian parameters and provides a collect() method
    to transform features into render-ready parameters.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        features_albedo: torch.Tensor,
        features_specular: torch.Tensor,
        device: torch.device,
    ):
        """Initialize a GaussianGroup.

        Args:
            positions: Gaussian centers [N, 3]
            rotations: Gaussian rotations (quaternions) [N, 4]
            scales: Gaussian scales [N, 3]
            densities: Gaussian densities (pre-activation) [N, 1]
            features_albedo: Albedo features [N, 3] or [N, 5, 3]
            features_specular: Specular features [N, 45]
            device: Torch device
        """
        self.positions = positions.to(device)
        self.rotations = rotations.to(device)
        self.scales = scales.to(device)
        self.densities = densities.to(device)
        self.features_albedo = features_albedo.to(device)
        self.features_specular = features_specular.to(device)
        self.device = device

    def collect(
        self,
        viewmat: torch.Tensor | None = None,
        sh_degree: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform features into render-ready Gaussian parameters.

        This method:
        1. Normalizes quaternions
        2. Converts densities to opacities via sigmoid
        3. Combines albedo and specular features into SH coefficients
        4. If viewmat is provided and K > 1, converts SH coefficients to RGB colors

        Args:
            viewmat: Optional view matrix [4, 4] for SH-to-RGB conversion
            sh_degree: Spherical harmonics degree (default: 3)

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
            - means: Gaussian centers [N, 3]
            - quats: Normalized quaternions [N, 4]
            - scales: Scales (exp activated) [N, 3]
            - opacities: Opacities (sigmoid activated) [N]
            - colors: RGB colors [N, 3] if viewmat provided, else SH coefficients [N, 20, 3]
        """
        # Normalize quaternions
        quats = F.normalize(self.rotations, p=2, dim=-1)

        # Convert densities to opacities (sigmoid activation)
        opacities = torch.sigmoid(self.densities.squeeze(-1))

        # Handle different albedo feature dimensions
        # Background: [N, 5, 3], Road: [N, 3]
        if self.features_albedo.dim() == 2:
            # Road Gaussians: [N, 3] -> [N, 5, 3], only DC component
            features_albedo = torch.zeros(
                self.features_albedo.shape[0], 5, 3,
                device=self.device, dtype=self.features_albedo.dtype
            )
            features_albedo[:, 0, :] = self.features_albedo
        else:
            # Background Gaussians: already [N, 5, 3]
            features_albedo = self.features_albedo

        # Combine albedo and specular features for proper SH rendering
        # features_albedo: [N, 5, 3] - 5 SH bases (degree 0-2)
        # features_specular: [N, 45] - 15 SH bases (flattened as [N, 15, 3])
        # Combined: [N, 20, 3] - total 20 SH bases for degree 3
        features_specular_reshaped = self.features_specular.reshape(-1, 15, 3)
        colors = torch.cat((features_albedo, features_specular_reshaped), dim=1)  # [N, 20, 3]

        # Convert SH coefficients to RGB colors if viewmat is provided
        if viewmat is not None and colors.shape[1] > 1:
            # Compute view directions: dirs = means - camera_position
            camtoworld = viewmat.inverse()
            camera_position = camtoworld[:3, 3]  # [3]
            dirs = self.positions - camera_position  # [N, 3]

            # Convert SH coefficients to RGB colors
            colors = spherical_harmonics(sh_degree, dirs, colors)  # [N, 3]
            colors = torch.clamp_min(colors + 0.5, 0.0)  # Shift and clamp

        return self.positions, quats, self.scales.exp(), opacities, colors


class GaussianSet:
    """Manages a collection of Gaussian groups (background + road).

    This class loads Gaussian data from a checkpoint and provides
    separate access to background and road Gaussians.
    """

    def __init__(self, state_dict: dict, device: torch.device):
        """Initialize a GaussianSet from a state dictionary.

        Args:
            state_dict: Model state dict containing Gaussian parameters
            device: Torch device
        """
        self.device = device

        # Extract background Gaussians
        self.background = GaussianGroup(
            positions=state_dict["model.gaussians_nodes.background.positions"],
            rotations=state_dict["model.gaussians_nodes.background.rotations"],
            scales=state_dict["model.gaussians_nodes.background.scales"],
            densities=state_dict["model.gaussians_nodes.background.densities"],
            features_albedo=state_dict["model.gaussians_nodes.background.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.background.features_specular"],
            device=device
        )

        # Extract road Gaussians
        self.road = GaussianGroup(
            positions=state_dict["model.gaussians_nodes.road.positions"],
            rotations=state_dict["model.gaussians_nodes.road.rotations"],
            scales=state_dict["model.gaussians_nodes.road.scales"],
            densities=state_dict["model.gaussians_nodes.road.densities"],
            features_albedo=state_dict["model.gaussians_nodes.road.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.road.features_specular"],
            device=device
        )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device):
        """Load GaussianSet from a checkpoint file.

        Args:
            ckpt_path: Path to the checkpoint file
            device: Torch device

        Returns:
            GaussianSet instance
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        return cls(ckpt["state_dict"], device)

    def print_summary(self):
        """Print a summary of the loaded Gaussians."""
        bg_count = self.background.positions.shape[0]
        road_count = self.road.positions.shape[0]
        total_count = bg_count + road_count

        print(f"Loaded GaussianSet:")
        print(f"  Background: {bg_count:,} Gaussians")
        print(f"  Road: {road_count:,} Gaussians")
        print(f"  Total: {total_count:,} Gaussians")
