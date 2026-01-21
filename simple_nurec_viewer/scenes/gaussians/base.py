"""
Base Gaussian class for 3D Gaussian Splatting.

This module provides the abstract base class for all Gaussian types,
defining the common interface for Gaussian parameter management and rendering.
"""

from abc import ABC
from typing import Tuple

import torch
import torch.nn.functional as F


class BaseGaussian(ABC):
    """
    Abstract base class for Gaussian primitives.

    This class defines the common interface for managing Gaussian parameters
    (positions, rotations, scales, opacities, features) and transforming them
    into render-ready parameters via the collect() method.

    Attributes:
        positions: Gaussian centers [N, 3]
        rotations: Gaussian rotations (quaternions) [N, 4]
        scales: Gaussian scales [N, 3]
        densities: Gaussian densities (pre-activation) [N, 1]
        features_albedo: Albedo features [N, 3] or [N, 5, 3]
        features_specular: Specular features [N, 45]
        device: Torch device
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
        """
        Initialize a BaseGaussian.

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

    def collect(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform features into render-ready Gaussian parameters.

        This method provides the default implementation for static Gaussians.
        Subclasses can override this to provide specific transformation logic
        (e.g., rigid body transforms for dynamic Gaussians).

        Args:
            **kwargs: Optional parameters (ignored by BaseGaussian, but may be used by subclasses)
                - timestamp: Optional timestamp for rigid transforms

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
            - means: Gaussian centers [N, 3]
            - quats: Normalized quaternions [N, 4]
            - scales: Scales (exp activated) [N, 3]
            - opacities: Opacities (sigmoid activated) [N]
            - colors: SH coefficients [N, K, 3] where K depends on feature dimensions
        """
        return self._collect_impl()

    def _collect_impl(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Default implementation of collect() for static Gaussians.

        This method performs the common transformation steps:
        1. Normalize quaternions
        2. Convert densities to opacities via sigmoid
        3. Combine albedo and specular features into SH coefficients

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
        """
        # Normalize quaternions
        quats = F.normalize(self.rotations, p=2, dim=-1)

        # Convert densities to opacities (sigmoid activation)
        opacities = torch.sigmoid(self.densities.squeeze(-1))

        # Handle different albedo feature dimensions
        # Background: [N, 5, 3], Road: [N, 3]
        if self.features_albedo.dim() == 2:
            # Road Gaussians: [N, 3] -> [N, 1, 3] as 0th order SH
            colors = self.features_albedo.unsqueeze(1)  # [N, 1, 3]
        else:
            # Background Gaussians: [N, K, 3] -> [N, 1, 3] by averaging SH coefficients
            colors = self.features_albedo.sum(dim=1, keepdim=True)  # [N, 1, 3]

        # Reshape features_specular [N, P] -> [N, P//3, 3] and concatenate
        # Total SH bases: albedo K + P//3 (specular)
        P = self.features_specular.shape[1]
        specular_sh = self.features_specular.reshape(-1, P // 3, 3)  # [N, P//3, 3]
        colors = torch.cat([colors, specular_sh], dim=1)  # [N, K+P//3, 3]

        return self.positions, quats, self.scales.exp(), opacities, colors
