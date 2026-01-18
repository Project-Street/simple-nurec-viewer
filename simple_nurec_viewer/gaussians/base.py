"""
Base Gaussian class for 3D Gaussian Splatting.

This module provides the abstract base class for all Gaussian types,
defining the common interface for Gaussian parameter management and rendering.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics


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

    @abstractmethod
    def collect(
        self,
        viewmat: Optional[torch.Tensor] = None,
        sh_degree: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform features into render-ready Gaussian parameters.

        This method must be implemented by subclasses to provide specific
        transformation logic (e.g., rigid body transforms for dynamic Gaussians).

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
        pass

    def _collect_impl(
        self,
        viewmat: Optional[torch.Tensor] = None,
        sh_degree: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Default implementation of collect() for static Gaussians.

        This method performs the common transformation steps:
        1. Normalize quaternions
        2. Convert densities to opacities via sigmoid
        3. Combine albedo and specular features into SH coefficients
        4. If viewmat is provided, convert SH coefficients to RGB colors

        Args:
            viewmat: Optional view matrix [4, 4] for SH-to-RGB conversion
            sh_degree: Spherical harmonics degree (default: 3)

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
            # Road Gaussians: [N, 3] -> [N, 5, 3], only DC component
            features_albedo = torch.zeros(
                self.features_albedo.shape[0], 5, 3, device=self.device, dtype=self.features_albedo.dtype
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
