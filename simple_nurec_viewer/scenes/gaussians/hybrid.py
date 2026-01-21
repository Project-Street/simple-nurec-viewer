"""
Hybrid Gaussian class for aggregating multiple Gaussian groups.

This module provides the HybridGaussian class that acts as a proxy
to aggregate multiple Gaussian objects and merge their collect() outputs.
"""

from typing import List, Tuple

import torch

from .base import BaseGaussian


class HybridGaussian(BaseGaussian):
    """
    Aggregator for multiple Gaussian objects.

    This class manages a collection of Gaussian objects (BaseGaussian or subclasses)
    and provides a unified interface to collect all their parameters.
    It acts as a proxy that delegates to the managed Gaussians and merges results.

    Attributes:
        gaussians: List of Gaussian objects to aggregate
    """

    def __init__(self, gaussians: List[BaseGaussian]):
        """
        Initialize a HybridGaussian from a list of Gaussian objects.

        Args:
            gaussians: List of Gaussian objects (BaseGaussian, RigidGaussian, etc.)

        Raises:
            ValueError: If gaussians list is empty
        """
        if not gaussians:
            raise ValueError("HybridGaussian requires at least one Gaussian object")

        self.gaussians = gaussians

        # Infer device from the first Gaussian
        self.device = gaussians[0].device

        # Store placeholder attributes for compatibility with BaseGaussian interface
        # These are not used directly but are needed for type checking
        self.positions = torch.cat([g.positions for g in gaussians], dim=0)
        self.rotations = torch.cat([g.rotations for g in gaussians], dim=0)
        self.scales = torch.cat([g.scales for g in gaussians], dim=0)
        self.densities = torch.cat([g.densities for g in gaussians], dim=0)
        self.features_albedo = None  # Not applicable for aggregated Gaussians
        self.features_specular = None  # Not applicable for aggregated Gaussians

    def collect(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect render-ready parameters from all managed Gaussians.

        This method calls collect() on each Gaussian and concatenates the results
        along the first dimension (number of Gaussians).

        Args:
            **kwargs: Optional parameters including:
                - timestamp: Optional timestamp for rigid transforms (seconds)

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
            - means: Concatenated Gaussian centers [total_N, 3]
            - quats: Concatenated normalized quaternions [total_N, 4]
            - scales: Concatenated scales (exp activated) [total_N, 3]
            - opacities: Concatenated opacities (sigmoid activated) [total_N]
            - colors: Concatenated SH coefficients [total_N, K, 3] where K is the maximum
                     number of SH bases across all Gaussian groups (padded with zeros if needed)
        """
        # Collect parameters from each Gaussian
        all_means = []
        all_quats = []
        all_scales = []
        all_opacities = []
        all_colors = []

        for gaussian in self.gaussians:
            # For RigidGaussian, pass timestamp; for BaseGaussian, it will be ignored
            if hasattr(gaussian, "collect"):
                means, quats, scales, opacities, colors = gaussian.collect(**kwargs)
            else:
                raise TypeError(f"Gaussian object {type(gaussian)} does not have collect() method")

            all_means.append(means)
            all_quats.append(quats)
            all_scales.append(scales)
            all_opacities.append(opacities)
            all_colors.append(colors)

        # Concatenate all parameters along the first dimension
        means_concat = torch.cat(all_means, dim=0)
        quats_concat = torch.cat(all_quats, dim=0)
        scales_concat = torch.cat(all_scales, dim=0)
        opacities_concat = torch.cat(all_opacities, dim=0)

        # Find max K and pad all colors to match (different Gaussian types may have different K)
        max_k = max(c.shape[1] for c in all_colors)
        all_colors_padded = []
        for colors in all_colors:
            k = colors.shape[1]
            if k < max_k:
                # Pad with zeros to max_k
                padding = torch.zeros(colors.shape[0], max_k - k, 3, device=colors.device, dtype=colors.dtype)
                colors_padded = torch.cat([colors, padding], dim=1)
                all_colors_padded.append(colors_padded)
            else:
                all_colors_padded.append(colors)

        colors_concat = torch.cat(all_colors_padded, dim=0)

        return means_concat, quats_concat, scales_concat, opacities_concat, colors_concat

    def get_gaussian_count(self) -> int:
        """
        Get the total number of Gaussians across all managed groups.

        Returns:
            Total number of Gaussians
        """
        return sum(g.positions.shape[0] for g in self.gaussians)

    def get_group_count(self) -> int:
        """
        Get the number of Gaussian groups being managed.

        Returns:
            Number of Gaussian objects in the aggregation
        """
        return len(self.gaussians)
