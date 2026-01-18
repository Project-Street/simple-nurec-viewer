"""
Gaussian abstraction layer for 3D Gaussian Splatting.

This module provides the Gaussian hierarchy (BaseGaussian, RigidGaussian, HybridGaussian)
following the StreetStudio reference pattern for managing and rendering Gaussian primitives.
"""

from simple_nurec_viewer.scenes.gaussians.base import BaseGaussian
from simple_nurec_viewer.scenes.gaussians.hybrid import HybridGaussian
from simple_nurec_viewer.scenes.gaussians.rigid import RigidGaussian

__all__ = ["BaseGaussian", "RigidGaussian", "HybridGaussian"]
