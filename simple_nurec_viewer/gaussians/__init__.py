"""
Gaussian abstraction layer for 3D Gaussian Splatting.

This module provides the Gaussian hierarchy (BaseGaussian, RigidGaussian, HybridGaussian)
following the StreetStudio reference pattern for managing and rendering Gaussian primitives.
"""

from simple_nurec_viewer.gaussians.base import BaseGaussian
from simple_nurec_viewer.gaussians.rigid import RigidGaussian
from simple_nurec_viewer.gaussians.hybrid import HybridGaussian

__all__ = ["BaseGaussian", "RigidGaussian", "HybridGaussian"]
