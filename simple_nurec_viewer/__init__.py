"""
Simple NuRec Viewer - A 3D Gaussian Splatting viewer for NuRec USDZ files.

This package provides tools for loading, rendering, and visualizing NuRec USDZ files
using Gaussian Splatting with interactive web-based visualization.
"""

__version__ = "0.1.0"

from simple_nurec_viewer.gaussians import BaseGaussian, RigidGaussian, HybridGaussian
from simple_nurec_viewer.core import GaussianSet, SkyCubeMap, load_nurec_data

__all__ = [
    "BaseGaussian",
    "RigidGaussian",
    "HybridGaussian",
    "GaussianSet",
    "SkyCubeMap",
    "load_nurec_data",
]
