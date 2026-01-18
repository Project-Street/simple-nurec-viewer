"""
Simple NuRec Viewer - A 3D Gaussian Splatting viewer for NuRec USDZ files.

This package provides tools for loading, rendering, and visualizing NuRec USDZ files
using Gaussian Splatting with interactive web-based visualization.
"""

__version__ = "0.1.0"

from simple_nurec_viewer.core import GaussianSet, load_nurec_data
from simple_nurec_viewer.scenes.gaussians import BaseGaussian, HybridGaussian, RigidGaussian
from simple_nurec_viewer.scenes.sky import SkyCubeMap

__all__ = [
    "BaseGaussian",
    "RigidGaussian",
    "HybridGaussian",
    "GaussianSet",
    "SkyCubeMap",
    "load_nurec_data",
]
