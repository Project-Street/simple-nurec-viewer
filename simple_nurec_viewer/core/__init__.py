"""
Core rendering and visualization components.

This module contains the main rendering logic, viewer implementation, and
sky cube map rendering for the NuRec viewer.
"""

from simple_nurec_viewer.core.loader import NuRecData, load_nurec_data
from simple_nurec_viewer.core.rendering import RenderContext, render_frame, render_gaussians
from simple_nurec_viewer.core.viewer import GaussianSet, add_camera_trajectories, render_fn
from simple_nurec_viewer.scenes.sky import SkyCubeMap, generate_ray_directions

__all__ = [
    # Data structures
    "NuRecData",
    "RenderContext",
    # Data loading
    "load_nurec_data",
    # Rendering
    "render_gaussians",
    "render_frame",
    # Viewer
    "GaussianSet",
    "render_fn",
    "add_camera_trajectories",
    # Sky
    "SkyCubeMap",
    "generate_ray_directions",
]
