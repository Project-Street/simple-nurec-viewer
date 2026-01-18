"""
NuRec Viewer Core Module

This module provides the core rendering logic for viewing NuRec USDZ files,
including GaussianSet class and viewer-specific UI integration.
"""

from typing import Optional

import nerfview
import numpy as np
import torch
import viser

from ..scenes.gaussians.base import BaseGaussian
from ..scenes.gaussians.hybrid import HybridGaussian
from ..scenes.gaussians.rigid import RigidGaussian
from ..scenes.sky import SkyCubeMap
from .rendering import RenderContext, render_frame


class GaussianSet:
    """
    Manages a collection of Gaussian groups using HybridGaussian.

    This class loads Gaussian data from a checkpoint and provides
    unified access to all Gaussians through a HybridGaussian aggregator.
    """

    def __init__(self, state_dict: dict, device: torch.device, tracks_data: Optional[dict] = None):
        """
        Initialize a GaussianSet from a state dictionary.

        Args:
            state_dict: Model state dict containing Gaussian parameters
            device: Torch device
            tracks_data: Optional tracks data from sequence_tracks.json
        """
        self.device = device

        # Create background Gaussian using BaseGaussian
        background = BaseGaussian(
            positions=state_dict["model.gaussians_nodes.background.positions"],
            rotations=state_dict["model.gaussians_nodes.background.rotations"],
            scales=state_dict["model.gaussians_nodes.background.scales"],
            densities=state_dict["model.gaussians_nodes.background.densities"],
            features_albedo=state_dict["model.gaussians_nodes.background.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.background.features_specular"],
            device=device,
        )

        # Create road Gaussian using BaseGaussian
        road = BaseGaussian(
            positions=state_dict["model.gaussians_nodes.road.positions"],
            rotations=state_dict["model.gaussians_nodes.road.rotations"],
            scales=state_dict["model.gaussians_nodes.road.scales"],
            densities=state_dict["model.gaussians_nodes.road.densities"],
            features_albedo=state_dict["model.gaussians_nodes.road.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.road.features_specular"],
            device=device,
        )

        # Collect all Gaussians to aggregate
        gaussian_list = [background, road]

        # Extract rigid Gaussians if available
        dr_key = "model.gaussians_nodes.dynamic_rigids.positions"
        if dr_key in state_dict and state_dict[dr_key].shape[0] > 0:
            rigids = RigidGaussian(
                positions=state_dict["model.gaussians_nodes.dynamic_rigids.positions"],
                rotations=state_dict["model.gaussians_nodes.dynamic_rigids.rotations"],
                scales=state_dict["model.gaussians_nodes.dynamic_rigids.scales"],
                densities=state_dict["model.gaussians_nodes.dynamic_rigids.densities"],
                features_albedo=state_dict["model.gaussians_nodes.dynamic_rigids.features_albedo"],
                features_specular=state_dict["model.gaussians_nodes.dynamic_rigids.features_specular"],
                cuboid_ids=state_dict["model.gaussians_nodes.dynamic_rigids.gaussian_cuboid_ids"],
                tracks_data=tracks_data,
                device=device,
            )
            gaussian_list.append(rigids)
            self.rigids = rigids
        else:
            self.rigids = None

        # Create HybridGaussian to aggregate all Gaussian groups
        self.hybrid = HybridGaussian(gaussian_list)

        # Keep references for backward compatibility
        self.background = background
        self.road = road

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device, tracks_data: Optional[dict] = None):
        """
        Load GaussianSet from a checkpoint file.

        Args:
            ckpt_path: Path to the checkpoint file
            device: Torch device
            tracks_data: Optional tracks data from sequence_tracks.json

        Returns:
            GaussianSet instance
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        return cls(ckpt["state_dict"], device, tracks_data)

    def print_summary(self):
        """Print a summary of the loaded Gaussians."""
        bg_count = self.background.positions.shape[0]
        road_count = self.road.positions.shape[0]
        rigid_count = self.rigids.positions.shape[0] if self.rigids is not None else 0
        total_count = self.hybrid.get_gaussian_count()

        print("Loaded GaussianSet:")
        print(f"  Background: {bg_count:,} Gaussians")
        print(f"  Road: {road_count:,} Gaussians")
        if self.rigids is not None:
            num_unique_cuboids = len(torch.unique(self.rigids.cuboid_ids))
            print(f"  Rigids: {rigid_count:,} Gaussians ({num_unique_cuboids} cuboids)")
        print(f"  Total: {total_count:,} Gaussians")


@torch.no_grad()
def render_fn(
    camera_state: nerfview.CameraState,
    render_tab_state: nerfview.RenderTabState,
    gaussian_set: GaussianSet,
    sky_cubemap: Optional[SkyCubeMap],
    device: torch.device,
    timestamp: Optional[float] = None,
) -> np.ndarray:
    """
    Render function for nerfview with merged Gaussians and sky cubemap.

    Args:
        camera_state: Current camera state from nerfview
        render_tab_state: Render tab state from nerfview
        gaussian_set: GaussianSet containing background, road, and rigid Gaussians
        sky_cubemap: Optional SkyCubeMap for sky rendering
        device: Torch device
        timestamp: Optional timestamp for rigid body animation

    Returns:
        Rendered RGB image as numpy array [H, W, 3]
    """
    # Get camera parameters
    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = torch.from_numpy(camera_state.c2w).float().to(device)
    K = torch.from_numpy(camera_state.get_K([width, height])).float().to(device)
    viewmat = c2w.inverse()

    # Create RenderContext
    ctx = RenderContext(
        gaussian_set=gaussian_set,
        sky_cubemap=sky_cubemap,
        device=device,
    )

    # Render frame using shared rendering function
    return render_frame(ctx, viewmat, K, (width, height), timestamp=timestamp)


def add_camera_trajectories(
    server: viser.ViserServer,
    trajectories: dict,
    world_to_nre: Optional[np.ndarray] = None,
) -> None:
    """
    Add camera trajectory visualization to the viewer.

    Args:
        server: Viser server instance
        trajectories: Trajectory data from rig_trajectories.json
        world_to_nre: Optional world_to_nre transformation matrix [4, 4]
    """
    if not trajectories:
        return

    T_rig_worlds = trajectories["T_rig_worlds"]

    # Extract camera centers from transformation matrices
    camera_centers = []
    for T in T_rig_worlds:
        T_mat = np.array(T)
        camera_centers.append(T_mat[:3, 3])  # Extract translation (camera center position)

    camera_centers = np.array(camera_centers)

    # Apply world_to_nre transformation if available
    if world_to_nre is not None:
        T_base = world_to_nre
        camera_centers = camera_centers @ T_base[:3, :3].T + T_base[:3, 3]

    # Prepare line segments: shape (N, 2, 3) where N is number of segments
    segments = np.stack([camera_centers[:-1], camera_centers[1:]], axis=1)  # (N-1, 2, 3)

    # Add trajectory as line segments
    server.scene.add_line_segments(
        name="/trajectory",
        points=segments,
        colors=np.array([1.0, 0.0, 0.0]),  # Red color
        line_width=2.0,
    )

    print(f"Added camera trajectory with {len(camera_centers)} poses")


__all__ = [
    "GaussianSet",
    "render_fn",
    "add_camera_trajectories",
]
