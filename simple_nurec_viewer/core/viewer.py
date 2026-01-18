"""
NuRec Viewer Core Module

This module provides the core rendering logic for viewing NuRec USDZ files,
including data loading, Gaussian rendering, and sky cubemap integration.
"""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import viser
import nerfview
from gsplat.rendering import rasterization

from ..gaussians.base import BaseGaussian
from ..gaussians.rigid import RigidGaussian
from ..gaussians.hybrid import HybridGaussian
from .sky import SkyCubeMap, generate_ray_directions


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
def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    render_mode: str = "RGB+ED",
    return_alpha: bool = False,
    camera_model: str = "pinhole",
    ftheta_coeffs=None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a single group of Gaussians using gsplat.

    Args:
        means: Gaussian centers [N, 3]
        quats: Gaussian rotations (normalized quaternions) [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        colors: RGB colors [N, 3]
        viewmat: View matrix [4, 4]
        K: Camera intrinsic matrix [3, 3]
        width: Image width
        height: Image height
        device: Torch device
        render_mode: "RGB" or "RGB+ED" (RGB + alpha/depth)
        return_alpha: If True, return (rgb, alpha), otherwise return rgb only
        camera_model: Camera model type ("pinhole" or "ftheta")
        ftheta_coeffs: FThetaCameraDistortionParameters for ftheta cameras

    Returns:
        Rendered image [H, W, 3] or tuple of (rgb [H, W, 3], alpha [H, W])
    """
    backgrounds = torch.zeros(1, 3, device=device)

    # Determine if we need UT mode based on distortion
    use_ut = ftheta_coeffs is not None or camera_model != "pinhole"

    render_colors, render_alphas, meta = rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3] - RGB colors
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=None,  # No SH conversion, colors are already RGB
        render_mode=render_mode,
        backgrounds=backgrounds,
        radius_clip=3,  # Cull distant Gaussians for performance
        camera_model=camera_model,
        with_ut=use_ut,  # Enable UT based on distortion
        with_eval3d=True,  # Enable Eval3D
        packed=False,  # UT mode requires packed=False
        ftheta_coeffs=ftheta_coeffs,  # Pass distortion coefficients
    )

    rgb = render_colors[0]  # [H, W, C]
    alpha = render_alphas[0]  # [H, W, 1]
    # Squeeze alpha to [H, W] if needed
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)  # [H, W]

    if return_alpha:
        return rgb, alpha
    return rgb


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

    # Collect all Gaussians using HybridGaussian
    # HybridGaussian handles merging of background, road, and rigid Gaussians
    means, quats, scales, opacities, colors = gaussian_set.hybrid.collect(
        timestamp=timestamp, viewmat=viewmat
    )

    # Render Gaussians with alpha channel for blending
    rgb, alpha = render_gaussians(
        means, quats, scales, opacities, colors, viewmat, K, width, height, device, render_mode="RGB", return_alpha=True
    )  # rgb: [H, W, 3], alpha: [H, W]

    # Render sky and blend with Gaussians if sky cubemap is available
    if sky_cubemap is not None:
        # Generate ray directions for all pixels
        ray_d = generate_ray_directions(height, width, K, c2w)  # [H, W, 3]
        # Render sky colors
        sky_rgb = sky_cubemap.render(height, width, ray_d)  # [3, H, W]
        # Alpha blend: final = gaussian_rgb * alpha + sky_rgb * (1 - alpha)
        alpha_expanded = alpha.unsqueeze(-1)
        sky_rgb = sky_rgb.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        final_image = rgb * alpha_expanded + sky_rgb * (1 - alpha_expanded)
    else:
        final_image = rgb

    return final_image.cpu().numpy()


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


def load_nurec_data(
    usdz_path: str, device: torch.device
) -> Tuple[GaussianSet, Optional[dict], Optional[SkyCubeMap], Optional[dict], Optional[np.ndarray]]:
    """
    Load NuRec USDZ file and extract Gaussian parameters.

    Args:
        usdz_path: Path to the USDZ file
        device: Torch device to load tensors to

    Returns:
        Tuple of (GaussianSet, camera_trajectories, sky_cubemap, tracks_data, world_to_nre)
    """
    # Extract USDZ to temp directory
    with tempfile.TemporaryDirectory(dir="./tmp") as tmpdir:
        print(f"Extracting USDZ to {tmpdir}...")
        with zipfile.ZipFile(usdz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Load sequence tracks for rigid bodies from datasource_summary.json
        datasource_path = Path(tmpdir) / "datasource_summary.json"
        tracks_data = None
        if datasource_path.exists():
            print(f"Loading tracks data from {datasource_path}...")
            with open(datasource_path, "r") as f:
                datasource_data = json.load(f)
            # Extract dynamic tracks data
            # datasource_data['sequence_tracks_dynamic'] is a dict with one entry
            seq_tracks_dynamic = datasource_data.get("sequence_tracks_dynamic", {})
            if seq_tracks_dynamic:
                # Get the first (and only) entry
                first_key = list(seq_tracks_dynamic.keys())[0]
                tracks_data = seq_tracks_dynamic[first_key]
                print(
                    f"Loaded tracks data with {len(tracks_data.get('tracks_data', {}).get('tracks_id', []))} tracks"
                )
            else:
                print("No sequence_tracks_dynamic found in datasource_summary.json")
        else:
            print("No datasource_summary.json found")

        # Load checkpoint using GaussianSet
        ckpt_path = Path(tmpdir) / "checkpoint.ckpt"
        print(f"Loading checkpoint from {ckpt_path}...")
        gaussian_set = GaussianSet.from_checkpoint(str(ckpt_path), device, tracks_data=tracks_data)
        gaussian_set.print_summary()

        # Load camera trajectories from datasource_summary.json
        # Note: rig_trajectories.json is not included in USDZ, use datasource_summary.json instead
        datasource_path = Path(tmpdir) / "datasource_summary.json"
        trajectories = None
        world_to_nre = None
        if datasource_path.exists():
            print(f"Loading camera trajectories from {datasource_path}...")
            with open(datasource_path, "r") as f:
                datasource_data = json.load(f)
            rig_traj_data = datasource_data.get("rig_trajectories", {})
            if "rig_trajectories" in rig_traj_data and len(rig_traj_data["rig_trajectories"]) > 0:
                trajectories = rig_traj_data["rig_trajectories"][0]
                T_rig_worlds = trajectories["T_rig_worlds"]
                print(f"Loaded {len(T_rig_worlds)} camera poses")
            else:
                print("No rig_trajectories found in datasource_summary.json")
            # Load world_to_nre transformation
            if "world_to_nre" in rig_traj_data and "matrix" in rig_traj_data["world_to_nre"]:
                world_to_nre = np.array(rig_traj_data["world_to_nre"]["matrix"])
                print(f"Loaded world_to_nre transformation matrix")
        else:
            print("No datasource_summary.json found")

        # Load sky cubemap texture from checkpoint
        sky_cubemap = None
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model.background.textures" in ckpt["state_dict"]:
            tex = ckpt["state_dict"]["model.background.textures"]
            # Remove batch dim: [1, 6, H, W, 3] -> [6, H, W, 3]
            tex = tex.squeeze(0).to(device)
            print(f"Loaded sky cubemap: shape={tex.shape}")
            sky_cubemap = SkyCubeMap(tex)
        else:
            print("No sky cubemap found in checkpoint")
        del ckpt  # Free memory

        return gaussian_set, trajectories, sky_cubemap, tracks_data, world_to_nre
