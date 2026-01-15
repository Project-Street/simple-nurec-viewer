#!/usr/bin/env python3
"""
NuRec Viewer - View NuRec USDZ files using gsplat + nerfview

This viewer loads NuRec USDZ files and renders 3D Gaussian Splatting models
with interactive camera controls and trajectory visualization.
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import viser
from gsplat.rendering import rasterization
import nerfview

from gaussian import GaussianSet
from sky_cubemap import SkyCubeMap, generate_ray_directions


def load_nurec_data(usdz_path: str, device: torch.device) -> Tuple[GaussianSet, Optional[dict], Optional[SkyCubeMap]]:
    """
    Load NuRec USDZ file and extract Gaussian parameters.

    Args:
        usdz_path: Path to the USDZ file
        device: Torch device to load tensors to

    Returns:
        Tuple of (GaussianSet, camera_trajectories, sky_cubemap)
    """
    # Extract USDZ to temp directory
    with tempfile.TemporaryDirectory(dir="./tmp") as tmpdir:
        print(f"Extracting USDZ to {tmpdir}...")
        with zipfile.ZipFile(usdz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Load checkpoint using GaussianSet
        ckpt_path = Path(tmpdir) / "checkpoint.ckpt"
        print(f"Loading checkpoint from {ckpt_path}...")
        gaussian_set = GaussianSet.from_checkpoint(str(ckpt_path), device)
        gaussian_set.print_summary()

        # Load camera trajectories from rig_trajectories.json
        traj_path = Path(tmpdir) / "rig_trajectories.json"
        trajectories = None
        if traj_path.exists():
            print(f"Loading camera trajectories from {traj_path}...")
            with open(traj_path, "r") as f:
                rig_data = json.load(f)
            trajectories = rig_data["rig_trajectories"][0]
            T_rig_worlds = trajectories["T_rig_worlds"]
            print(f"Loaded {len(T_rig_worlds)} camera poses")
        else:
            print("No rig_trajectories.json found")

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

        return gaussian_set, trajectories, sky_cubemap


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

    Returns:
        Rendered image [H, W, 3] or tuple of (rgb [H, W, 3], alpha [H, W])
    """
    backgrounds = torch.zeros(1, 3, device=device)

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
        camera_model="pinhole",
        with_ut=True,  # Enable Undistorted Transform (UT)
        with_eval3d=True,  # Enable Eval3D
        packed=False,  # UT mode requires packed=False
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
    gaussian_set: "GaussianSet",
    sky_cubemap: Optional["SkyCubeMap"],
    device: torch.device,
) -> np.ndarray:
    """
    Render function for nerfview with merged background and road Gaussians and sky cubemap.

    Args:
        camera_state: Current camera state from nerfview
        render_tab_state: Render tab state from nerfview
        gaussian_set: GaussianSet containing background and road Gaussians
        sky_cubemap: Optional SkyCubeMap for sky rendering
        device: Torch device

    Returns:
        Rendered RGB image as numpy array [H, W, 3]
    """
    # Get camera parameters
    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = torch.from_numpy(camera_state.c2w).float().to(device)
    K = torch.from_numpy(camera_state.get_K([width, height])).float().to(device)
    viewmat = c2w.inverse()

    # Collect background and road Gaussians with SH-to-RGB conversion
    bg_means, bg_quats, bg_scales, bg_opacities, bg_colors = gaussian_set.background.collect(viewmat=viewmat)
    road_means, road_quats, road_scales, road_opacities, road_colors = gaussian_set.road.collect(viewmat=viewmat)

    # Merge background and road Gaussians
    means = torch.cat((bg_means, road_means), dim=0)
    quats = torch.cat((bg_quats, road_quats), dim=0)
    scales = torch.cat((bg_scales, road_scales), dim=0)
    opacities = torch.cat((bg_opacities, road_opacities), dim=0)
    colors = torch.cat((bg_colors, road_colors), dim=0)

    # Render Gaussians with alpha channel for blending
    # Use RGB mode for standard 3-channel output
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
        # Reshape alpha for broadcasting: [H, W] -> [H, W, 1]
        alpha_expanded = alpha.unsqueeze(-1)
        # Permute sky_rgb to [H, W, 3] for blending
        sky_rgb = sky_rgb.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        final_image = rgb * alpha_expanded + sky_rgb * (1 - alpha_expanded)
    else:
        final_image = rgb

    return final_image.cpu().numpy()


def add_camera_trajectories(
    server: viser.ViserServer,
    trajectories: dict,
) -> None:
    """
    Add camera trajectory visualization to the viewer.

    Args:
        server: Viser server instance
        trajectories: Trajectory data from rig_trajectories.json
    """
    if not trajectories:
        return

    T_rig_worlds = trajectories["T_rig_worlds"]

    # Extract positions from transformation matrices
    positions = []
    for T in T_rig_worlds:
        T_mat = np.array(T)  # Convert to numpy array first
        pos = T_mat[:3, 3]  # Extract translation
        positions.append(pos)

    positions = np.array(positions)

    # Prepare line segments: shape (N, 2, 3) where N is number of segments
    segments = np.stack([positions[:-1], positions[1:]], axis=1)  # (N-1, 2, 3)

    # Add trajectory as line segments
    server.scene.add_line_segments(
        name="/trajectory",
        points=segments,
        colors=np.array([1.0, 0.0, 0.0]),  # Red color
        line_width=2.0,
    )

    print(f"Added camera trajectory with {len(positions)} poses")


def main():
    """Main entry point for the NuRec viewer."""
    parser = argparse.ArgumentParser(description="View NuRec USDZ files using gsplat + nerfview")
    parser.add_argument(
        "--usdz",
        type=str,
        required=True,
        help="Path to the USDZ file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for rendering",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load NuRec data
    print(f"\nLoading NuRec data from {args.usdz}")
    print("=" * 60)
    gaussian_set, trajectories, sky_cubemap = load_nurec_data(args.usdz, device)
    print("=" * 60)

    # Setup viewer
    print(f"\nStarting viewer server on port {args.port}...")
    server = viser.ViserServer(port=args.port, verbose=False)

    # Add camera trajectories to viewer (if available)
    if trajectories:
        add_camera_trajectories(server, trajectories)

    # Create nerfview viewer with layered rendering and sky cubemap
    viewer = nerfview.Viewer(
        server=server,
        render_fn=lambda cam, tab: render_fn(cam, tab, gaussian_set, sky_cubemap, device),
        mode="rendering",
    )

    print(f"\nViewer running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit\n")

    # Keep the viewer running
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    main()
