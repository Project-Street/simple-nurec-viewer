"""
Shared data loading module for NuRec USDZ files.

This module provides the single source of truth for loading NuRec USDZ files,
used by both viewer and export modules.
"""

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..scenes.sky import SkyCubeMap
from .viewer import GaussianSet


@dataclass
class NuRecData:
    """
    Container for all NuRec scene data.

    This dataclass encapsulates all data loaded from a NuRec USDZ file,
    providing a clean interface for the rest of the codebase.
    """

    gaussian_set: GaussianSet
    """The Gaussian set containing all 3D Gaussians"""

    sky_cubemap: Optional[SkyCubeMap]
    """Optional sky cubemap for background rendering"""

    tracks_data: Optional[dict]
    """Optional tracks data for rigid body animation"""

    camera_trajectories: Optional[dict]
    """Optional camera trajectory data"""

    world_to_nre: Optional[np.ndarray]
    """Optional world-to-NRE coordinate transformation matrix"""


def load_nurec_data(
    usdz_path: str | Path,
    device: torch.device,
    load_trajectories: bool = True,
) -> NuRecData:
    """
    Load all data from a NuRec USDZ file.

    This is the single source of truth for USDZ loading, used by
    both viewer and export modules.

    Args:
        usdz_path: Path to the USDZ file
        device: Torch device to load tensors to
        load_trajectories: Whether to load camera trajectories (default: True)

    Returns:
        NuRecData container with all loaded data

    Raises:
        FileNotFoundError: If USDZ file doesn't exist
        ValueError: If required data is missing from USDZ
    """
    usdz_path = Path(usdz_path)
    if not usdz_path.exists():
        raise FileNotFoundError(f"USDZ file not found: {usdz_path}")

    # Extract USDZ to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
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
            seq_tracks_dynamic = datasource_data.get("sequence_tracks_dynamic", {})
            if seq_tracks_dynamic:
                # Get the first (and only) entry
                first_key = list(seq_tracks_dynamic.keys())[0]
                tracks_data = seq_tracks_dynamic[first_key]
                print(f"Loaded tracks data with {len(tracks_data.get('tracks_data', {}).get('tracks_id', []))} tracks")
            else:
                print("No sequence_tracks_dynamic found in datasource_summary.json")
        else:
            print("No datasource_summary.json found")

        # Load checkpoint using GaussianSet
        ckpt_path = Path(tmpdir) / "checkpoint.ckpt"
        if not ckpt_path.exists():
            raise ValueError("checkpoint.ckpt not found in USDZ file")

        print(f"Loading checkpoint from {ckpt_path}...")
        gaussian_set = GaussianSet.from_checkpoint(str(ckpt_path), device, tracks_data=tracks_data)
        gaussian_set.print_summary()

        # Load camera trajectories from datasource_summary.json
        trajectories = None
        world_to_nre = None
        if load_trajectories and datasource_path.exists():
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
                print("Loaded world_to_nre transformation matrix")

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

        return NuRecData(
            gaussian_set=gaussian_set,
            sky_cubemap=sky_cubemap,
            tracks_data=tracks_data,
            camera_trajectories=trajectories,
            world_to_nre=world_to_nre,
        )


__all__ = [
    "NuRecData",
    "load_nurec_data",
]
