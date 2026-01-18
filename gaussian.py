#!/usr/bin/env python3
"""
Gaussian Splatting Data Management

This module provides classes for managing Gaussian Splatting data from NuRec checkpoints,
supporting both background and road Gaussians with proper feature handling.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import spherical_harmonics
from rigid_utils import quaternion_multiply, build_rotation, slerp


class GaussianGroup:
    """Represents a single group of Gaussians (e.g., background or road).

    This class stores the raw Gaussian parameters and provides a collect() method
    to transform features into render-ready parameters.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        features_albedo: torch.Tensor,
        features_specular: torch.Tensor,
        device: torch.device,
    ):
        """Initialize a GaussianGroup.

        Args:
            positions: Gaussian centers [N, 3]
            rotations: Gaussian rotations (quaternions) [N, 4]
            scales: Gaussian scales [N, 3]
            densities: Gaussian densities (pre-activation) [N, 1]
            features_albedo: Albedo features [N, 3] or [N, 5, 3]
            features_specular: Specular features [N, 45]
            device: Torch device
        """
        self.positions = positions.to(device)
        self.rotations = rotations.to(device)
        self.scales = scales.to(device)
        self.densities = densities.to(device)
        self.features_albedo = features_albedo.to(device)
        self.features_specular = features_specular.to(device)
        self.device = device

    def collect(
        self, viewmat: torch.Tensor | None = None, sh_degree: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform features into render-ready Gaussian parameters.

        This method:
        1. Normalizes quaternions
        2. Converts densities to opacities via sigmoid
        3. Combines albedo and specular features into SH coefficients
        4. If viewmat is provided and K > 1, converts SH coefficients to RGB colors

        Args:
            viewmat: Optional view matrix [4, 4] for SH-to-RGB conversion
            sh_degree: Spherical harmonics degree (default: 3)

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
            - means: Gaussian centers [N, 3]
            - quats: Normalized quaternions [N, 4]
            - scales: Scales (exp activated) [N, 3]
            - opacities: Opacities (sigmoid activated) [N]
            - colors: RGB colors [N, 3] if viewmat provided, else SH coefficients [N, 20, 3]
        """
        # Normalize quaternions
        quats = F.normalize(self.rotations, p=2, dim=-1)

        # Convert densities to opacities (sigmoid activation)
        opacities = torch.sigmoid(self.densities.squeeze(-1))

        # Handle different albedo feature dimensions
        # Background: [N, 5, 3], Road: [N, 3]
        if self.features_albedo.dim() == 2:
            # Road Gaussians: [N, 3] -> [N, 5, 3], only DC component
            features_albedo = torch.zeros(
                self.features_albedo.shape[0], 5, 3, device=self.device, dtype=self.features_albedo.dtype
            )
            features_albedo[:, 0, :] = self.features_albedo
        else:
            # Background Gaussians: already [N, 5, 3]
            features_albedo = self.features_albedo

        # Combine albedo and specular features for proper SH rendering
        # features_albedo: [N, 5, 3] - 5 SH bases (degree 0-2)
        # features_specular: [N, 45] - 15 SH bases (flattened as [N, 15, 3])
        # Combined: [N, 20, 3] - total 20 SH bases for degree 3
        features_specular_reshaped = self.features_specular.reshape(-1, 15, 3)
        colors = torch.cat((features_albedo, features_specular_reshaped), dim=1)  # [N, 20, 3]

        # Convert SH coefficients to RGB colors if viewmat is provided
        if viewmat is not None and colors.shape[1] > 1:
            # Compute view directions: dirs = means - camera_position
            camtoworld = viewmat.inverse()
            camera_position = camtoworld[:3, 3]  # [3]
            dirs = self.positions - camera_position  # [N, 3]

            # Convert SH coefficients to RGB colors
            colors = spherical_harmonics(sh_degree, dirs, colors)  # [N, 3]
            colors = torch.clamp_min(colors + 0.5, 0.0)  # Shift and clamp

        return self.positions, quats, self.scales.exp(), opacities, colors


class RigidGaussianGroup(GaussianGroup):
    """Represents rigid body Gaussians (e.g., vehicles, pedestrians).

    This class extends GaussianGroup to support time-varying rigid body transforms.
    Each Gaussian is associated with a track ID and can be transformed over time.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        features_albedo: torch.Tensor,
        features_specular: torch.Tensor,
        cuboid_ids: torch.Tensor,
        tracks_calib: Dict[str, torch.Tensor],
        tracks_data: Optional[dict],
        device: torch.device,
    ):
        """Initialize a RigidGaussianGroup.

        Args:
            positions: Gaussian centers [N, 3]
            rotations: Gaussian rotations (quaternions) [N, 4]
            scales: Gaussian scales [N, 3]
            densities: Gaussian densities (pre-activation) [N, 1]
            features_albedo: Albedo features [N, 5, 3]
            features_specular: Specular features [N, 45]
            cuboid_ids: Track/box ID for each Gaussian [N]
            tracks_calib: Dictionary with 'tracks_delta_q' [M, 4] and 'tracks_delta_t' [M, 3]
            tracks_data: Optional tracks data from sequence_tracks.json
            device: Torch device
        """
        super().__init__(positions, rotations, scales, densities, features_albedo, features_specular, device)

        self.cuboid_ids = cuboid_ids.to(device)
        # Note: tracks_delta_q/t are training optimization parameters that we ignore for inference
        # They are used to fine-tune trajectory poses during training but are complex to index
        # For rendering, we use the base trajectories from sequence_tracks.json directly

        # Load track trajectories from JSON if provided
        self.tracks_data = tracks_data

    def _load_track_from_usd(self, track_idx: int, timestamp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load track transform from datasource_summary.json with time interpolation.

        Args:
            track_idx: Track index (0-based, matches cuboid_id)
            timestamp: Target timestamp in seconds

        Returns:
            Tuple of (quaternion [4], translation [3])
        """
        if self.tracks_data is None:
            # Return identity transform if no tracks data
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device), torch.zeros(3, device=self.device)

        tracks_dict = self.tracks_data.get("tracks_data", {})
        tracks_poses = tracks_dict.get("tracks_poses", [])
        tracks_timestamps_us = tracks_dict.get("tracks_timestamps_us", [])

        # Check if track_idx is valid
        if track_idx >= len(tracks_poses) or track_idx >= len(tracks_timestamps_us):
            # Track index out of range, return identity
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device), torch.zeros(3, device=self.device)

        # Get poses and timestamps for this track
        poses = tracks_poses[track_idx]  # List of [N_frames, 7]
        timestamps_us = tracks_timestamps_us[track_idx]  # List of timestamps in microseconds

        # Convert to numpy arrays
        import numpy as np

        poses = np.array(poses)  # [N_frames, 7]
        timestamps_us = np.array(timestamps_us)  # [N_frames]
        timestamps_s = timestamps_us / 1e6  # Convert to seconds

        # Check if we have poses
        if len(poses) == 0:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device), torch.zeros(3, device=self.device)

        # Check if timestamp is within range
        if timestamp <= timestamps_s[0]:
            pose = poses[0]
            t = torch.tensor(pose[:3], device=self.device, dtype=torch.float32)
            q = torch.tensor(pose[3:7], device=self.device, dtype=torch.float32)
            return q, t
        elif timestamp >= timestamps_s[-1]:
            pose = poses[-1]
            t = torch.tensor(pose[:3], device=self.device, dtype=torch.float32)
            q = torch.tensor(pose[3:7], device=self.device, dtype=torch.float32)
            return q, t

        # Find the two closest timestamps for interpolation
        idx = np.searchsorted(timestamps_s, timestamp)

        # Interpolate between idx-1 and idx
        t0, t1 = timestamps_s[idx - 1], timestamps_s[idx]
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        # Get poses at t0 and t1
        pose0 = poses[idx - 1]  # [7]
        pose1 = poses[idx]  # [7]

        # Extract translations and quaternions
        # pose format: [x, y, z, qw, qx, qy, qz]
        t0_np = pose0[:3]
        t1_np = pose1[:3]
        q0_np = pose0[3:7]
        q1_np = pose1[3:7]

        # Linear interpolation for translation
        t_interp = t0_np + alpha * (t1_np - t0_np)

        # SLERP interpolation for quaternion
        q0_tensor = torch.tensor(q0_np, device=self.device, dtype=torch.float32)
        q1_tensor = torch.tensor(q1_np, device=self.device, dtype=torch.float32)
        q_interp = slerp(q0_tensor, q1_tensor, alpha)

        t = torch.tensor(t_interp, device=self.device, dtype=torch.float32)

        return q_interp, t

    def _get_base_transform(self, timestamp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base rigid transform for a given timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            Tuple of (quaternions [N], translations [N, 3]) for each Gaussian
        """
        # Build mapping from cuboid_id to its transform for this timestamp
        cuboid_to_transform = {}

        for cuboid_id in torch.unique(self.cuboid_ids):
            # cuboid_id is the track index (0-based)
            track_idx = int(cuboid_id)

            # Load track transform from JSON with interpolation
            q, t = self._load_track_from_usd(track_idx, timestamp)

            cuboid_to_transform[int(cuboid_id)] = (q, t)

        # Expand to all Gaussians based on their cuboid_ids
        num_gaussians = len(self.cuboid_ids)
        expanded_q = torch.zeros(num_gaussians, 4, device=self.device)
        expanded_t = torch.zeros(num_gaussians, 3, device=self.device)

        for i, cuboid_id in enumerate(self.cuboid_ids):
            cid_int = int(cuboid_id)
            if cid_int in cuboid_to_transform:
                q, t = cuboid_to_transform[cid_int]
                expanded_q[i] = q
                expanded_t[i] = t
            else:
                # Identity transform if no track found
                expanded_q[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                expanded_t[i] = torch.zeros(3, device=self.device)

        return expanded_q, expanded_t

    def collect(
        self, timestamp: Optional[float] = None, viewmat: Optional[torch.Tensor] = None, sh_degree: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform features into render-ready Gaussian parameters with rigid transform.

        Args:
            timestamp: Optional timestamp for rigid transform
            viewmat: Optional view matrix [4, 4] for SH-to-RGB conversion
            sh_degree: Spherical harmonics degree (default: 3)

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
        """
        # Get base Gaussian parameters
        quats = F.normalize(self.rotations, p=2, dim=-1)
        opacities = torch.sigmoid(self.densities.squeeze(-1))

        # Handle albedo features
        if self.features_albedo.dim() == 2:
            # [N, 3] -> [N, 5, 3], only DC component
            features_albedo = torch.zeros(
                self.features_albedo.shape[0], 5, 3, device=self.device, dtype=self.features_albedo.dtype
            )
            features_albedo[:, 0, :] = self.features_albedo
        else:
            features_albedo = self.features_albedo

        # Combine albedo and specular features
        features_specular_reshaped = self.features_specular.reshape(-1, 15, 3)
        colors = torch.cat((features_albedo, features_specular_reshaped), dim=1)  # [N, 20, 3]

        # Convert SH to RGB if viewmat provided
        if viewmat is not None and colors.shape[1] > 1:
            camtoworld = viewmat.inverse()
            camera_position = camtoworld[:3, 3]
            dirs = self.positions - camera_position
            colors = spherical_harmonics(sh_degree, dirs, colors)
            colors = torch.clamp_min(colors + 0.5, 0.0)

        # Apply rigid transform if timestamp is provided
        if timestamp is not None:
            # Get base transform for this timestamp
            track_q, track_t = self._get_base_transform(timestamp)

            # Build rotation matrices
            track_R = build_rotation(track_q)  # [N, 3, 3]

            # Apply rotation to positions
            # positions_transformed = (R @ positions.T).T + translation
            positions_transformed = torch.bmm(track_R, self.positions.unsqueeze(-1)).squeeze(-1) + track_t

            # Also rotate the Gaussian orientations
            # new_rotation = quaternion_multiply(track_q, base_rotation)
            rotations_transformed = quaternion_multiply(
                track_q,  # [N, 4]
                self.rotations,  # [N, 4]
            )

            return positions_transformed, rotations_transformed, self.scales.exp(), opacities, colors
        else:
            # No timestamp, return original positions
            return self.positions, quats, self.scales.exp(), opacities, colors


class GaussianSet:
    """Manages a collection of Gaussian groups (background + road + rigids).

    This class loads Gaussian data from a checkpoint and provides
    separate access to background, road, and rigid Gaussians.
    """

    def __init__(self, state_dict: dict, device: torch.device, tracks_data: Optional[dict] = None):
        """Initialize a GaussianSet from a state dictionary.

        Args:
            state_dict: Model state dict containing Gaussian parameters
            device: Torch device
            tracks_data: Optional tracks data from sequence_tracks.json
        """
        self.device = device

        # Extract background Gaussians
        self.background = GaussianGroup(
            positions=state_dict["model.gaussians_nodes.background.positions"],
            rotations=state_dict["model.gaussians_nodes.background.rotations"],
            scales=state_dict["model.gaussians_nodes.background.scales"],
            densities=state_dict["model.gaussians_nodes.background.densities"],
            features_albedo=state_dict["model.gaussians_nodes.background.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.background.features_specular"],
            device=device,
        )

        # Extract road Gaussians
        self.road = GaussianGroup(
            positions=state_dict["model.gaussians_nodes.road.positions"],
            rotations=state_dict["model.gaussians_nodes.road.rotations"],
            scales=state_dict["model.gaussians_nodes.road.scales"],
            densities=state_dict["model.gaussians_nodes.road.densities"],
            features_albedo=state_dict["model.gaussians_nodes.road.features_albedo"],
            features_specular=state_dict["model.gaussians_nodes.road.features_specular"],
            device=device,
        )

        # Extract rigid Gaussians if available
        dr_key = "model.gaussians_nodes.dynamic_rigids.positions"
        if dr_key in state_dict and state_dict[dr_key].shape[0] > 0:
            self.rigids = RigidGaussianGroup(
                positions=state_dict["model.gaussians_nodes.dynamic_rigids.positions"],
                rotations=state_dict["model.gaussians_nodes.dynamic_rigids.rotations"],
                scales=state_dict["model.gaussians_nodes.dynamic_rigids.scales"],
                densities=state_dict["model.gaussians_nodes.dynamic_rigids.densities"],
                features_albedo=state_dict["model.gaussians_nodes.dynamic_rigids.features_albedo"],
                features_specular=state_dict["model.gaussians_nodes.dynamic_rigids.features_specular"],
                cuboid_ids=state_dict["model.gaussians_nodes.dynamic_rigids.gaussian_cuboid_ids"],
                tracks_calib={
                    "tracks_delta_q": state_dict["model.gaussians_nodes.dynamic_rigids.tracks_calib.tracks_delta_q"],
                    "tracks_delta_t": state_dict["model.gaussians_nodes.dynamic_rigids.tracks_calib.tracks_delta_t"],
                },
                tracks_data=tracks_data,
                device=device,
            )
        else:
            self.rigids = None

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device, tracks_data: Optional[dict] = None):
        """Load GaussianSet from a checkpoint file.

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
        total_count = bg_count + road_count + rigid_count

        print("Loaded GaussianSet:")
        print(f"  Background: {bg_count:,} Gaussians")
        print(f"  Road: {road_count:,} Gaussians")
        if self.rigids is not None:
            num_unique_cuboids = len(torch.unique(self.rigids.cuboid_ids))
            print(f"  Rigids: {rigid_count:,} Gaussians ({num_unique_cuboids} cuboids)")
        print(f"  Total: {total_count:,} Gaussians")
