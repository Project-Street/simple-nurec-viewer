"""
Rigid Gaussian class for time-varying 3D Gaussian Splatting.

This module provides the RigidGaussian class that extends BaseGaussian
to support time-varying rigid body transforms from trajectory data.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from ...utils.rigid import build_rotation, quaternion_multiply, slerp
from .base import BaseGaussian


class RigidGaussian(BaseGaussian):
    """
    Rigid body Gaussians with time-varying transforms.

    This class extends BaseGaussian to support Gaussians that move over time
    according to trajectory data (e.g., vehicles, pedestrians).
    Each Gaussian is associated with a track ID and can be transformed over time.

    Attributes:
        cuboid_ids: Track/box ID for each Gaussian [N]
        tracks_data: Optional tracks data from datasource_summary.json
                     containing tracks_poses and tracks_timestamps_us
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
        tracks_data: Optional[dict],
        device: torch.device,
    ):
        """
        Initialize a RigidGaussian.

        Args:
            positions: Gaussian centers [N, 3]
            rotations: Gaussian rotations (quaternions) [N, 4]
            scales: Gaussian scales [N, 3]
            densities: Gaussian densities (pre-activation) [N, 1]
            features_albedo: Albedo features [N, 5, 3]
            features_specular: Specular features [N, 45]
            cuboid_ids: Track/box ID for each Gaussian [N]
            tracks_data: Optional tracks data from sequence_tracks.json
            device: Torch device
        """
        super().__init__(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            features_albedo=features_albedo,
            features_specular=features_specular,
            device=device,
        )
        self.cuboid_ids = cuboid_ids.to(device)
        self.tracks_data = tracks_data

    def _load_track_from_usd(self, track_idx: int, timestamp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load track transform from datasource_summary.json with time interpolation.

        Args:
            track_idx: Track index (0-based, matches cuboid_id)
            timestamp: Target timestamp in seconds

        Returns:
            Tuple of (quaternion [4], translation [3])
        """
        if self.tracks_data is None:
            # Return identity transform if no tracks data
            return (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                torch.zeros(3, device=self.device),
            )

        tracks_dict = self.tracks_data.get("tracks_data", {})
        tracks_poses = tracks_dict.get("tracks_poses", [])
        tracks_timestamps_us = tracks_dict.get("tracks_timestamps_us", [])

        # Check if track_idx is valid
        if track_idx >= len(tracks_poses) or track_idx >= len(tracks_timestamps_us):
            # Track index out of range, return identity
            return (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                torch.zeros(3, device=self.device),
            )

        # Get poses and timestamps for this track
        poses = tracks_poses[track_idx]  # List of [N_frames, 7]
        timestamps_us = tracks_timestamps_us[track_idx]  # List of timestamps in microseconds

        # Convert to numpy arrays
        poses = np.array(poses)  # [N_frames, 7]
        timestamps_us = np.array(timestamps_us)  # [N_frames]
        timestamps_s = timestamps_us / 1e6  # Convert to seconds

        # Check if we have poses
        if len(poses) == 0:
            return (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
                torch.zeros(3, device=self.device),
            )

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
        """
        Get base rigid transform for a given timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            Tuple of (quaternions [N, 4], translations [N, 3]) for each Gaussian
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

    def collect(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform features into render-ready Gaussian parameters with rigid transform.

        Args:
            **kwargs: Optional parameters including:
                - timestamp: Optional timestamp for rigid transform (seconds)

        Returns:
            Tuple of (means, quats, scales, opacities, colors)
            - means: Gaussian centers [N, 3]
            - quats: Normalized quaternions [N, 4]
            - scales: Scales (exp activated) [N, 3]
            - opacities: Opacities (sigmoid activated) [N]
            - colors: SH coefficients [N, K, 3]
        """
        timestamp = kwargs.get("timestamp", None)

        # Get base Gaussian parameters using the default implementation
        means, quats, scales, opacities, colors = self._collect_impl()

        # Apply rigid transform if timestamp is provided
        if timestamp is not None:
            # Get base transform for this timestamp
            track_q, track_t = self._get_base_transform(timestamp)

            # Build rotation matrices
            track_R = build_rotation(track_q)  # [N, 3, 3]

            # Apply rotation to positions
            # positions_transformed = (R @ positions.T).T + translation
            positions_transformed = torch.bmm(track_R, means.unsqueeze(-1)).squeeze(-1) + track_t

            # Also rotate the Gaussian orientations
            # new_rotation = quaternion_multiply(track_q, base_rotation)
            rotations_transformed = quaternion_multiply(
                track_q,  # [N, 4]
                self.rotations,  # [N, 4]
            )

            return positions_transformed, rotations_transformed, scales, opacities, colors
        else:
            # No timestamp, return original positions
            return means, quats, scales, opacities, colors
