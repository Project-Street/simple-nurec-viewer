"""
Rigid Gaussian class for time-varying 3D Gaussian Splatting.

This module provides the RigidGaussian class that extends BaseGaussian
to support time-varying rigid body transforms from trajectory data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...utils.rigid import build_rotation, matrix_to_quaternion, quaternion_multiply, slerp
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
        dynamic_rigids_track_mapping: Optional[list] = None,
        device: torch.device = torch.device("cuda"),
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
            dynamic_rigids_track_mapping: Optional list mapping cuboid_id to track name
                                          (from model._extra_state['obj_track_ids']['dynamic_rigids'])
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

        # Build mapping from cuboid_id to track_idx
        # The correct flow is:
        #   1. gaussian_cuboid_ids value (e.g., 2) -> index into dynamic_rigids_track_mapping
        #   2. dynamic_rigids_track_mapping[2] -> track name (e.g., '2@scene:...')
        #   3. track name -> index in tracks_id array -> actual track_idx
        self.cuboid_to_track_idx = {}
        if tracks_data is not None and dynamic_rigids_track_mapping is not None:
            tracks_id_list = tracks_data.get("tracks_data", {}).get("tracks_id", [])
            # Build map: track_name (e.g., '2@...') -> track_idx (index in tracks_id array)
            track_name_to_idx = {name: idx for idx, name in enumerate(tracks_id_list)}
            # Build map: cuboid_id (which is index into dynamic_rigids_track_mapping) -> track_idx
            for idx, track_name in enumerate(dynamic_rigids_track_mapping):
                if track_name in track_name_to_idx:
                    self.cuboid_to_track_idx[idx] = track_name_to_idx[track_name]

        # Preprocess and normalize track poses: convert quaternion from xyzw to wxyz format
        # This avoids repeated conversions during runtime
        if tracks_data is not None:
            tracks_dict = tracks_data.get("tracks_data", {})
            tracks_poses = tracks_dict.get("tracks_poses", [])

            # Convert all track poses from xyzw to wxyz quaternion format
            # Original format: [x, y, z, qx, qy, qz, qw]
            # Target format: [x, y, z, qw, qx, qy, qz]
            for track_idx in range(len(tracks_poses)):
                poses = np.array(tracks_poses[track_idx])  # [N_frames, 7]
                if len(poses) > 0:
                    # Swap quaternion components: [qx, qy, qz, qw] -> [qw, qx, qy, qz]
                    # poses[:, 3:7] is [qx, qy, qz, qw], need to reorder to [qw, qx, qy, qz]
                    q_xyzw = poses[:, 3:7]  # [N, 4] in xyzw format
                    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # [N, 4] in wxyz format
                    poses[:, 3:7] = q_wxyz
                    tracks_poses[track_idx] = poses.tolist()

            # Update tracks_data with converted poses
            tracks_dict["tracks_poses"] = tracks_poses
            tracks_data["tracks_data"] = tracks_dict

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
        # Note: poses are now in wxyz format after preprocessing in __init__
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
        # pose format: [x, y, z, qw, qx, qy, qz] (already converted in __init__)
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

    def _resolve_override_track_indices(self, object_id: str) -> List[int]:
        """Resolve object_id to track indices (cuboid ids)."""
        if not object_id:
            return []

        resolved = set()

        if self.tracks_data is not None:
            tracks_dict = self.tracks_data.get("tracks_data", {})
            tracks_id = tracks_dict.get("tracks_id", [])

            for idx, track_id in enumerate(tracks_id):
                if str(track_id) == object_id:
                    resolved.add(idx)

        if not resolved:
            try:
                resolved.add(int(object_id))
            except (TypeError, ValueError):
                pass

        return sorted(resolved)

    def _build_override_map(self, traffic_pose_override: Optional[dict]) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Build an override map from traffic pose payload."""
        if not traffic_pose_override:
            return None

        object_id = traffic_pose_override.get("object_id")
        pose_4x4 = traffic_pose_override.get("pose_4x4")
        if not object_id or pose_4x4 is None:
            return None

        pose_tensor = torch.as_tensor(pose_4x4, device=self.device, dtype=torch.float32)
        if pose_tensor.shape != (4, 4):
            return None

        rotation = pose_tensor[:3, :3]
        translation = pose_tensor[:3, 3]
        quat = matrix_to_quaternion(rotation)

        track_indices = self._resolve_override_track_indices(object_id)
        if not track_indices:
            return None

        return {track_idx: (quat, translation) for track_idx in track_indices}

    def _get_base_transform(
        self, timestamp: float, override_map: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get base rigid transform for a given timestamp.

        Args:
            timestamp: Target timestamp in seconds
            override_map: Optional per-track override map

        Returns:
            Tuple of (quaternions [N, 4], translations [N, 3]) for each Gaussian
        """
        # Build mapping from cuboid_id to its transform for this timestamp
        cuboid_to_transform = {}

        for cuboid_id in torch.unique(self.cuboid_ids):
            # Map cuboid_id to track_idx using the correct mapping
            # cuboid_id -> track_name -> track_idx in tracks_id array
            track_idx = self.cuboid_to_track_idx.get(int(cuboid_id), int(cuboid_id))

            # Load track transform from JSON with interpolation
            if override_map is not None and track_idx in override_map:
                q, t = override_map[track_idx]
            else:
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
                - traffic_pose_override: Optional traffic pose override payload

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

        traffic_pose_override = kwargs.get("traffic_pose_override", None)
        override_map = self._build_override_map(traffic_pose_override)

        # Apply rigid transform if timestamp or override is provided
        if timestamp is not None or override_map is not None:
            # Get base transform for this timestamp
            effective_timestamp = 0.0 if timestamp is None else timestamp
            track_q, track_t = self._get_base_transform(effective_timestamp, override_map=override_map)

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
