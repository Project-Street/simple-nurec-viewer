"""
Rigid Gaussian class for time-varying 3D Gaussian Splatting.

This module provides the RigidGaussian class that extends BaseGaussian
to support time-varying rigid body transforms from trajectory data.
"""

import logging
from typing import Optional, Tuple

import torch
import time
from ...utils.rigid import build_rotation, matrix_to_quaternion, quaternion_multiply, slerp
from .base import BaseGaussian


logger = logging.getLogger(__name__)


def _slerp_batch(v1: torch.Tensor, v2: torch.Tensor, t: torch.Tensor, dot_thr: float = 0.9995) -> torch.Tensor:
    """Vectorized SLERP for tensors shaped ``[N, 4]`` and interpolation weights ``[N]``."""
    dot = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    linear_mask = torch.abs(dot) > dot_thr

    result = (1.0 - t.unsqueeze(-1)) * v1 + t.unsqueeze(-1) * v2
    spherical_mask = ~linear_mask
    if spherical_mask.any():
        theta = torch.acos(torch.clamp(dot[spherical_mask], -1.0, 1.0))
        theta_t = theta * t[spherical_mask]
        sin_theta = torch.sin(theta)
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = torch.sin(theta_t) / sin_theta
        result[spherical_mask] = s1.unsqueeze(-1) * v1[spherical_mask] + s2.unsqueeze(-1) * v2[spherical_mask]
    return result


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

        # Naming:
        # - object_track_id: string id from sequence_tracks_dynamic["tracks_id"]
        # - sequence_track_index: dense row index into sequence_tracks_dynamic tables
        # - cuboid_index: index from checkpoint dynamic_rigids_track_mapping / gaussian_cuboid_ids
        self._sequence_track_index_by_object_track_id: dict[str, int] = {}
        self._object_track_id_by_sequence_track_index: list[str] = []
        self._sequence_track_index_per_gaussian: Optional[torch.Tensor] = None
        self._sequence_track_count = 0
        self._sequence_track_has_gaussian_mask: Optional[torch.Tensor] = None
        if tracks_data is not None and dynamic_rigids_track_mapping is not None:
            object_track_ids = tracks_data.get("tracks_data", {}).get("tracks_id", [])
            self._object_track_id_by_sequence_track_index = [str(object_track_id) for object_track_id in object_track_ids]
            self._sequence_track_index_by_object_track_id = {
                str(object_track_id): sequence_track_index
                for sequence_track_index, object_track_id in enumerate(object_track_ids)
            }
            self._sequence_track_count = len(object_track_ids)
            sequence_track_index_per_cuboid = []
            for cuboid_index, object_track_id in enumerate(dynamic_rigids_track_mapping):
                object_track_id_str = str(object_track_id)
                if object_track_id_str not in self._sequence_track_index_by_object_track_id:
                    raise KeyError(
                        f"Missing track data for cuboid_index={cuboid_index}, object_track_id={object_track_id_str}"
                    )
                sequence_track_index_per_cuboid.append(
                    self._sequence_track_index_by_object_track_id[object_track_id_str]
                )
            sequence_track_index_per_cuboid_tensor = torch.tensor(
                sequence_track_index_per_cuboid,
                device=device,
                dtype=torch.long,
            )
            cuboid_index_per_gaussian = self.cuboid_ids.to(torch.long)
            if cuboid_index_per_gaussian.numel() > 0:
                if (
                    cuboid_index_per_gaussian.min().item() < 0
                    or cuboid_index_per_gaussian.max().item() >= sequence_track_index_per_cuboid_tensor.numel()
                ):
                    raise IndexError("gaussian_cuboid_ids contains out-of-range cuboid indices")
            self._sequence_track_index_per_gaussian = sequence_track_index_per_cuboid_tensor[cuboid_index_per_gaussian]
            self._sequence_track_has_gaussian_mask = torch.zeros(self._sequence_track_count, device=device, dtype=torch.bool)
            self._sequence_track_has_gaussian_mask[self._sequence_track_index_per_gaussian] = True

        # Preprocess and normalize track poses: convert quaternion from xyzw to wxyz format
        # This avoids repeated conversions during runtime
        self._track_pose_table: Optional[torch.Tensor] = None
        self._track_timestamp_table_s: Optional[torch.Tensor] = None
        self._track_frame_counts: Optional[torch.Tensor] = None
        self._gaussian_track_pose_table: Optional[torch.Tensor] = None
        self._gaussian_track_timestamp_table_s: Optional[torch.Tensor] = None
        self._gaussian_track_frame_counts: Optional[torch.Tensor] = None
        if tracks_data is not None:
            tracks_dict = tracks_data.get("tracks_data", {})
            tracks_poses = tracks_dict.get("tracks_poses", [])
            tracks_timestamps_us = tracks_dict.get("tracks_timestamps_us", [])

            # Convert all track poses from xyzw to wxyz quaternion format
            # Original format: [x, y, z, qx, qy, qz, qw]
            # Target format: [x, y, z, qw, qx, qy, qz]
            for track_idx in range(len(tracks_poses)):
                poses = tracks_poses[track_idx]
                if poses:
                    pose_tensor = torch.as_tensor(poses, dtype=torch.float32)  # [N_frames, 7]
                    q_xyzw = pose_tensor[:, 3:7]  # [N, 4] in xyzw format
                    pose_tensor[:, 3:7] = q_xyzw[:, [3, 0, 1, 2]]  # [N, 4] in wxyz format
                    tracks_poses[track_idx] = pose_tensor.tolist()

            # Update tracks_data with converted poses
            tracks_dict["tracks_poses"] = tracks_poses
            tracks_data["tracks_data"] = tracks_dict

            if len(tracks_poses) != len(tracks_timestamps_us):
                raise ValueError("tracks_poses and tracks_timestamps_us length mismatch")
            if tracks_poses:
                max_track_len = max(len(track_pose) for track_pose in tracks_poses)
                track_pose_table = torch.zeros((len(tracks_poses), max_track_len, 7), device=device, dtype=torch.float32)
                track_timestamp_table_s = torch.zeros((len(tracks_timestamps_us), max_track_len), device=device, dtype=torch.float32)
                track_frame_counts = torch.zeros(len(tracks_poses), device=device, dtype=torch.long)
                for track_idx, (track_pose_sequence, track_timestamp_sequence) in enumerate(zip(tracks_poses, tracks_timestamps_us)):
                    if len(track_pose_sequence) != len(track_timestamp_sequence):
                        raise ValueError(f"Track {track_idx} pose/timestamp length mismatch")
                    if not track_pose_sequence:
                        raise ValueError(f"Track {track_idx} has no poses")
                    frame_count = len(track_pose_sequence)
                    track_pose_table[track_idx, :frame_count] = torch.as_tensor(
                        track_pose_sequence,
                        device=device,
                        dtype=torch.float32,
                    )
                    track_timestamp_table_s[track_idx, :frame_count] = (
                        torch.as_tensor(track_timestamp_sequence, device=device, dtype=torch.float32) / 1e6
                    )
                    track_frame_counts[track_idx] = frame_count
                self._track_pose_table = track_pose_table
                self._track_timestamp_table_s = track_timestamp_table_s
                self._track_frame_counts = track_frame_counts
                if self._sequence_track_index_per_gaussian is not None:
                    self._gaussian_track_pose_table = track_pose_table[self._sequence_track_index_per_gaussian]
                    self._gaussian_track_timestamp_table_s = track_timestamp_table_s[self._sequence_track_index_per_gaussian]
                    self._gaussian_track_frame_counts = track_frame_counts[self._sequence_track_index_per_gaussian]

        self.tracks_data = tracks_data

    def _sample_transforms_for_tracks(
        self,
        sequence_track_indices: torch.Tensor,
        timestamp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample rigid transforms for the given sequence-track indices in parallel."""
        if self._track_pose_table is None or self._track_timestamp_table_s is None or self._track_frame_counts is None:
            raise RuntimeError("Rigid track tables are not initialized")

        sequence_track_indices = sequence_track_indices.to(device=self.device, dtype=torch.long)
        if sequence_track_indices.numel() == 0:
            return (
                torch.empty((0, 4), device=self.device, dtype=torch.float32),
                torch.empty((0, 3), device=self.device, dtype=torch.float32),
            )

        sampled_track_poses = self._track_pose_table[sequence_track_indices]
        sampled_track_timestamps_s = self._track_timestamp_table_s[sequence_track_indices]
        sampled_track_frame_counts = self._track_frame_counts[sequence_track_indices]
        query_timestamps_s = torch.full((sequence_track_indices.shape[0], 1), float(timestamp), device=self.device, dtype=torch.float32)

        insertion_indices = torch.searchsorted(sampled_track_timestamps_s.contiguous(), query_timestamps_s, right=False).squeeze(-1)
        insertion_indices = torch.clamp(insertion_indices, min=1)
        insertion_indices = torch.minimum(insertion_indices, sampled_track_frame_counts - 1)

        first_frame_indices = torch.zeros_like(insertion_indices)
        last_frame_indices = sampled_track_frame_counts - 1
        first_timestamps_s = sampled_track_timestamps_s[:, 0]
        last_timestamps_s = sampled_track_timestamps_s.gather(1, last_frame_indices.unsqueeze(-1)).squeeze(-1)

        use_first_frame_mask = query_timestamps_s.squeeze(-1) <= first_timestamps_s
        use_last_frame_mask = query_timestamps_s.squeeze(-1) >= last_timestamps_s

        previous_frame_indices = insertion_indices - 1
        next_frame_indices = insertion_indices
        previous_poses = sampled_track_poses.gather(
            1,
            previous_frame_indices[:, None, None].expand(-1, 1, 7),
        ).squeeze(1)
        next_poses = sampled_track_poses.gather(
            1,
            next_frame_indices[:, None, None].expand(-1, 1, 7),
        ).squeeze(1)
        previous_timestamps_s = sampled_track_timestamps_s.gather(1, previous_frame_indices.unsqueeze(-1)).squeeze(-1)
        next_timestamps_s = sampled_track_timestamps_s.gather(1, next_frame_indices.unsqueeze(-1)).squeeze(-1)

        interpolation_denominator = torch.where(
            next_timestamps_s != previous_timestamps_s,
            next_timestamps_s - previous_timestamps_s,
            torch.ones_like(next_timestamps_s),
        )
        interpolation_alpha = (query_timestamps_s.squeeze(-1) - previous_timestamps_s) / interpolation_denominator
        interpolation_alpha = torch.where(
            next_timestamps_s != previous_timestamps_s,
            interpolation_alpha,
            torch.zeros_like(interpolation_alpha),
        )
        interpolation_alpha = torch.clamp(interpolation_alpha, 0.0, 1.0)

        translations = previous_poses[:, :3] + interpolation_alpha.unsqueeze(-1) * (next_poses[:, :3] - previous_poses[:, :3])
        quaternions = _slerp_batch(previous_poses[:, 3:7], next_poses[:, 3:7], interpolation_alpha)

        first_poses = sampled_track_poses.gather(1, first_frame_indices[:, None, None].expand(-1, 1, 7)).squeeze(1)
        last_poses = sampled_track_poses.gather(1, last_frame_indices[:, None, None].expand(-1, 1, 7)).squeeze(1)

        translations = torch.where(use_first_frame_mask.unsqueeze(-1), first_poses[:, :3], translations)
        translations = torch.where(use_last_frame_mask.unsqueeze(-1), last_poses[:, :3], translations)
        quaternions = torch.where(use_first_frame_mask.unsqueeze(-1), first_poses[:, 3:7], quaternions)
        quaternions = torch.where(use_last_frame_mask.unsqueeze(-1), last_poses[:, 3:7], quaternions)
        return quaternions, translations

    def _sample_transforms_for_all_gaussians(self, timestamp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample rigid transforms for the fixed per-Gaussian track assignment."""

        sampled_track_poses = self._gaussian_track_pose_table
        sampled_track_timestamps_s = self._gaussian_track_timestamp_table_s
        sampled_track_frame_counts = self._gaussian_track_frame_counts
        query_timestamps_s = torch.full(
            (sampled_track_frame_counts.shape[0], 1),
            float(timestamp),
            device=self.device,
            dtype=torch.float32,
        )

        insertion_indices = torch.searchsorted(sampled_track_timestamps_s.contiguous(), query_timestamps_s, right=False).squeeze(-1)
        insertion_indices = torch.clamp(insertion_indices, min=1)
        insertion_indices = torch.minimum(insertion_indices, sampled_track_frame_counts - 1)

        first_frame_indices = torch.zeros_like(insertion_indices)
        last_frame_indices = sampled_track_frame_counts - 1
        first_timestamps_s = sampled_track_timestamps_s[:, 0]
        last_timestamps_s = sampled_track_timestamps_s.gather(1, last_frame_indices.unsqueeze(-1)).squeeze(-1)

        use_first_frame_mask = query_timestamps_s.squeeze(-1) <= first_timestamps_s
        use_last_frame_mask = query_timestamps_s.squeeze(-1) >= last_timestamps_s

        previous_frame_indices = insertion_indices - 1
        next_frame_indices = insertion_indices
        previous_poses = sampled_track_poses.gather(
            1,
            previous_frame_indices[:, None, None].expand(-1, 1, 7),
        ).squeeze(1)
        next_poses = sampled_track_poses.gather(
            1,
            next_frame_indices[:, None, None].expand(-1, 1, 7),
        ).squeeze(1)
        previous_timestamps_s = sampled_track_timestamps_s.gather(1, previous_frame_indices.unsqueeze(-1)).squeeze(-1)
        next_timestamps_s = sampled_track_timestamps_s.gather(1, next_frame_indices.unsqueeze(-1)).squeeze(-1)

        interpolation_denominator = torch.where(
            next_timestamps_s != previous_timestamps_s,
            next_timestamps_s - previous_timestamps_s,
            torch.ones_like(next_timestamps_s),
        )
        interpolation_alpha = (query_timestamps_s.squeeze(-1) - previous_timestamps_s) / interpolation_denominator
        interpolation_alpha = torch.where(
            next_timestamps_s != previous_timestamps_s,
            interpolation_alpha,
            torch.zeros_like(interpolation_alpha),
        )
        interpolation_alpha = torch.clamp(interpolation_alpha, 0.0, 1.0)

        translations = previous_poses[:, :3] + interpolation_alpha.unsqueeze(-1) * (next_poses[:, :3] - previous_poses[:, :3])
        quaternions = _slerp_batch(previous_poses[:, 3:7], next_poses[:, 3:7], interpolation_alpha)

        first_poses = sampled_track_poses.gather(1, first_frame_indices[:, None, None].expand(-1, 1, 7)).squeeze(1)
        last_poses = sampled_track_poses.gather(1, last_frame_indices[:, None, None].expand(-1, 1, 7)).squeeze(1)

        translations = torch.where(use_first_frame_mask.unsqueeze(-1), first_poses[:, :3], translations)
        translations = torch.where(use_last_frame_mask.unsqueeze(-1), last_poses[:, :3], translations)
        quaternions = torch.where(use_first_frame_mask.unsqueeze(-1), first_poses[:, 3:7], quaternions)
        quaternions = torch.where(use_last_frame_mask.unsqueeze(-1), last_poses[:, 3:7], quaternions)
        return quaternions, translations

    def _build_override_map(
        self, traffic_pose_override: Optional[dict]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Parse traffic pose payload into ``(track_indices, quaternions, translations)`` tensors."""
        if traffic_pose_override is None:
            return None

        override_object_track_ids = traffic_pose_override["tracks_id"]
        override_poses = torch.as_tensor(traffic_pose_override["poses_4x4"], device=self.device, dtype=torch.float32)
        override_poses = override_poses.reshape(len(override_object_track_ids), 4, 4)

        override_sequence_track_indices = torch.tensor(
            [
                self._sequence_track_index_by_object_track_id.get(object_track_id, -1)
                for object_track_id in override_object_track_ids
            ],
            device=self.device,
            dtype=torch.long,
        )
        valid_sequence_track_mask = override_sequence_track_indices >= 0
        if not valid_sequence_track_mask.any():
            return None

        override_sequence_track_indices = override_sequence_track_indices[valid_sequence_track_mask]
        override_poses = override_poses[valid_sequence_track_mask]
        if self._sequence_track_has_gaussian_mask is None:
            raise RuntimeError("Rigid track mappings are not initialized")

        valid_gaussian_track_mask = self._sequence_track_has_gaussian_mask[override_sequence_track_indices]
        if not valid_gaussian_track_mask.any():
            return None
        override_sequence_track_indices = override_sequence_track_indices[valid_gaussian_track_mask]
        override_poses = override_poses[valid_gaussian_track_mask]

        override_quaternions = matrix_to_quaternion(override_poses[:, :3, :3])
        override_translations = override_poses[:, :3, 3]
        return (
            override_sequence_track_indices,
            override_quaternions,
            override_translations,
        )

    def _get_base_transform(
        self, timestamp: float, override_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get base rigid transform for a given timestamp.

        Args:
            timestamp: Target timestamp in seconds
            override_map: Optional per-track override map

        Returns:
            Tuple of (quaternions [N, 4], translations [N, 3]) for each Gaussian
        """
        base_quaternions, base_translations = self._sample_transforms_for_all_gaussians(timestamp)
        if override_map is not None:
            override_sequence_track_indices, override_quaternions, override_translations = override_map
 
            override_quaternion_table = torch.zeros((self._sequence_track_count, 4), device=self.device, dtype=torch.float32)
            override_quaternion_table[:, 0] = 1.0
            override_translation_table = torch.zeros((self._sequence_track_count, 3), device=self.device, dtype=torch.float32)
            sequence_track_has_override = torch.zeros(self._sequence_track_count, device=self.device, dtype=torch.bool)

            override_quaternion_table[override_sequence_track_indices] = override_quaternions
            override_translation_table[override_sequence_track_indices] = override_translations
            sequence_track_has_override[override_sequence_track_indices] = True

            gaussian_has_override = sequence_track_has_override[self._sequence_track_index_per_gaussian]
            base_quaternions = torch.where(
                gaussian_has_override.unsqueeze(-1),
                override_quaternion_table[self._sequence_track_index_per_gaussian],
                base_quaternions,
            )
            base_translations = torch.where(
                gaussian_has_override.unsqueeze(-1),
                override_translation_table[self._sequence_track_index_per_gaussian],
                base_translations,
            )

        return base_quaternions, base_translations

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

        active_gaussian_mask = None
        if traffic_pose_override is not None:
            if self._sequence_track_index_per_gaussian is None:
                raise RuntimeError("Rigid track mappings are not initialized")
            active_sequence_track_indices = torch.tensor(
                [
                    self._sequence_track_index_by_object_track_id.get(track_id, -1)
                    for track_id in traffic_pose_override["tracks_id"]
                ],
                device=self.device,
                dtype=torch.long,
            )
            active_sequence_track_indices = active_sequence_track_indices[
                active_sequence_track_indices >= 0
            ]
            active_gaussian_mask = torch.isin(
                self._sequence_track_index_per_gaussian,
                active_sequence_track_indices,
            )
            means = means[active_gaussian_mask]
            quats = quats[active_gaussian_mask]
            scales = scales[active_gaussian_mask]
            opacities = opacities[active_gaussian_mask]
            colors = colors[active_gaussian_mask]
            base_rotations = self.rotations[active_gaussian_mask]
        else:
            base_rotations = self.rotations

        override_map = self._build_override_map(traffic_pose_override)

        # Apply rigid transform if timestamp or override is provided
        if timestamp is not None or override_map is not None:
            # Get base transform for this timestamp
            effective_timestamp = 0.0 if timestamp is None else timestamp
            rigid_quaternions, rigid_translations = self._get_base_transform(
                effective_timestamp,
                override_map=override_map,
            )
            if active_gaussian_mask is not None:
                rigid_quaternions = rigid_quaternions[active_gaussian_mask]
                rigid_translations = rigid_translations[active_gaussian_mask]

            # Build rotation matrices
            rigid_rotations = build_rotation(rigid_quaternions)  # [N, 3, 3]

            # Apply rotation to positions
            # positions_transformed = (R @ positions.T).T + translation
            positions_transformed = torch.bmm(rigid_rotations, means.unsqueeze(-1)).squeeze(-1) + rigid_translations

            # Also rotate the Gaussian orientations
            # new_rotation = quaternion_multiply(track_q, base_rotation)
            rotations_transformed = quaternion_multiply(
                rigid_quaternions,  # [N, 4]
                base_rotations,  # [N, 4]
            )

            return positions_transformed, rotations_transformed, scales, opacities, colors
        else:
            # No timestamp, return original positions
            return means, quats, scales, opacities, colors
