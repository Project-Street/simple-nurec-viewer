"""
Data loading utilities for NuRec USDZ files.

This module provides functions to extract and parse camera calibration
and trajectory data from USDZ archives.
"""

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np


@dataclass
class CameraCalibration:
    """Camera calibration data from datasource_summary.json.

    Attributes:
        logical_sensor_name: Logical camera name (without sequence ID)
        T_sensor_rig: 4x4 transformation matrix from sensor to rig
        camera_model_type: Camera model type ("ftheta" or "pinhole")
        resolution: Image resolution as (width, height)
        principal_point: Principal point as (cx, cy)
        max_angle: Maximum FOV angle in radians (ftheta only)
    """
    logical_sensor_name: str
    T_sensor_rig: np.ndarray  # [4, 4]
    camera_model_type: str
    resolution: Tuple[int, int]
    principal_point: Tuple[float, float]
    max_angle: float


@dataclass
class RigTrajectory:
    """Vehicle rig trajectory data.

    Attributes:
        sequence_id: Sequence identifier
        T_rig_worlds: List of rig-to-world transformation matrices
        T_rig_world_timestamps_us: Rig trajectory timestamps in microseconds
        cameras_frame_timestamps_us: Camera frame timestamps per camera
    """
    sequence_id: str
    T_rig_worlds: List[np.ndarray]  # List of [4, 4] matrices
    T_rig_world_timestamps_us: List[int]
    cameras_frame_timestamps_us: Dict[str, List[List[int]]]


def load_camera_data(
    usdz_path: Path
) -> Tuple[Dict[str, CameraCalibration], RigTrajectory, np.ndarray]:
    """Load camera calibration and trajectory data from USDZ file.

    Args:
        usdz_path: Path to USDZ file

    Returns:
        Tuple of (camera_calibrations, rig_trajectories, world_to_nre)
    """
    # Extract USDZ to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(usdz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Load datasource_summary.json
        datasource_path = Path(tmpdir) / "datasource_summary.json"
        if not datasource_path.exists():
            raise ValueError("datasource_summary.json not found in USDZ file")

        with open(datasource_path, "r") as f:
            data = json.load(f)

        rig_traj_data = data.get("rig_trajectories", {})
        if not rig_traj_data:
            raise ValueError("rig_trajectories not found in datasource_summary.json")

        # Extract camera calibrations
        camera_calibrations_raw = rig_traj_data.get("camera_calibrations", {})
        camera_calibrations = {}

        for camera_key, calib_data in camera_calibrations_raw.items():
            camera_calibrations[camera_key] = CameraCalibration(
                logical_sensor_name=calib_data["logical_sensor_name"],
                T_sensor_rig=np.array(calib_data["T_sensor_rig"]),
                camera_model_type=calib_data["camera_model"]["type"],
                resolution=tuple(calib_data["camera_model"]["parameters"]["resolution"]),
                principal_point=tuple(calib_data["camera_model"]["parameters"]["principal_point"]),
                max_angle=calib_data["camera_model"]["parameters"].get("max_angle", 0.0),
            )

        # Extract rig trajectories (use first sequence)
        rig_trajectories_raw = rig_traj_data.get("rig_trajectories", [])
        if not rig_trajectories_raw:
            raise ValueError("No rig trajectories found in datasource_summary.json")

        rig_data = rig_trajectories_raw[0]
        rig_trajectories = RigTrajectory(
            sequence_id=rig_data["sequence_id"],
            T_rig_worlds=[np.array(T) for T in rig_data["T_rig_worlds"]],
            T_rig_world_timestamps_us=rig_data["T_rig_world_timestamps_us"],
            cameras_frame_timestamps_us=rig_data["cameras_frame_timestamps_us"],
        )

        # Extract world_to_nre transformation
        world_to_nre_data = rig_traj_data.get("world_to_nre", {})
        if not world_to_nre_data:
            raise ValueError("world_to_nre not found in datasource_summary.json")

        world_to_nre = np.array(world_to_nre_data["matrix"])

        return camera_calibrations, rig_trajectories, world_to_nre
