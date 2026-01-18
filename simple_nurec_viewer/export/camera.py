"""
Camera pose computation and utilities.

This module provides functions for computing camera poses in the NRE
coordinate system, including rig pose interpolation and camera intrinsics.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

from simple_nurec_viewer.utils.rigid import matrix_to_quaternion, slerp, build_rotation

try:
    from gsplat.cuda._wrapper import (
        FThetaCameraDistortionParameters,
        FThetaPolynomialType,
    )
except ImportError:
    FThetaCameraDistortionParameters = None
    FThetaPolynomialType = None


@dataclass
class CameraPose:
    """Camera pose for rendering.

    Attributes:
        camera_name: Logical camera name (without sequence ID)
        timestamp_us: Frame timestamp in microseconds
        T_camera_NRE: 4x4 transformation matrix from camera to NRE
        resolution: Image resolution as (width, height)
        K: 3x3 pinhole camera intrinsics matrix
        viewmat: 4x4 view matrix (NRE to camera transformation)
    """
    camera_name: str
    timestamp_us: int
    T_camera_NRE: np.ndarray  # [4, 4]
    resolution: Tuple[int, int]
    K: np.ndarray  # [3, 3]
    viewmat: np.ndarray  # [4, 4]


def compute_camera_pose_NRE(
    T_sensor_rig: np.ndarray,
    T_rig_world: np.ndarray,
    world_to_nre: np.ndarray,
) -> np.ndarray:
    """Compute camera pose in NRE coordinate system.

    Coordinate transformation chain: sensor -> rig -> world -> NRE

    Args:
        T_sensor_rig: Sensor to rig transformation [4, 4]
        T_rig_world: Rig to world transformation [4, 4]
        world_to_nre: World to NRE transformation [4, 4]

    Returns:
        T_camera_NRE: Camera to NRE transformation [4, 4]
    """
    # Compose transformations: T_camera_NRE = world_to_nre @ T_rig_world @ T_sensor_rig
    T_camera_NRE = world_to_nre @ T_rig_world @ T_sensor_rig

    return T_camera_NRE


def interpolate_rig_pose(
    camera_timestamp_us: int,
    rig_timestamps_us: list,
    rig_poses: list,
) -> np.ndarray:
    """Interpolate rig pose to camera timestamp.

    Uses SLERP for rotation and linear interpolation for translation.

    Args:
        camera_timestamp_us: Camera frame timestamp in microseconds
        rig_timestamps_us: Rig trajectory timestamps [N]
        rig_poses: Rig trajectory poses [N, 4, 4]

    Returns:
        Interpolated rig pose [4, 4]
    """
    # Convert to seconds for numerical stability
    t = camera_timestamp_us / 1e6
    rig_ts = np.array(rig_timestamps_us) / 1e6

    # Find surrounding rig timestamps
    idx = np.searchsorted(rig_ts, t) - 1
    idx = np.clip(idx, 0, len(rig_ts) - 2)

    # Get surrounding poses and timestamps
    t0, t1 = rig_ts[idx], rig_ts[idx + 1]
    pose0, pose1 = rig_poses[idx], rig_poses[idx + 1]

    # Linear interpolation parameter
    alpha = (t - t0) / (t1 - t0)

    # Interpolate rotation using SLERP
    q0 = matrix_to_quaternion(torch.from_numpy(pose0[:3, :3]))
    q1 = matrix_to_quaternion(torch.from_numpy(pose1[:3, :3]))

    # SLERP interpolation
    q_interp = slerp(q0, q1, alpha)
    R_interp = build_rotation(q_interp).numpy()

    # Linear interpolation for translation
    t_interp = (1 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]

    # Construct interpolated pose matrix
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp
    pose_interp[:3, 3] = t_interp

    return pose_interp


def build_pinhole_K_from_ftheta(
    resolution: Tuple[int, int],
    principal_point: Tuple[float, float],
    max_angle: float,
) -> np.ndarray:
    """Build pinhole intrinsics matrix from ftheta parameters.

    Estimates focal length from max_angle to create an equivalent
    pinhole camera model for rendering.

    Args:
        resolution: Image resolution as (width, height)
        principal_point: Principal point as (cx, cy)
        max_angle: Maximum FOV angle in radians

    Returns:
        Pinhole K matrix [3, 3]
    """
    width, height = resolution
    cx, cy = principal_point

    # Estimate focal length from max_angle
    # For pinhole: tan(max_angle/2) = max_radius / f
    # For ftheta: max_angle maps to max_radius
    max_radius = min(cx, width - cx, cy, height - cy)
    focal_length = max_radius / np.tan(max_angle / 2)

    # Construct K matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    return K


def build_pinhole_K_from_pinhole(
    resolution: Tuple[int, int],
    principal_point: Tuple[float, float],
    focal_length: float,
) -> np.ndarray:
    """Build pinhole intrinsics matrix from pinhole parameters.

    Args:
        resolution: Image resolution as (width, height)
        principal_point: Principal point as (cx, cy)
        focal_length: Focal length in pixels

    Returns:
        Pinhole K matrix [3, 3]
    """
    width, height = resolution
    cx, cy = principal_point

    # Construct K matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    return K


def get_camera_intrinsics(
    camera_model_type: str,
    resolution: Tuple[int, int],
    principal_point: Tuple[float, float],
    **kwargs,
) -> np.ndarray:
    """Get camera intrinsics matrix based on model type.

    Unified interface for constructing intrinsics matrices from different
    camera model types.

    Args:
        camera_model_type: Type of camera model ("ftheta" or "pinhole")
        resolution: Image resolution as (width, height)
        principal_point: Principal point as (cx, cy)
        **kwargs: Additional model-specific parameters
            - For ftheta: max_angle (float)
            - For pinhole: focal_length (float)

    Returns:
        Pinhole K matrix [3, 3]

    Raises:
        ValueError: If camera_model_type is not supported
    """
    if camera_model_type == "ftheta":
        max_angle = kwargs.get("max_angle", 0.0)
        if max_angle == 0.0:
            raise ValueError("max_angle must be provided for ftheta camera model")
        return build_pinhole_K_from_ftheta(resolution, principal_point, max_angle)

    elif camera_model_type == "pinhole":
        focal_length = kwargs.get("focal_length", 0.0)
        if focal_length == 0.0:
            raise ValueError("focal_length must be provided for pinhole camera model")
        return build_pinhole_K_from_pinhole(resolution, principal_point, focal_length)

    else:
        raise ValueError(f"Unsupported camera model type: {camera_model_type}")


def build_ftheta_coeffs(camera_calib: 'CameraCalibration') -> Optional[FThetaCameraDistortionParameters]:
    """Build FTheta distortion coefficients from camera calibration.

    Args:
        camera_calib: CameraCalibration data object containing distortion parameters

    Returns:
        FThetaCameraDistortionParameters for gsplat rendering, or None if gsplat unavailable
    """
    if FThetaCameraDistortionParameters is None or FThetaPolynomialType is None:
        return None

    # Map string to enum
    poly_type_map = {
        "ANGLE_TO_PIXELDIST": FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        "PIXELDIST_TO_ANGLE": FThetaPolynomialType.PIXELDIST_TO_ANGLE,
    }
    reference_poly = poly_type_map.get(
        camera_calib.reference_poly,
        FThetaPolynomialType.ANGLE_TO_PIXELDIST
    )

    return FThetaCameraDistortionParameters(
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=camera_calib.pixeldist_to_angle_poly,
        angle_to_pixeldist_poly=camera_calib.angle_to_pixeldist_poly,
        max_angle=camera_calib.max_angle,
        linear_cde=camera_calib.linear_cde,
    )


def validate_transform_chain(
    T_sensor_rig: np.ndarray,
    T_rig_world: np.ndarray,
    world_to_nre: np.ndarray,
    debug: bool = False,
) -> bool:
    """Validate coordinate transformation chain.

    Verifies that each transformation matrix is valid and checks
    the final transformation: T_camera_NRE = world_to_nre @ T_rig_world @ T_sensor_rig

    Args:
        T_sensor_rig: Sensor to rig transformation [4, 4]
        T_rig_world: Rig to world transformation [4, 4]
        world_to_nre: World to NRE transformation [4, 4]
        debug: If True, print debug information

    Returns:
        True if all transformations are valid
    """
    # Validate matrix shapes
    for name, matrix in [
        ("T_sensor_rig", T_sensor_rig),
        ("T_rig_world", T_rig_world),
        ("world_to_nre", world_to_nre),
    ]:
        if matrix.shape != (4, 4):
            if debug:
                print(f"Error: {name} has invalid shape {matrix.shape}, expected (4, 4)")
            return False

    # Validate bottom row (should be [0, 0, 0, 1] for homogeneous transforms)
    for name, matrix in [
        ("T_sensor_rig", T_sensor_rig),
        ("T_rig_world", T_rig_world),
        ("world_to_nre", world_to_nre),
    ]:
        if not np.allclose(matrix[3, :], [0, 0, 0, 1], atol=1e-6):
            if debug:
                print(f"Warning: {name}[3, :] = {matrix[3, :]}, expected [0, 0, 0, 1]")

    # Compute full transformation
    T_camera_NRE = world_to_nre @ T_rig_world @ T_sensor_rig

    if debug:
        print("\n=== Coordinate Transformation Debug ===")
        print(f"T_sensor_rig:\n{T_sensor_rig}")
        print(f"\nT_rig_world:\n{T_rig_world}")
        print(f"\nworld_to_nre:\n{world_to_nre}")
        print(f"\nT_camera_NRE = world_to_nre @ T_rig_world @ T_sensor_rig:")
        print(f"T_camera_NRE:\n{T_camera_NRE}")
        print(f"\nCamera position in NRE: {T_camera_NRE[:3, 3]}")
        print("=====================================\n")

    return True

