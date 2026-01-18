#!/usr/bin/env python3
"""
Rigid Body Transform Utilities

Utility functions for handling rigid body transformations, quaternions,
and trajectory interpolation. Adapted from EasyDrive.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix from quaternion.

    Args:
        r: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    q = r / torch.norm(r, dim=-1, keepdim=True)

    R = torch.zeros((*q.shape[:-1], 3, 3), device=r.device, dtype=r.dtype)

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def slerp(v1: torch.Tensor, v2: torch.Tensor, t: float,
          DOT_THR: float = 0.9995, dim: int = -1) -> torch.Tensor:
    """
    SLERP (Spherical Linear Interpolation) for pytorch tensors.

    Interpolates `v1` to `v2` with scale of `t`.

    Args:
        v1: Start vectors
        v2: End vectors
        t: Interpolation parameter (0 to 1)
        DOT_THR: Threshold for when vectors are too close to parallel
        dim: Feature dimension over which to compute norms

    Returns:
        Interpolated vectors
    """
    # Take the dot product between normalized vectors
    v1_norm = v1 / torch.norm(v1, dim=dim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=dim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(dim)

    # If the vectors are too close, return a simple linear interpolation
    if torch.abs(dot) > DOT_THR:
        res = (1 - t) * v1 + t * v2
    else:  # Else apply SLERP
        # Compute the angle terms we need
        theta = torch.acos(dot)
        theta_t = theta * t
        sin_theta = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # Compute the sine scaling terms for the vectors
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = sin_theta_t / sin_theta

        # Interpolate the vectors
        res = (s1.unsqueeze(dim) * v1) + (s2.unsqueeze(dim) * v2)

    return res


def apply_rigid_transform(
    positions: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    per_point_rotations: torch.Tensor = None
) -> torch.Tensor:
    """
    Apply rigid body transformation to positions.

    Args:
        positions: Input positions [N, 3]
        rotation: Rotation quaternion [4] or rotation matrix [3, 3]
        translation: Translation vector [3]
        per_point_rotations: Optional per-point rotations [N, 4] (quaternions)

    Returns:
        Transformed positions [N, 3]
    """
    # Build rotation matrix if quaternion is provided
    if rotation.dim() == 1:
        R = build_rotation(rotation)  # [3, 3]
    else:
        R = rotation  # [3, 3]

    # Apply rotation: positions @ R.T + translation
    transformed = positions @ R.T + translation

    # If per-point rotations are provided, apply them first
    if per_point_rotations is not None:
        # Build per-point rotation matrices
        R_per_point = build_rotation(per_point_rotations)  # [N, 3, 3]
        # Apply per-point rotation
        transformed = torch.bmm(
            R_per_point,
            positions.unsqueeze(-1)
        ).squeeze(-1) @ R.T + translation

    return transformed


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Quaternions with real part first, as tensor of shape (..., 4).
    """
    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # We produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # Pick the best-conditioned quaternion (with the largest denominator)
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
