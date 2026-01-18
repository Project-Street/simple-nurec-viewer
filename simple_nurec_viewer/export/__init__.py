"""
Camera frame export functionality for NuRec USDZ files.

This module provides tools to export rendered frames from NuRec USDZ files,
including data loading, camera pose computation, and batch rendering.
"""

from .loader import load_camera_data, CameraCalibration, RigTrajectory
from .camera import (
    compute_camera_pose_NRE,
    interpolate_rig_pose,
    build_pinhole_K_from_ftheta,
    build_pinhole_K_from_pinhole,
    get_camera_intrinsics,
    validate_transform_chain,
    CameraPose,
)
from .renderer import render_camera_frame
from .writer import save_rendered_frame, create_output_directories, ExportTask

__all__ = [
    # Data structures
    "CameraCalibration",
    "RigTrajectory",
    "CameraPose",
    "ExportTask",
    # Loader
    "load_camera_data",
    # Camera utilities
    "compute_camera_pose_NRE",
    "interpolate_rig_pose",
    "build_pinhole_K_from_ftheta",
    "build_pinhole_K_from_pinhole",
    "get_camera_intrinsics",
    "validate_transform_chain",
    # Rendering
    "render_camera_frame",
    # Output
    "save_rendered_frame",
    "create_output_directories",
]

# Main export function (implemented in US1)
def export_frames(task: ExportTask) -> None:
    """Export camera frames from NuRec USDZ file.

    This is the main entry point for the export functionality. It loads
    Gaussian data and camera information from the USDZ file, then
    renders and saves frames for all requested cameras and timestamps.

    Args:
        task: ExportTask containing configuration and parameters

    Raises:
        FileNotFoundError: If USDZ file doesn't exist
        ValueError: If required data is missing from USDZ
        RuntimeError: If rendering fails
    """
    import zipfile
    import tempfile
    import json
    from pathlib import Path
    from tqdm import tqdm

    import numpy as np
    import torch
    from simple_nurec_viewer.core.viewer import GaussianSet, SkyCubeMap

    from .loader import load_camera_data
    from .camera import (
        compute_camera_pose_NRE,
        interpolate_rig_pose,
        get_camera_intrinsics,
    )
    from .renderer import render_camera_frame
    from .writer import save_rendered_frame, create_output_directories

    # Validate inputs
    if not task.usdz_path.exists():
        raise FileNotFoundError(f"USDZ file not found: {task.usdz_path}")

    # Load GaussianSet from USDZ
    print(f"Loading GaussianSet from {task.usdz_path}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(task.usdz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        ckpt_path = Path(tmpdir) / "checkpoint.ckpt"
        if not ckpt_path.exists():
            raise ValueError(f"checkpoint.ckpt not found in USDZ file")

        # Load tracks data for rigid body animation from datasource_summary.json
        tracks_data = None
        datasource_path = Path(tmpdir) / "datasource_summary.json"
        if datasource_path.exists():
            print(f"Loading tracks data from datasource_summary.json...")
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

        # Load GaussianSet with tracks_data
        gaussian_set = GaussianSet.from_checkpoint(str(ckpt_path), task.device, tracks_data=tracks_data)

        # Load sky cubemap (optional)
        sky_cubemap = None
        ckpt = torch.load(ckpt_path, map_location=task.device, weights_only=False)
        if "model.background.textures" in ckpt["state_dict"]:
            tex = ckpt["state_dict"]["model.background.textures"].squeeze(0).to(task.device)
            sky_cubemap = SkyCubeMap(tex)
            print("Loaded sky cubemap")
        del ckpt

    # Load camera data
    print("Loading camera data...")
    camera_calibrations, rig_trajectories, world_to_nre = load_camera_data(task.usdz_path)

    # Determine which cameras to export
    if task.cameras is None:
        # Export all cameras (get unique logical names)
        camera_names = list(set(
            calib.logical_sensor_name for calib in camera_calibrations.values()
        ))
        print(f"Exporting all {len(camera_names)} cameras")
    else:
        camera_names = task.cameras
        print(f"Exporting {len(camera_names)} specified cameras")

    # Create output directories
    camera_dirs = create_output_directories(task.output_dir, camera_names)

    # Build mapping from logical name to calibration keys
    logical_name_to_keys = {}
    for key, calib in camera_calibrations.items():
        logical_name = calib.logical_sensor_name
        if logical_name not in logical_name_to_keys:
            logical_name_to_keys[logical_name] = []
        logical_name_to_keys[logical_name].append(key)

    # Count total frames for progress bar
    total_frames = 0
    frame_counts = {}
    for camera_name in camera_names:
        if camera_name not in logical_name_to_keys:
            print(f"Warning: Camera {camera_name} not found in calibrations, skipping")
            continue

        # Get the first calibration key for this camera
        calib_key = logical_name_to_keys[camera_name][0]
        if calib_key not in rig_trajectories.cameras_frame_timestamps_us:
            print(f"Warning: No timestamps found for camera {camera_name}, skipping")
            continue

        timestamps = rig_trajectories.cameras_frame_timestamps_us[calib_key]

        # Apply frame range filter
        start_idx = task.timestamp_start if task.timestamp_start is not None else 0
        end_idx = task.timestamp_end if task.timestamp_end is not None else len(timestamps)
        end_idx = min(end_idx, len(timestamps))

        frame_count = end_idx - start_idx
        total_frames += frame_count
        frame_counts[camera_name] = (start_idx, end_idx)

    print(f"Total frames to export: {total_frames}")

    # Export frames
    with tqdm(total=total_frames, desc="Exporting frames") as pbar:
        for camera_name in camera_names:
            if camera_name not in frame_counts:
                continue

            start_idx, end_idx = frame_counts[camera_name]
            calib_key = logical_name_to_keys[camera_name][0]
            calib = camera_calibrations[calib_key]
            timestamps = rig_trajectories.cameras_frame_timestamps_us[calib_key][start_idx:end_idx]
            camera_dir = camera_dirs[camera_name]

            # Validate camera model type
            if calib.camera_model_type not in ["ftheta", "pinhole"]:
                print(f"Warning: Unsupported camera model type '{calib.camera_model_type}' for {camera_name}, skipping")
                continue

            # Build camera intrinsics matrix based on model type
            try:
                if calib.camera_model_type == "ftheta":
                    K = get_camera_intrinsics(
                        calib.camera_model_type,
                        calib.resolution,
                        calib.principal_point,
                        max_angle=calib.max_angle,
                    )
                elif calib.camera_model_type == "pinhole":
                    # For pinhole, extract focal_length from parameters if available
                    # If not provided, estimate from resolution
                    params = camera_calibrations[calib_key].camera_model_type
                    # Check if original calib data has focal_length
                    focal_length = getattr(calib, 'focal_length', calib.resolution[0] * 0.8)  # Default estimate
                    K = get_camera_intrinsics(
                        calib.camera_model_type,
                        calib.resolution,
                        calib.principal_point,
                        focal_length=focal_length,
                    )
            except ValueError as e:
                print(f"Warning: Failed to build intrinsics for {camera_name}: {e}, skipping")
                continue

            # Apply resolution scale
            scaled_width = int(calib.resolution[0] * task.resolution_scale)
            scaled_height = int(calib.resolution[1] * task.resolution_scale)
            scaled_resolution = (scaled_width, scaled_height)

            # Scale intrinsics if needed
            if task.resolution_scale != 1.0:
                K_scaled = K.copy()
                K_scaled[0, 0] *= task.resolution_scale  # fx
                K_scaled[1, 1] *= task.resolution_scale  # fy
                K_scaled[0, 2] *= task.resolution_scale  # cx
                K_scaled[1, 2] *= task.resolution_scale  # cy
            else:
                K_scaled = K

            for frame_idx, (start_us, end_us) in enumerate(timestamps):
                timestamp_us = (start_us + end_us) // 2

                # Check if file already exists
                output_path = camera_dir / f"{timestamp_us}.jpg"
                if not task.overwrite and output_path.exists():
                    pbar.update(1)
                    continue

                try:
                    # Interpolate rig pose to camera timestamp
                    T_rig_world = interpolate_rig_pose(
                        timestamp_us,
                        rig_trajectories.T_rig_world_timestamps_us,
                        rig_trajectories.T_rig_worlds,
                    )

                    # Compute camera pose in NRE coordinates
                    T_camera_NRE = compute_camera_pose_NRE(
                        calib.T_sensor_rig,
                        T_rig_world,
                        world_to_nre,
                    )

                    # Validate transform chain if debug mode is enabled
                    if task.debug and frame_idx == 0:  # Only validate first frame
                        validate_transform_chain(
                            calib.T_sensor_rig,
                            T_rig_world,
                            world_to_nre,
                            debug=True,
                        )

                    # Compute view matrix
                    viewmat = np.linalg.inv(T_camera_NRE)

                    # Render frame
                    # Convert timestamp from microseconds to seconds for rigid body animation
                    timestamp_s = timestamp_us / 1e6
                    image = render_camera_frame(
                        gaussian_set,
                        sky_cubemap,
                        viewmat,
                        K_scaled,
                        scaled_resolution,
                        task.device,
                        timestamp=timestamp_s,
                    )

                    # Save frame
                    save_rendered_frame(image, output_path)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\nWarning: GPU OOM at {camera_name} frame {frame_idx}, skipping")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                # Update progress
                pbar.set_postfix({"camera": camera_name, "frame": frame_idx})
                pbar.update(1)

    print(f"\nExport completed! Frames saved to: {task.output_dir}")
