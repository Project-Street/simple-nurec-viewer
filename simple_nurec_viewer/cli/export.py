"""
Export subcommand for Simple NuRec Viewer.

This module provides the export subcommand for exporting camera frames
from NuRec USDZ files.
"""

from pathlib import Path
from typing import List, Optional

import torch


def export(
    usdz_path: Path,
    output_dir: Path = Path("outputs/"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resolution_scale: float = 1.0,
    cameras: Optional[List[str]] = None,
    timestamp_start: Optional[int] = None,
    timestamp_end: Optional[int] = None,
    overwrite: bool = True,
    debug: bool = False,
) -> None:
    """Export camera frames from a NuRec USDZ file.

    Args:
        usdz_path: Path to the USDZ file
        output_dir: Output directory for exported frames (default: outputs/)
        device: Device to use for rendering (default: cuda if available, else cpu)
        resolution_scale: Resolution scale factor (default: 1.0, must be in (0, 1.0])
        cameras: List of camera names to export (default: all cameras)
        timestamp_start: Start frame index (default: 0)
        timestamp_end: End frame index (default: last frame)
        overwrite: Overwrite existing files (default: True)
        debug: Enable debug output for coordinate transformations (default: False)
    """
    # Validate resolution scale
    if resolution_scale <= 0 or resolution_scale > 1.0:
        print(f"Error: Resolution scale must be in (0, 1.0], got {resolution_scale}")
        return

    # Validate USDZ file exists
    if not usdz_path.exists():
        print(f"Error: USDZ file not found: {usdz_path}")
        return

    # Convert device string to torch.device
    torch_device = torch.device(device)

    # Import here to avoid slow imports if --help is used
    from ..export import ExportTask, export_frames

    # Create export task
    task = ExportTask(
        usdz_path=usdz_path,
        output_dir=output_dir,
        cameras=cameras,
        resolution_scale=resolution_scale,
        device=torch_device,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        overwrite=overwrite,
        debug=debug,
    )

    # Run export
    try:
        export_frames(task)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")


__all__ = ["export"]
