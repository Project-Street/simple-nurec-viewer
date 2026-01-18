"""
CLI interface for camera frame export functionality.

This module provides the command-line interface for exporting camera frames
from NuRec USDZ files.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    """Main CLI entry point for camera frame export."""
    parser = argparse.ArgumentParser(
        description="Export camera frames from NuRec USDZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "usdz",
        type=Path,
        help="Path to the USDZ file",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("outputs/"),
        help="Output directory",
    )

    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for rendering",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        dest="resolution_scale",
        help="Resolution scale factor (0.0, 1.0]",
    )

    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="List of camera names to export (default: all cameras)",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        dest="timestamp_start",
        help="Start frame index (default: 0)",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        dest="timestamp_end",
        help="End frame index (default: last frame)",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Skip existing files instead of overwriting",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for coordinate transformations",
    )

    args = parser.parse_args()

    # Validate resolution scale
    if args.resolution_scale <= 0 or args.resolution_scale > 1.0:
        parser.error(f"Resolution scale must be in (0, 1.0], got {args.resolution_scale}")

    # Validate USDZ file exists
    if not args.usdz.exists():
        parser.error(f"USDZ file not found: {args.usdz}")

    # Convert device string to torch.device
    device = torch.device(args.device)

    # Import here to avoid slow imports if --help is used
    from simple_nurec_viewer.export import export_frames, ExportTask

    # Create export task
    task = ExportTask(
        usdz_path=args.usdz,
        output_dir=args.output_dir,
        cameras=args.cameras if args.cameras else None,
        resolution_scale=args.resolution_scale,
        device=device,
        timestamp_start=args.timestamp_start,
        timestamp_end=args.timestamp_end,
        overwrite=args.overwrite,
        debug=args.debug,
    )

    # Run export
    try:
        export_frames(task)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
