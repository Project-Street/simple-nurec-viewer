"""
Server subcommand for Simple NuRec Viewer.

This module provides the gRPC server subcommand for remote rendering
of NuRec USDZ files.
"""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.console import Console

import grpc

if not importlib.util.find_spec("simple_nurec_grpc"):
    print("ERROR: Cannot import grpc package. Please install the gRPC package:\n" "   pip install -e grpc/\n")
    raise SystemExit(1)

# Import FTheta distortion parameters
from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType
from simple_nurec_grpc import render_pb2, render_pb2_grpc

from simple_nurec_viewer.core.loader import load_nurec_data
from simple_nurec_viewer.core.rendering import RenderContext, render_frame


@dataclass
class ServerConfig:
    """Configuration for the gRPC rendering server."""

    usdz_path: Path
    """Path to the USDZ file to load."""

    host: str = "0.0.0.0"
    """Host address to bind the server."""

    port: int = 50051
    """Port to listen on."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for rendering ('cuda' or 'cpu')."""

    verbose: bool = False
    """Whether to log camera information for each render request."""


class RenderServicer(render_pb2_grpc.RenderServiceServicer):
    """gRPC servicer implementing the RenderService."""

    def __init__(
        self,
        gaussian_set,
        sky_cubemap,
        device: torch.device,
        verbose: bool = False,
    ):
        self.gaussian_set = gaussian_set
        self.sky_cubemap = sky_cubemap
        # self.world_to_nre = world_to_nre  # Unused, converted by client
        self.device = device
        self.verbose = verbose
        self.console = Console()
        self.render_count = 0
        self.last_traffic_pose: Optional[dict] = None

    def SetTrafficPose(self, request: render_pb2.TrafficPoseRequest, context) -> render_pb2.TrafficPoseResponse:
        """Handle traffic pose update requests."""
        response = render_pb2.TrafficPoseResponse()
        try:
            traffic_type_id = request.traffic_type_id
            if not traffic_type_id:
                raise ValueError("traffic_type_id must be a non-empty string")

            if len(request.pose_4x4) != 16:
                raise ValueError(f"pose_4x4 must have 16 elements, got {len(request.pose_4x4)}")

            pose = np.array(request.pose_4x4, dtype=np.float32).reshape(4, 4)
            if not np.isfinite(pose).all():
                raise ValueError("pose_4x4 contains non-finite values")

            self.last_traffic_pose = {
                "traffic_type_id": traffic_type_id,
                "pose_4x4": pose,
            }
            response.success = True
        except Exception as e:
            response.success = False
            response.error_message = f"{type(e).__name__}: {str(e)}"

        return response

    def Render(self, request: render_pb2.RenderRequest, context) -> render_pb2.RenderResponse:
        """Handle render requests."""
        import time
        import traceback

        start_time = time.perf_counter()
        response = render_pb2.RenderResponse()

        try:
            camera = request.camera

            # Validate camera_to_world matrix
            if len(camera.camera_to_world) != 12:
                raise ValueError(f"camera_to_world must have 12 elements, got {len(camera.camera_to_world)}")

            # Extend to 4x4 and compute view matrix (inverse of T_camera_NRE)
            T_camera_NRE = np.eye(4, dtype=np.float32)
            T_camera_NRE[:3, :4] = np.array(camera.camera_to_world, dtype=np.float32).reshape(3, 4)
            viewmat = np.linalg.inv(T_camera_NRE)  # NRE → camera transform

            # Build intrinsic matrix
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = camera.fx
            K[1, 1] = camera.fy
            K[0, 2] = camera.cx
            K[1, 2] = camera.cy

            resolution = (camera.width, camera.height)

            # Get camera model (default to pinhole)
            camera_model = "pinhole"
            if camera.HasField("camera_model") and camera.camera_model:
                camera_model = camera.camera_model

            # Get timestamp for dynamic scenes
            timestamp = None
            if camera.HasField("time"):
                timestamp = camera.time

            # Build FTheta distortion parameters if provided
            ftheta_coeffs = None
            if camera_model == "ftheta" and FThetaCameraDistortionParameters is not None:
                if camera.HasField("ftheta_params") and camera.ftheta_params is not None:
                    ftheta_params = camera.ftheta_params
                    # Map reference_poly string to enum
                    poly_type_map = {
                        "ANGLE_TO_PIXELDIST": FThetaPolynomialType.ANGLE_TO_PIXELDIST,
                        "PIXELDIST_TO_ANGLE": FThetaPolynomialType.PIXELDIST_TO_ANGLE,
                    }
                    reference_poly = poly_type_map.get(
                        ftheta_params.reference_poly, FThetaPolynomialType.ANGLE_TO_PIXELDIST
                    )

                    # Convert protobuf repeated fields to tuples
                    pixeldist_to_angle_poly = tuple(ftheta_params.pixeldist_to_angle_poly)
                    angle_to_pixeldist_poly = tuple(ftheta_params.angle_to_pixeldist_poly)
                    max_angle = ftheta_params.max_angle
                    linear_cde = tuple(ftheta_params.linear_cde) if len(ftheta_params.linear_cde) > 0 else None

                    ftheta_coeffs = FThetaCameraDistortionParameters(
                        reference_poly=reference_poly,
                        pixeldist_to_angle_poly=pixeldist_to_angle_poly,
                        angle_to_pixeldist_poly=angle_to_pixeldist_poly,
                        max_angle=max_angle,
                        linear_cde=linear_cde,
                    )

            # Log camera information if verbose
            if self.verbose:
                time_str = f"{timestamp:.3f}" if timestamp is not None else "N/A"
                pos = T_camera_NRE[:3, 3]
                self.console.print(
                    f"[dim]📷 Render:[/dim] "
                    f"{camera.width}×{camera.height} | "
                    f"fx={camera.fx:.1f} fy={camera.fy:.1f} cx={camera.cx:.1f} cy={camera.cy:.1f} | "
                    f"model={camera_model} | "
                    f"time={time_str} | "
                    f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                )

            # Create render context
            ctx = RenderContext(
                gaussian_set=self.gaussian_set,
                sky_cubemap=self.sky_cubemap,
                device=self.device,
            )

            # Render frame
            with torch.no_grad():
                image = render_frame(
                    ctx,
                    viewmat,
                    K,
                    resolution,
                    timestamp=timestamp,
                    camera_model=camera_model,
                    ftheta_coeffs=ftheta_coeffs,
                    traffic_pose_override=self.last_traffic_pose,
                )

            # Convert to uint8 bytes
            rgb_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            rgb_data = rgb_uint8.tobytes()

            # Fill response
            response.rgb_image.width = camera.width
            response.rgb_image.height = camera.height
            response.rgb_image.rgb_data = rgb_data
            response.success = True
            response.render_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Increment render counter and clear cache every 20 renders
            self.render_count += 1
            if self.render_count % 20 == 0:
                torch.cuda.empty_cache()
                if self.verbose:
                    self.console.print(f"[dim]🧹 CUDA cache cleared after {self.render_count} renders[/dim]")

        except Exception as e:
            response.success = False
            response.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            response.render_time_ms = (time.perf_counter() - start_time) * 1000.0

        return response


def serve(config: ServerConfig):
    """Start the gRPC server."""
    # Validate USDZ file exists
    if not config.usdz_path.exists():
        print(f"Error: USDZ file not found: {config.usdz_path}")
        return

    # Determine device
    if config.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(config.device)
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # Load NuRec data
    print(f"Loading NuRec data from {config.usdz_path}...")
    data = load_nurec_data(config.usdz_path, device)
    gaussian_set = data.gaussian_set
    sky_cubemap = data.sky_cubemap

    # Create servicer
    servicer = RenderServicer(gaussian_set, sky_cubemap, device, verbose=config.verbose)

    # Start server with increased message size (256 MB for high-res images)
    max_message_length = 256 * 1024 * 1024  # 256 MB
    server_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]
    server = grpc.server(ThreadPoolExecutor(max_workers=10), options=server_options)
    render_pb2_grpc.add_RenderServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()

    console = Console()
    console.print(f"🚀 Server started on [bold cyan]{config.host}:{config.port}[/bold cyan]")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Shutting down server...[/bold yellow]")
        server.stop(0)


def server(
    usdz_path: Path,
    host: str = "0.0.0.0",
    port: int = 50051,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> None:
    """Start gRPC server for remote rendering.

    Args:
        usdz_path: Path to the USDZ file
        host: Host address to bind (default: 0.0.0.0)
        port: Port to listen on (default: 50051)
        device: Device for rendering (default: cuda if available)
        verbose: Enable verbose logging (default: False)
    """
    config = ServerConfig(usdz_path=usdz_path, host=host, port=port, device=device, verbose=verbose)
    serve(config)


__all__ = ["server"]
