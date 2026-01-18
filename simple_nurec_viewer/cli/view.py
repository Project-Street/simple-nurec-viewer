"""
View subcommand for Simple NuRec Viewer.

This module provides the view subcommand for interactively viewing
NuRec USDZ files using 3D Gaussian Splatting.
"""

import time
from pathlib import Path

import nerfview
import torch
import viser

from ..core.loader import load_nurec_data
from ..core.viewer import add_camera_trajectories, render_fn


def view(
    usdz_path: Path,
    port: int = 8080,
    device: str = "cuda",
) -> None:
    """View a NuRec USDZ file interactively.

    Args:
        usdz_path: Path to the USDZ file
        port: Port for the viewer server (default: 8080)
        device: Device to use for rendering (default: "cuda")
    """
    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but not available. Use --device cpu or install CUDA.")
        return

    torch_device = torch.device(device)

    # Load NuRec data
    print(f"\nLoading NuRec data from {usdz_path}")
    print("=" * 60)
    try:
        data = load_nurec_data(usdz_path, torch_device)
    except FileNotFoundError:
        print(f"Error: File not found: {usdz_path}")
        return
    except Exception as e:
        print(f"Error: Failed to load NuRec data: {e}")
        return
    print("=" * 60)

    # Setup viewer
    print(f"\nStarting viewer server on port {port}...")
    try:
        server = viser.ViserServer(port=port, verbose=False)
    except OSError as e:
        if "already in use" in str(e):
            print(f"Error: Port {port} already in use. Use --port to specify a different port.")
        else:
            print(f"Error: Failed to start server: {e}")
        return

    # Add camera trajectories to viewer (if available)
    if data.camera_trajectories:
        add_camera_trajectories(server, data.camera_trajectories, data.world_to_nre)

    # Detect time range from tracks_data
    if data.gaussian_set.rigids is not None and data.tracks_data is not None:
        import numpy as np

        tracks_dict = data.tracks_data.get("tracks_data", {})
        timestamps_us_list = tracks_dict.get("tracks_timestamps_us", [])

        # Find global time range
        all_min_times = []
        all_max_times = []
        for timestamps_us in timestamps_us_list:
            if len(timestamps_us) > 0:
                timestamps_s = np.array(timestamps_us) / 1e6
                all_min_times.append(timestamps_s[0])
                all_max_times.append(timestamps_s[-1])

        if all_min_times and all_max_times:
            time_offset = min(all_min_times)
            time_max = max(all_max_times) - time_offset
            time_min = 0.0
            print(f"Detected track time range: {time_min:.2f}s to {time_max:.2f}s (offset: {time_offset:.2f}s)")
        else:
            time_offset = 0.0
            time_min = 0.0
            time_max = 480.0
    else:
        time_offset = 0.0
        time_min = 0.0
        time_max = 480.0

    # Create render wrapper that includes timestamp
    slider_ref = [None]
    is_playing_ref = [False]
    playback_thread_ref = [None]

    def render_wrapper(camera_state, render_tab_state):
        # Use normalized time + offset for absolute timestamp
        if data.gaussian_set.rigids is not None and slider_ref[0] is not None:
            abs_timestamp = slider_ref[0].value + time_offset
        else:
            abs_timestamp = None

        return render_fn(
            camera_state, render_tab_state, data.gaussian_set, data.sky_cubemap, torch_device, timestamp=abs_timestamp
        )

    # Add timeline slider and playback button
    viewer = None  # Will be set after viewer creation
    if data.gaussian_set.rigids is not None:
        import threading

        slider = server.gui.add_slider(
            "Timeline (s)",
            min=time_min,
            max=time_max,
            step=0.1,
            initial_value=time_min,
        )
        slider_ref[0] = slider

        print(f"Added timeline slider: {time_min}s to {time_max}s")

        # Add playback button
        play_button = server.gui.add_button(
            "Play",
            icon=viser.Icon.PLAYER_PLAY,
        )

        # Playback function running in separate thread
        def playback_loop():
            """Playback loop running at 10fps"""
            sleep_time_min = 2.0
            frame_duration = 0.1

            while is_playing_ref[0]:
                start_time = time.time()

                # Update timeline
                current_time = slider_ref[0].value
                next_time = current_time + frame_duration

                # Loop back to start if at end
                if next_time >= time_max:
                    next_time = time_min

                slider_ref[0].value = next_time

                # Trigger re-render
                if viewer is not None:
                    viewer.rerender(None)

                # Sleep for remaining frame time
                elapsed = time.time() - start_time
                sleep_time = sleep_time_min + frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Button click handler
        @play_button.on_click
        def _(event):
            if not is_playing_ref[0]:
                # Start playing
                is_playing_ref[0] = True
                playback_thread_ref[0] = threading.Thread(target=playback_loop, daemon=True)
                playback_thread_ref[0].start()

                # Update button to show pause option
                play_button.label = "Pause"
                play_button.icon = viser.Icon.PLAYER_PAUSE
            else:
                # Pause playback
                is_playing_ref[0] = False
                if playback_thread_ref[0] is not None:
                    playback_thread_ref[0].join(timeout=1.0)
                    playback_thread_ref[0] = None

                # Update button back to play
                play_button.label = "Play"
                play_button.icon = viser.Icon.PLAYER_PLAY

        # Setup slider update callback to trigger re-render
        @slider.on_update
        def _(_):
            # Trigger a re-render when slider value changes
            if viewer is not None:
                viewer.rerender(None)

    # Create nerfview viewer
    viewer = nerfview.Viewer(
        server=server,
        render_fn=render_wrapper,
        mode="rendering",
    )

    print(f"\nViewer running at http://localhost:{port}")
    if data.gaussian_set.rigids is not None:
        print("Use the Play button or timeline slider to animate rigid bodies")
    print("Press Ctrl+C to exit\n")

    # Keep the viewer running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


__all__ = ["view"]
