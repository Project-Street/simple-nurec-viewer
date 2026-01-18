# Simple NuRec Viewer

A lightweight 3D Gaussian Splatting viewer for NuRec USDZ files with interactive web-based visualization and camera frame export capabilities.

## Features

- **Interactive 3D Visualization**: Web-based viewer for real-time exploration of NuRec scenes
- **Rigid Body Animation**: Timeline control for animating dynamic objects in the scene
- **Camera Frame Export**: Batch export frames from multiple cameras with support for:
  - FTheta camera model with distortion correction
  - Pinhole camera model
  - Resolution scaling
  - Custom timestamp ranges
- **Sky Box Rendering**: Optional sky cubemap for realistic backgrounds
- **Camera Trajectories**: Visualize camera paths in the viewer

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (recommended for GPU rendering)
- Conda environment

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd simple-nurec-viewer
```

2. Create and activate conda environment:
```bash
conda create -n nurec python=3.10
conda activate nurec
```

3. Install PyTorch with CUDA support:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install nvdiffrast:
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

5. Install the package in editable mode:
```bash
pip install -e .
```

## Usage

The `simple-nurec` CLI provides two main commands: `view` for interactive visualization and `export` for batch frame export.

### View Command

Interactively view NuRec USDZ files in a web browser.

```bash
# Basic usage
simple-nurec view path/to/file.usdz

# Specify custom port
simple-nurec view --port 9000 path/to/file.usdz

# Use CPU rendering
simple-nurec view --device cpu path/to/file.usdz
```

**Viewer Controls:**
- **Mouse Left Click + Drag**: Rotate camera
- **Mouse Right Click + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Timeline Slider**: Animate rigid bodies (if available)
- **Play/Pause Button**: Automatic playback of rigid body animation

### Export Command

Export camera frames from NuRec USDZ files.

```bash
# Export all cameras at full resolution
simple-nurec export path/to/file.usdz

# Export specific cameras
simple-nurec export path/to/file.usdz --cameras front_wide rear_wide

# Export with resolution scaling
simple-nurec export path/to/file.usdz --resolution-scale 0.5

# Export specific timestamp range
simple-nurec export path/to/file.usdz --timestamp-start 0 --timestamp-end 100

# Specify output directory
simple-nurec export path/to/file.usdz --output-dir ./my_outputs

# Use CPU rendering
simple-nurec export path/to/file.usdz --device cpu

# Enable debug output for coordinate transformations
simple-nurec export path/to/file.usdz --debug
```

**Export Options:**
- `--output-dir`: Output directory for exported frames (default: `outputs/`)
- `--resolution-scale`: Resolution scale factor in (0, 1.0] (default: 1.0)
- `--cameras`: List of camera names to export (default: all cameras)
- `--timestamp-start`: Start frame index (default: 0)
- `--timestamp-end`: End frame index (default: last frame)
- `--device`: Rendering device - `cuda` or `cpu` (default: cuda if available)
- `--overwrite`: Overwrite existing files (default: True)
- `--debug`: Enable debug output (default: False)

## Project Structure

```
simple_nurec_viewer/
├── cli/                    # Command-line interface
│   ├── view.py            # View subcommand
│   └── export.py          # Export subcommand
├── core/                   # Core rendering logic
│   ├── loader.py          # NuRec data loader
│   ├── rendering.py       # Rendering functions
│   └── viewer.py          # Viewer state management
├── export/                 # Camera frame export
│   ├── camera.py          # Camera pose computation
│   ├── loader.py          # Camera data loading
│   ├── renderer.py        # Frame rendering
│   └── writer.py          # Output writing
├── scenes/                 # Scene components
│   ├── gaussians/         # Gaussian hierarchy
│   │   ├── base.py        # BaseGaussian - static primitives
│   │   ├── rigid.py       # RigidGaussian - animated objects
│   │   └── hybrid.py      # HybridGaussian - composite rendering
│   └── sky.py             # Sky cubemap rendering
└── utils/                  # Utility functions
    └── rigid.py           # Rigid body transformations
```

## Architecture

### Gaussian Hierarchy

The viewer implements a three-layer Gaussian architecture:

- **BaseGaussian**: Abstract base class for static 3D Gaussian primitives
- **RigidGaussian**: Extends BaseGaussian with time-varying rigid body transforms
- **HybridGaussian**: Aggregates multiple Gaussian objects for unified rendering

This design enables flexible composition of:
- Background Gaussians (static scene elements)
- Road Gaussians (static road surface)
- Dynamic Object Gaussians (animated vehicles, pedestrians, etc.)

### Coordinate Systems

The viewer handles multiple coordinate systems:
- **World**: Global coordinate system
- **NRE**: NuRec Engine coordinates
- **Rig**: Camera rig coordinate system
- **Sensor**: Individual camera sensor coordinates

### Camera Models

Supported camera models:
- **FTheta**: Fisheye cameras with radial distortion
- **Pinhole**: Standard perspective cameras

## Development

### Code Style

This project follows these principles:
- **KISS**: Keep It Simple, Stupid
- **YAGNI**: You Ain't Gonna Need It
- **DRY**: Don't Repeat Yourself
- **LOD**: Law of Demeter

Additional guidelines:
- 4-space indentation (PEP 8)
- Type hints for all public functions
- English docstrings using NumPy style
- Ruff for linting and formatting

## Roadmap

- [ ] gRPC server and protocol for remote rendering
- [ ] Specular feature for Gaussian rendering
  - [ ] Requires migration from gsplat to original 3dgrut renderer
- [ ] Deformable Gaussian support

## License

MIT License
