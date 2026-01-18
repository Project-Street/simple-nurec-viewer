# Simple NuRec Viewer

A 3D Gaussian Splatting viewer for NuRec USDZ files with interactive web-based visualization.

## Features

- **Interactive 3D Visualization**: View NuRec USDZ files using 3D Gaussian Splatting rendering
- **Web-based Interface**: Browser-based viewer with intuitive camera controls
- **Rigid Body Animation**: Support for time-varying rigid body transforms (vehicles, pedestrians)
- **Sky Rendering**: High-quality sky cubemap rendering
- **Camera Trajectories**: Visualize camera paths from trajectory data
- **Timeline Control**: Interactive timeline slider for animating dynamic scenes

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (if using GPU rendering)
- Conda environment named "nurec"

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

3. Install dependencies:
```bash
# Install torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# Install the package in editable mode
pip install -e .
```

## Usage

### Basic Usage

```bash
simple-nurec-viewer path/to/file.usdz
```

### Advanced Options

```bash
# Specify custom port
simple-nurec-viewer --port 9000 path/to/file.usdz

# Use CPU rendering (if no GPU available)
simple-nurec-viewer --device cpu path/to/file.usdz

# Combine options
simple-nurec-viewer -p 8080 -d cuda path/to/file.usdz
```

### Viewer Controls

- **Mouse Left Click + Drag**: Rotate camera
- **Mouse Right Click + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Timeline Slider**: Animate rigid bodies (if available)

## Example Data

Example USDZ files can be found in:
```
data/sample_set/25.07_release/Batch0001/0b1c9f60-1e7b-40e4-a221-f9e1c281aad7/0b1c9f60-1e7b-40e4-a221-f9e1c281aad7.usdz
```

## Project Structure

```
simple_nurec_viewer/
├── gaussians/          # Gaussian hierarchy (BaseGaussian, RigidGaussian, HybridGaussian)
├── core/              # Core rendering logic and viewer
├── utils/             # Utility functions (rigid transforms, etc.)
└── cli.py             # Command-line interface
```

## Architecture

The viewer follows a three-layer Gaussian architecture:

- **BaseGaussian**: Abstract base class for static Gaussian primitives
- **RigidGaussian**: Extends BaseGaussian with time-varying rigid body transforms
- **HybridGaussian**: Aggregates multiple Gaussian objects for unified rendering

This design enables flexible composition of background, road, and dynamic object Gaussians.

## Development

### Code Style

- All code follows KISS, YAGNI, DRY, and LOD principles
- 4-space indentation (PEP 8)
- Type hints for all public functions
- English docstrings using NumPy style

### Testing

Manual testing with real USDZ files for visual regression testing.

## License

MIT License
