# SplatRS

SplatRS is a Rust/wgpu course project for viewing pre-trained 3D Gaussian
Splatting scenes exported in the GraphDECO PLY format. The project focuses on
the viewer side of the pipeline: PLY parsing, Gaussian data modeling, spherical
harmonics color evaluation, camera presets, GPU splat rendering, UI tuning, and
offline rendering utilities.

This repository does not train 3DGS models, run COLMAP, or depend on CUDA. It
loads existing `point_cloud.ply` models and renders them with a native desktop
viewer.

## Quick Start

```sh
cargo build --release -p splatrs
cargo test -p splatrs
target/release/splatrs view path/to/point_cloud.ply
```

If local demo assets are unpacked under `models/`, the default command opens the
course-project demo scene:

```sh
target/release/splatrs view
```

The viewer provides an egui control panel for scene selection, camera index,
screenshots, background, SH degree, opacity, splat scale, radius, exposure,
saturation, color clamp, and alpha parameters.

## Model Assets

Large model files are intentionally not tracked by Git. To reproduce the local
experiments, download the official GraphDECO pre-trained model package:

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

Then unpack the scenes you need under `models/`. The report experiments used
the `train`, `bonsai`, `room`, `truck`, and `flowers` scenes at
`point_cloud/iteration_7000/point_cloud.ply`.

## Useful Commands

```sh
# Open a scene interactively.
target/release/splatrs view models/zip_scenes/truck/point_cloud/iteration_7000/point_cloud.ply

# Inspect scene statistics and projected radius information.
cargo run -p splatrs -- inspect path/to/point_cloud.ply --camera-index 5

# Render one offscreen frame.
cargo run -p splatrs -- render path/to/point_cloud.ply -o frame.bmp --width 1280 --height 720

# Render several camera presets into one contact sheet.
cargo run -p splatrs -- contact-sheet path/to/point_cloud.ply -o cameras.bmp --camera-indices 0,5,10,20
```

The detailed crate-level usage notes are in [splatrs/README.md](splatrs/README.md).

## Repository Layout

- `splatrs/src/loader.rs`: GraphDECO PLY schema parsing and activation.
- `splatrs/src/scene.rs`: CPU/GPU Gaussian structures, SH color, and sorting.
- `splatrs/src/camera.rs`, `splatrs/src/cameras.rs`: orbit camera and
  `cameras.json` preset loading.
- `splatrs/src/renderer.rs`: wgpu pipeline and screen-space splat rendering.
- `splatrs/src/app.rs`: winit event loop, egui panel, scene reload, and
  screenshot workflow.
- `splatrs/src/headless.rs`: offscreen render, contact sheet, and quality sweep.
- `splatrs/examples/tiny_ascii.ply`: tiny tracked fixture for loader tests.

## Verification

The current test suite covers loader behavior, GraphDECO quaternion layout,
camera presets, SH color evaluation, depth sorting, the CPU raster reference
path, and WGSL shader parsing.

```sh
cargo test -p splatrs
```

At the time of final cleanup, the suite reports `37 passed; 0 failed`.

## Local Outputs

The following files and directories are generated locally and ignored by Git:

- `models/` and downloaded archives such as `models.zip`
- `target/`
- `screenshots/`
- local report drafts and rendered PDFs under `paper/` and `SplatRS_report.pdf`
