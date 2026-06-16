# SplatRS

SplatRS is a small native Rust/wgpu viewer for pre-trained 3D Gaussian
Splatting models exported as GraphDECO-style `.ply` files.

The first version focuses on readability and course-project scope:

- load official 3DGS PLY files
- apply scale, opacity, and quaternion activations
- keep the most visually important splats when `--max-splats` is used
- CPU-sort splats front-to-back each frame for transmittance blending
- evaluate SH degree 0-3 color on the CPU
- render instanced screen-space elliptical splats with wgpu
- orbit camera controls and simple keyboard toggles

## Usage

```sh
cargo run -p splatrs -- view path/to/point_cloud.ply
cargo run -p splatrs -- inspect path/to/point_cloud.ply
```

Useful options:

```sh
cargo run -p splatrs -- view model.ply --max-splats 100000 --width 1280 --height 720
cargo run -p splatrs -- view model.ply --sh-degree d3 --camera-index 5
cargo run -p splatrs -- view model.ply --splat-scale 0.4 --opacity-scale 1.5 --max-splat-radius 80
cargo run -p splatrs -- render model.ply -o frame.bmp --sh-degree d3 --width 1280 --height 720
cargo run -p splatrs -- contact-sheet model.ply -o cameras.bmp --sh-degree d3 --camera-indices 0,5,10,20
cargo run -p splatrs -- inspect model.ply --camera-index 5 --width 1280 --height 720
```

`--max-splats` takes a deterministic high-importance subset of the PLY instead
of the first N rows, which preserves most visible content for large official
models.

When a `cameras.json` file is found in an ancestor directory of the PLY, SplatRS
uses `--camera-index` from that file as the initial viewer pose.

Controls:

- Left mouse drag: orbit
- Mouse wheel: zoom
- `P`: toggle point/splat mode
- `O` / `I`: increase/decrease opacity scale
- `+` / `-`: increase/decrease splat scale
- `0`-`3`: switch SH degree
- `R`: reset camera
- `Esc`: quit

The Rust library exposes the main course-project building blocks through
`splatrs::{loader, scene, camera, renderer}`.

## Scope

This viewer does not train 3DGS models, run COLMAP, use CUDA, or implement GPU
sorting. Those are natural follow-up projects after the viewer pipeline is
working.
