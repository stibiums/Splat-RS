# SplatRS

SplatRS is a small native Rust/wgpu viewer for pre-trained 3D Gaussian
Splatting models exported as GraphDECO-style `.ply` files.

The first version focuses on readability and course-project scope:

- load official 3DGS PLY files
- apply scale, opacity, and quaternion activations
- CPU-sort splats back-to-front each frame
- render instanced elliptical splats with wgpu
- orbit camera controls and simple keyboard toggles

## Usage

```sh
cargo run -p splatrs -- view path/to/point_cloud.ply
```

Useful options:

```sh
cargo run -p splatrs -- view model.ply --max-splats 100000 --width 1280 --height 720
```

Controls:

- Left mouse drag: orbit
- Mouse wheel: zoom
- `P`: toggle point/splat mode
- `O` / `I`: increase/decrease opacity scale
- `Esc`: quit

## Scope

This viewer does not train 3DGS models, run COLMAP, use CUDA, or implement GPU
sorting. Those are natural follow-up projects after the viewer pipeline is
working.
