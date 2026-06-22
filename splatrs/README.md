# SplatRS

SplatRS is a small native Rust/wgpu viewer for pre-trained 3D Gaussian
Splatting models exported as GraphDECO-style `.ply` files.

The first version focuses on readability and course-project scope:

- load official 3DGS PLY files
- apply scale, opacity, and quaternion activations
- keep the most visually important splats when `--max-splats` is used
- CPU-sort splats front-to-back for transmittance blending, with throttled
  resorting while interacting
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
cargo run -p splatrs -- view model.ply --sh-degree auto --camera-index 5
cargo run -p splatrs -- view model.ply --splat-scale 0.4 --opacity-scale 1.5 --max-splat-radius 80
cargo run -p splatrs -- view model.ply --background sky
cargo run -p splatrs -- view model.ply --sort-interval-ms 120
cargo run -p splatrs -- render model.ply -o frame.bmp --width 1280 --height 720
cargo run -p splatrs -- render model.ply -o cpu-frame.bmp --backend cpu-tile --cpu-sort tile-local --width 640 --height 360
cargo run -p splatrs -- contact-sheet model.ply -o cameras.bmp --camera-indices 0,5,10,20
cargo run -p splatrs -- quality-sweep model.ply -o tuned-frames --max-splats 100000 --camera-index 0
cargo run -p splatrs -- inspect model.ply --camera-index 5 --width 1280 --height 720
```

`--max-splats` takes a deterministic high-importance subset of the PLY instead
of the first N rows, which preserves most visible content for large official
models.

`--sort-interval-ms` trades interaction smoothness for exact transparency
ordering while orbiting or zooming. Higher values reduce CPU sorting and GPU
buffer uploads during camera motion; `0` restores immediate resorting.

When a `cameras.json` file is found in an ancestor directory of the PLY, SplatRS
uses `--camera-index` from that file as the initial viewer pose.

`--sh-degree auto` is the default for view, render, contact-sheet, and
quality-sweep. It evaluates the highest SH degree present in the PLY, capped at
degree 3. Use `--sh-degree d0` for DC-only debugging or to reproduce older
low-cost renders.

Quality experiments:

These options are intended for controlled comparisons. The default `axes`,
`area`, and `1/255` alpha cutoff path is still the safest visual baseline for
the sample train scene.

- `--footprint axes|covariance`: choose between the original axis-projection
  footprint and an explicit 3D covariance to 2D covariance projection.
- `--kernel-cutoff`: controls quad radius and fragment discard radius.
- `--lowpass-pixels`: controls the screen-space low-pass variance added to each
  projected footprint.
- `--radius-alpha area|linear|preserve`: controls how opacity changes when a
  very large splat is radius-clamped.
- `--alpha-cutoff` and `--max-alpha`: tune fragment-level alpha rejection and
  saturation.
- `--color-max` and `--saturation`: clamp and desaturate evaluated SH colors
  before exposure/tone mapping; useful for diagnosing colorful SH outliers.
- `--backend cpu-tile --cpu-sort tile-local`: use the CPU tile renderer with
  per-tile depth sorting and flat tile bins instead of a full-scene sort.

Controls:

- Left mouse drag: orbit
- Mouse wheel: zoom
- `P`: toggle point/splat mode
- `O` / `I`: increase/decrease opacity scale
- `+` / `-`: increase/decrease splat scale
- `E` / `D`: increase/decrease exposure
- `T`: cycle display tone mapping
- `0`-`3`: switch SH degree
- `R`: reset camera
- `Esc`: quit

The Rust library exposes the main course-project building blocks through
`splatrs::{loader, scene, camera, renderer}`.

## Scope

This viewer does not train 3DGS models, run COLMAP, use CUDA, or implement GPU
sorting. Those are natural follow-up projects after the viewer pipeline is
working.
