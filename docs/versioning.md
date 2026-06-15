# Versioning Notes

This repository uses a small, course-project-friendly Git workflow.

## Branches

- `main`: always keep buildable and demo-ready.
- `feature/<topic>`: use for focused work such as `feature/binary-ply`,
  `feature/sh-color`, or `feature/camera-controls`.
- `fix/<topic>`: use for small bug fixes after a milestone is working.

## Commits

- Prefer one working change per commit.
- Run `cargo fmt` before committing Rust changes.
- Run `cargo test` before merging a feature branch into `main`.
- Keep real 3DGS datasets outside Git under `models/` or `data/`.

## Tags

Suggested milestone tags:

- `v0.1-loader`: CLI, PLY loading, activations, tests.
- `v0.2-points`: first interactive point/quad viewer.
- `v0.3-splats`: sorted alpha-blended splat mode.
- `v1.0-demo`: final course-demo version with report screenshots.

