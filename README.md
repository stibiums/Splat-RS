# SplatRS

SplatRS 是一个 Rust/wgpu 课程项目，用于加载和查看 GraphDECO 风格的 3D Gaussian Splatting 预训练模型。项目关注的是渲染器侧工程链路：PLY 文件解析、Gaussian 数据建模、球谐颜色恢复、相机预设读取、GPU splat 绘制、交互式参数调节和离屏渲染工具。

本项目不训练 3DGS 模型，不运行 COLMAP，也不依赖 CUDA。它加载已有的 `point_cloud.ply` 模型，并通过原生桌面窗口进行交互式渲染。

## 快速开始

```sh
cargo build --release -p splatrs
cargo test -p splatrs
target/release/splatrs view path/to/point_cloud.ply
```

如果本地已经在 `models/` 下解压了课程项目使用的模型资产，可以直接运行默认场景：

```sh
target/release/splatrs view
```

窗口左侧提供 egui 控制面板，可以选择场景、切换 camera index、截图，并调整背景、SH degree、opacity、splat scale、最大半径、exposure、saturation、color clamp 和 alpha 参数。

## 模型资产

大模型文件不放入 Git。复现实验时可以下载 GraphDECO 官方预训练模型包：

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

下载后按需要把场景解压到 `models/` 目录。报告中使用的场景包括 `train`、`bonsai`、`room`、`truck` 和 `flowers`，模型文件路径形如：

```text
models/.../point_cloud/iteration_7000/point_cloud.ply
```

## 常用命令

```sh
# 打开一个场景并进入交互窗口
target/release/splatrs view models/zip_scenes/truck/point_cloud/iteration_7000/point_cloud.ply

# 查看场景统计信息和投影半径分布
cargo run -p splatrs -- inspect path/to/point_cloud.ply --camera-index 5

# 离屏渲染一帧
cargo run -p splatrs -- render path/to/point_cloud.ply -o frame.bmp --width 1280 --height 720

# 将多个官方相机视角渲染成 contact sheet
cargo run -p splatrs -- contact-sheet path/to/point_cloud.ply -o cameras.bmp --camera-indices 0,5,10,20

# 输出若干显示参数组合，便于比较画面效果
cargo run -p splatrs -- quality-sweep path/to/point_cloud.ply -o tuned-frames --camera-index 0
```

默认显示参数偏向演示效果：sky 背景、camera index 5、`opacity=8.0`、`splat_scale=0.7`、`max_radius=240`、`exposure=0.9`、`saturation=1.05`，并启用排序节流和交互 LOD。拖动或缩放时会减少交互期间绘制的 splat 数量，停止交互后恢复完整质量。

## 主要功能

- 解析 GraphDECO 风格的 ASCII 和 binary little endian PLY schema。
- 将 opacity logit、log-scale 和四元数激活为可渲染数据。
- 支持 DC 到 3 阶球谐颜色，并默认自动选择 PLY 中可用的最高阶数。
- 支持 `cameras.json` 官方相机预设。
- 使用 wgpu instanced drawing 渲染屏幕空间 Gaussian splat。
- 使用 CPU 深度排序、排序节流和交互 LOD 改善交互体验。
- 提供 egui 控制面板，避免依赖键盘快捷键或重复启动命令行。
- 支持截图，并输出同名参数记录文件。
- 提供 `render`、`contact-sheet`、`quality-sweep` 和 `inspect` 等离屏工具。

## 代码结构

- `splatrs/src/loader.rs`：PLY schema 解析与字段激活。
- `splatrs/src/scene.rs`：CPU/GPU Gaussian 数据结构、球谐颜色和排序。
- `splatrs/src/camera.rs`、`splatrs/src/cameras.rs`：orbit 相机和 `cameras.json` 预设读取。
- `splatrs/src/renderer.rs`：wgpu pipeline 和屏幕空间 splat 渲染。
- `splatrs/src/app.rs`：winit 事件循环、egui 面板、场景切换和截图流程。
- `splatrs/src/headless.rs`：离屏渲染、contact sheet 和质量扫描。
- `splatrs/examples/tiny_ascii.ply`：用于 loader 测试的最小 PLY fixture。

## 验证

测试覆盖 loader、GraphDECO 四元数布局、相机预设、SH 颜色、深度排序、CPU raster 参考路径和 WGSL shader 解析。

```sh
cargo test -p splatrs
```

当前整理时测试结果为：

```text
37 passed; 0 failed
```

## Git 中不包含的内容

以下内容属于本地资产或生成结果，不提交到 Git：

- `models/` 和 `models.zip` 等下载模型包。
- `target/` 编译产物和渲染中间图。
- `screenshots/` 截图输出。
- `paper/`、`SplatRS_report.pdf` 和其他报告编译产物。
- `docs/` 中的本地草稿或过程记录。

最终提交仓库只保留代码、最小测试 fixture 和这份主 README，方便评分者快速了解、构建和运行项目。
