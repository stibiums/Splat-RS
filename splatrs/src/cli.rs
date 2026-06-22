use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::loader::LoadOptions;
use crate::renderer::{CpuSortMode, Footprint, RadiusAlpha, ToneMap};

#[derive(Debug, Parser)]
#[command(name = "splatrs")]
#[command(about = "Native Rust/wgpu 3D Gaussian Splatting viewer")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Open a GraphDECO-style 3DGS PLY model.
    View(ViewArgs),
    /// Render one offscreen frame to a BMP image.
    Render(RenderArgs),
    /// Render several cameras into one BMP contact sheet for debugging.
    ContactSheet(ContactSheetArgs),
    /// Render curated quality-tuning profiles for one camera.
    QualitySweep(QualitySweepArgs),
    /// Parse a GraphDECO-style 3DGS PLY model and print scene statistics.
    Inspect(InspectArgs),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ShDegree {
    Auto,
    D0,
    D1,
    D2,
    D3,
}

impl ShDegree {
    pub fn resolve(self, detected_degree: u32) -> u32 {
        match self {
            Self::Auto => detected_degree.min(3),
            Self::D0 => 0,
            Self::D1 => 1,
            Self::D2 => 2,
            Self::D3 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RenderBackend {
    GpuQuad,
    CpuTile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum CpuSortArg {
    Global,
    TileLocal,
}

impl CpuSortArg {
    pub fn as_renderer(self) -> CpuSortMode {
        match self {
            Self::Global => CpuSortMode::Global,
            Self::TileLocal => CpuSortMode::TileLocal,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Background {
    Dark,
    Gray,
    White,
    Sky,
}

impl Background {
    pub fn as_rgb(self) -> [f32; 3] {
        match self {
            Self::Dark => [0.015, 0.017, 0.02],
            Self::Gray => [0.18, 0.18, 0.18],
            Self::White => [1.0, 1.0, 1.0],
            Self::Sky => [0.72, 0.80, 0.92],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ToneMapArg {
    None,
    Reinhard,
    Aces,
}

impl ToneMapArg {
    pub fn as_renderer(self) -> ToneMap {
        match self {
            Self::None => ToneMap::None,
            Self::Reinhard => ToneMap::Reinhard,
            Self::Aces => ToneMap::Aces,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum FootprintArg {
    Axes,
    Covariance,
}

impl FootprintArg {
    pub fn as_renderer(self) -> Footprint {
        match self {
            Self::Axes => Footprint::Axes,
            Self::Covariance => Footprint::Covariance,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RadiusAlphaArg {
    Area,
    Linear,
    Preserve,
}

impl RadiusAlphaArg {
    pub fn as_renderer(self) -> RadiusAlpha {
        match self {
            Self::Area => RadiusAlpha::Area,
            Self::Linear => RadiusAlpha::Linear,
            Self::Preserve => RadiusAlpha::Preserve,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Args)]
pub struct SplatFilterArgs {
    /// Drop splats with activated opacity below this threshold.
    #[arg(long)]
    pub min_opacity: Option<f32>,

    /// Drop splats whose largest activated world-space scale exceeds this value.
    #[arg(long)]
    pub max_world_scale: Option<f32>,
}

impl SplatFilterArgs {
    pub fn load_options(self, max_splats: Option<usize>) -> LoadOptions {
        LoadOptions {
            max_splats,
            min_opacity: self.min_opacity,
            max_world_scale: self.max_world_scale,
        }
    }
}

#[derive(Debug, Args)]
pub struct ViewArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Keep a deterministic high-importance subset of at most N splats.
    #[arg(long)]
    pub max_splats: Option<usize>,

    #[command(flatten)]
    pub filters: SplatFilterArgs,

    /// Spherical harmonics degree to evaluate for view-dependent color.
    #[arg(long, value_enum, default_value_t = ShDegree::Auto)]
    pub sh_degree: ShDegree,

    /// Initial opacity multiplier.
    #[arg(long, default_value_t = 1.5)]
    pub opacity_scale: f32,

    /// Initial splat radius multiplier.
    #[arg(long, default_value_t = 0.4)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 80.0)]
    pub max_splat_radius: f32,

    /// Screen-space footprint projection mode.
    #[arg(long, value_enum, default_value_t = FootprintArg::Axes)]
    pub footprint: FootprintArg,

    /// Gaussian kernel cutoff used for quad radius and fragment discard.
    #[arg(long, default_value_t = 8.0)]
    pub kernel_cutoff: f32,

    /// Low-pass variance added to the projected 2D footprint in pixels squared.
    #[arg(long, default_value_t = 0.3)]
    pub lowpass_pixels: f32,

    /// Fragment alpha threshold below which splat samples are discarded.
    #[arg(long, default_value_t = 1.0 / 255.0)]
    pub alpha_cutoff: f32,

    /// Maximum per-fragment alpha after Gaussian falloff.
    #[arg(long, default_value_t = 0.99)]
    pub max_alpha: f32,

    /// Clamp evaluated SH color channels before exposure/tone mapping.
    #[arg(long, default_value_t = 1024.0)]
    pub color_max: f32,

    /// Saturation multiplier applied before exposure/tone mapping.
    #[arg(long, default_value_t = 1.0)]
    pub saturation: f32,

    /// Opacity policy used after a very large footprint is radius-clamped.
    #[arg(long, value_enum, default_value_t = RadiusAlphaArg::Area)]
    pub radius_alpha: RadiusAlphaArg,

    /// Background color preset used behind transparent splats.
    #[arg(long, value_enum, default_value_t = Background::Dark)]
    pub background: Background,

    /// Linear exposure multiplier applied before display.
    #[arg(long, default_value_t = 1.0)]
    pub exposure: f32,

    /// Display tone mapping curve used to compress bright SH colors.
    #[arg(long, value_enum, default_value_t = ToneMapArg::None)]
    pub tone_map: ToneMapArg,

    /// Apply experimental low-pass alpha compensation for tiny splats.
    #[arg(long)]
    pub lowpass_alpha_compensation: bool,

    /// Zero-based camera index from cameras.json to use as the initial view.
    #[arg(long, default_value_t = 0)]
    pub camera_index: usize,

    /// Initial window width.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Initial window height.
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Minimum milliseconds between CPU depth resort/upload while interacting.
    #[arg(long, default_value_t = 80)]
    pub sort_interval_ms: u64,
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Keep a deterministic high-importance subset of at most N splats.
    #[arg(long)]
    pub max_splats: Option<usize>,

    #[command(flatten)]
    pub filters: SplatFilterArgs,

    /// Print projected screen-space radius statistics for this cameras.json index.
    #[arg(long)]
    pub camera_index: Option<usize>,

    /// Viewport width used for projected radius statistics.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Viewport height used for projected radius statistics.
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Splat radius multiplier used for projected radius statistics.
    #[arg(long, default_value_t = 0.4)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels for projected statistics.
    #[arg(long, default_value_t = 80.0)]
    pub max_splat_radius: f32,
}

#[derive(Debug, Args)]
pub struct RenderArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Path to the output BMP image.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Keep a deterministic high-importance subset of at most N splats.
    #[arg(long)]
    pub max_splats: Option<usize>,

    #[command(flatten)]
    pub filters: SplatFilterArgs,

    /// Spherical harmonics degree to evaluate for view-dependent color.
    #[arg(long, value_enum, default_value_t = ShDegree::Auto)]
    pub sh_degree: ShDegree,

    /// Opacity multiplier.
    #[arg(long, default_value_t = 1.5)]
    pub opacity_scale: f32,

    /// Splat radius multiplier.
    #[arg(long, default_value_t = 0.4)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 80.0)]
    pub max_splat_radius: f32,

    /// Screen-space footprint projection mode.
    #[arg(long, value_enum, default_value_t = FootprintArg::Axes)]
    pub footprint: FootprintArg,

    /// Gaussian kernel cutoff used for quad radius and fragment discard.
    #[arg(long, default_value_t = 8.0)]
    pub kernel_cutoff: f32,

    /// Low-pass variance added to the projected 2D footprint in pixels squared.
    #[arg(long, default_value_t = 0.3)]
    pub lowpass_pixels: f32,

    /// Fragment alpha threshold below which splat samples are discarded.
    #[arg(long, default_value_t = 1.0 / 255.0)]
    pub alpha_cutoff: f32,

    /// Maximum per-fragment alpha after Gaussian falloff.
    #[arg(long, default_value_t = 0.99)]
    pub max_alpha: f32,

    /// Clamp evaluated SH color channels before exposure/tone mapping.
    #[arg(long, default_value_t = 1024.0)]
    pub color_max: f32,

    /// Saturation multiplier applied before exposure/tone mapping.
    #[arg(long, default_value_t = 1.0)]
    pub saturation: f32,

    /// Opacity policy used after a very large footprint is radius-clamped.
    #[arg(long, value_enum, default_value_t = RadiusAlphaArg::Area)]
    pub radius_alpha: RadiusAlphaArg,

    /// Background color preset used behind transparent splats.
    #[arg(long, value_enum, default_value_t = Background::Dark)]
    pub background: Background,

    /// Linear exposure multiplier applied before display.
    #[arg(long, default_value_t = 1.0)]
    pub exposure: f32,

    /// Display tone mapping curve used to compress bright SH colors.
    #[arg(long, value_enum, default_value_t = ToneMapArg::None)]
    pub tone_map: ToneMapArg,

    /// Apply experimental low-pass alpha compensation for tiny splats.
    #[arg(long)]
    pub lowpass_alpha_compensation: bool,

    /// Zero-based camera index from cameras.json to render.
    #[arg(long, default_value_t = 0)]
    pub camera_index: usize,

    /// Output image width.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Output image height.
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Headless rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackend::GpuQuad)]
    pub backend: RenderBackend,

    /// CPU backend sorting path. Ignored by the GPU backend.
    #[arg(long, value_enum, default_value_t = CpuSortArg::TileLocal)]
    pub cpu_sort: CpuSortArg,
}

#[derive(Debug, Args)]
pub struct ContactSheetArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Path to the output BMP image.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Keep a deterministic high-importance subset of at most N splats.
    #[arg(long)]
    pub max_splats: Option<usize>,

    #[command(flatten)]
    pub filters: SplatFilterArgs,

    /// Spherical harmonics degree to evaluate for view-dependent color.
    #[arg(long, value_enum, default_value_t = ShDegree::Auto)]
    pub sh_degree: ShDegree,

    /// Opacity multiplier.
    #[arg(long, default_value_t = 1.5)]
    pub opacity_scale: f32,

    /// Splat radius multiplier.
    #[arg(long, default_value_t = 0.4)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 80.0)]
    pub max_splat_radius: f32,

    /// Screen-space footprint projection mode.
    #[arg(long, value_enum, default_value_t = FootprintArg::Axes)]
    pub footprint: FootprintArg,

    /// Gaussian kernel cutoff used for quad radius and fragment discard.
    #[arg(long, default_value_t = 8.0)]
    pub kernel_cutoff: f32,

    /// Low-pass variance added to the projected 2D footprint in pixels squared.
    #[arg(long, default_value_t = 0.3)]
    pub lowpass_pixels: f32,

    /// Fragment alpha threshold below which splat samples are discarded.
    #[arg(long, default_value_t = 1.0 / 255.0)]
    pub alpha_cutoff: f32,

    /// Maximum per-fragment alpha after Gaussian falloff.
    #[arg(long, default_value_t = 0.99)]
    pub max_alpha: f32,

    /// Clamp evaluated SH color channels before exposure/tone mapping.
    #[arg(long, default_value_t = 1024.0)]
    pub color_max: f32,

    /// Saturation multiplier applied before exposure/tone mapping.
    #[arg(long, default_value_t = 1.0)]
    pub saturation: f32,

    /// Opacity policy used after a very large footprint is radius-clamped.
    #[arg(long, value_enum, default_value_t = RadiusAlphaArg::Area)]
    pub radius_alpha: RadiusAlphaArg,

    /// Background color preset used behind transparent splats.
    #[arg(long, value_enum, default_value_t = Background::Dark)]
    pub background: Background,

    /// Linear exposure multiplier applied before display.
    #[arg(long, default_value_t = 1.0)]
    pub exposure: f32,

    /// Display tone mapping curve used to compress bright SH colors.
    #[arg(long, value_enum, default_value_t = ToneMapArg::None)]
    pub tone_map: ToneMapArg,

    /// Apply experimental low-pass alpha compensation for tiny splats.
    #[arg(long)]
    pub lowpass_alpha_compensation: bool,

    /// Comma-separated zero-based camera indices from cameras.json.
    #[arg(long, value_delimiter = ',', default_value = "0,5,10,20")]
    pub camera_indices: Vec<usize>,

    /// Number of columns in the contact sheet.
    #[arg(long, default_value_t = 2)]
    pub columns: usize,

    /// Width of each rendered tile.
    #[arg(long, default_value_t = 640)]
    pub width: u32,

    /// Height of each rendered tile.
    #[arg(long, default_value_t = 360)]
    pub height: u32,

    /// Headless rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackend::GpuQuad)]
    pub backend: RenderBackend,

    /// CPU backend sorting path. Ignored by the GPU backend.
    #[arg(long, value_enum, default_value_t = CpuSortArg::TileLocal)]
    pub cpu_sort: CpuSortArg,
}

#[derive(Debug, Args)]
pub struct QualitySweepArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Directory where profile BMP images will be written.
    #[arg(short, long)]
    pub output_dir: PathBuf,

    /// Keep a deterministic high-importance subset of at most N splats.
    #[arg(long)]
    pub max_splats: Option<usize>,

    #[command(flatten)]
    pub filters: SplatFilterArgs,

    /// Spherical harmonics degree used by all non-DC quality profiles.
    #[arg(long, value_enum, default_value_t = ShDegree::Auto)]
    pub sh_degree: ShDegree,

    /// Background color preset used behind transparent splats.
    #[arg(long, value_enum, default_value_t = Background::Sky)]
    pub background: Background,

    /// Zero-based camera index from cameras.json to render.
    #[arg(long, default_value_t = 0)]
    pub camera_index: usize,

    /// Output image width.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Output image height.
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Headless rendering backend.
    #[arg(long, value_enum, default_value_t = RenderBackend::GpuQuad)]
    pub backend: RenderBackend,

    /// CPU backend sorting path. Ignored by the GPU backend.
    #[arg(long, value_enum, default_value_t = CpuSortArg::TileLocal)]
    pub cpu_sort: CpuSortArg,
}
