use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::loader::LoadOptions;

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
    /// Parse a GraphDECO-style 3DGS PLY model and print scene statistics.
    Inspect(InspectArgs),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ShDegree {
    D0,
    D1,
    D2,
    D3,
}

impl ShDegree {
    pub fn as_u32(self) -> u32 {
        match self {
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
    #[arg(long, value_enum, default_value_t = ShDegree::D0)]
    pub sh_degree: ShDegree,

    /// Initial opacity multiplier.
    #[arg(long, default_value_t = 1.4)]
    pub opacity_scale: f32,

    /// Initial splat radius multiplier.
    #[arg(long, default_value_t = 0.43)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 96.0)]
    pub max_splat_radius: f32,

    /// Zero-based camera index from cameras.json to use as the initial view.
    #[arg(long, default_value_t = 0)]
    pub camera_index: usize,

    /// Initial window width.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Initial window height.
    #[arg(long, default_value_t = 720)]
    pub height: u32,
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
    #[arg(long, default_value_t = 0.43)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels for projected statistics.
    #[arg(long, default_value_t = 96.0)]
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
    #[arg(long, value_enum, default_value_t = ShDegree::D0)]
    pub sh_degree: ShDegree,

    /// Opacity multiplier.
    #[arg(long, default_value_t = 1.4)]
    pub opacity_scale: f32,

    /// Splat radius multiplier.
    #[arg(long, default_value_t = 0.43)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 96.0)]
    pub max_splat_radius: f32,

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
    #[arg(long, value_enum, default_value_t = ShDegree::D0)]
    pub sh_degree: ShDegree,

    /// Opacity multiplier.
    #[arg(long, default_value_t = 1.4)]
    pub opacity_scale: f32,

    /// Splat radius multiplier.
    #[arg(long, default_value_t = 0.43)]
    pub splat_scale: f32,

    /// Maximum screen-space splat quad radius in pixels.
    #[arg(long, default_value_t = 96.0)]
    pub max_splat_radius: f32,

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
}
