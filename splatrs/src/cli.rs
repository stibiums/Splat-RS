use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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

#[derive(Debug, Args)]
pub struct ViewArgs {
    /// Path to a GraphDECO-style point_cloud.ply file.
    pub model: PathBuf,

    /// Keep only the first N splats after loading.
    #[arg(long)]
    pub max_splats: Option<usize>,

    /// Spherical harmonics degree to evaluate for view-dependent color.
    #[arg(long, value_enum, default_value_t = ShDegree::D0)]
    pub sh_degree: ShDegree,

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

    /// Keep only the first N splats after loading.
    #[arg(long)]
    pub max_splats: Option<usize>,
}
