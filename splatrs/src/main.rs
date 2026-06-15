mod app;
mod camera;
mod cli;
mod loader;
mod renderer;
mod scene;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "splatrs=info,wgpu=warn".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::View(args) => app::run(args),
    }
}
