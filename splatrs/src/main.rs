use anyhow::Result;
use clap::Parser;
use splatrs::{
    app,
    cli::{Cli, Command},
    headless, inspect,
};

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
        Command::Render(args) => headless::run(args),
        Command::ContactSheet(args) => headless::run_contact_sheet(args),
        Command::Inspect(args) => inspect::run(args),
    }
}
