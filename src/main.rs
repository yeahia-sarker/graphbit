//! GraphBit CLI Application
//!
//! Command-line interface for the GraphBit agentic workflow automation framework.

mod cli;

use clap::Parser;
use cli::{
    args::{Cli, Commands},
    commands::{handle_init, handle_run, handle_validate, handle_version},
};
use graphbit_core::{init, GraphBitResult};

#[tokio::main]
async fn main() -> GraphBitResult<()> {
    // Initialize the GraphBit core
    init()?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { name, path } => {
            handle_init(name, path).await?;
        }
        Commands::Validate { workflow } => {
            handle_validate(workflow).await?;
        }
        Commands::Run { workflow, config } => {
            handle_run(workflow, config).await?;
        }
        Commands::Version => {
            handle_version().await?;
        }
    }

    Ok(())
}
