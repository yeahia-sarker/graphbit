//! CLI argument parsing and command definitions
//!
//! This module defines the command-line interface structure using clap.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "graphbit")]
#[command(about = "Declarative framework for agentic workflow automation")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new GraphBit project
    Init {
        /// Project name
        name: String,
        /// Project directory (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
    /// Validate a workflow definition
    Validate {
        /// Path to workflow file
        workflow: PathBuf,
    },
    /// Execute a workflow
    Run {
        /// Path to workflow file
        workflow: PathBuf,
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Show version information
    Version,
}
