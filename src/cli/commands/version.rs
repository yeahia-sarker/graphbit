//! Version command handler
//!
//! This module handles the `version` command for displaying version information.

use graphbit_core::GraphBitResult;

/// Handle the version command
pub async fn handle_version() -> GraphBitResult<()> {
    println!("GraphBit v{}", graphbit_core::VERSION);
    println!("Declarative framework for agentic workflow automation");
    println!();
    println!("Built with Rust for maximum performance and safety");
    println!("For more information, visit: https://github.com/InfinitiBit/graphbit");

    Ok(())
}
