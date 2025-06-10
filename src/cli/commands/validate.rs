//! Workflow validation command handler
//!
//! This module handles the `validate` command for checking workflow files.

use crate::cli::workflow::validate_workflow;
use graphbit_core::GraphBitResult;
use std::path::PathBuf;

/// Handle the validate command
pub async fn handle_validate(workflow: PathBuf) -> GraphBitResult<()> {
    println!("Validating workflow: {:?}", workflow);

    // Load and validate workflow
    validate_workflow(&workflow).await?;

    println!("✅ Workflow is valid!");

    Ok(())
}
