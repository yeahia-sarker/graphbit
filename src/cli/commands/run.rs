//! Workflow execution command handler
//!
//! This module handles the `run` command for executing workflows.

use crate::cli::{
    config::{create_llm_config, load_config},
    workflow::{convert_workflow_data, extract_agent_ids, WorkflowFile},
};
use graphbit_core::{AgentBuilder, GraphBitResult, WorkflowExecutor};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

/// Handle the run command
pub async fn handle_run(workflow: PathBuf, config: Option<PathBuf>) -> GraphBitResult<()> {
    // Load configuration
    let config = load_config(config.as_ref()).await?;

    // Load and parse workflow
    let workflow_content = fs::read_to_string(workflow)?;
    let workflow_data: WorkflowFile = serde_json::from_str(&workflow_content)?;
    let workflow = convert_workflow_data(workflow_data)?;

    // Create LLM configuration
    let llm_config = create_llm_config(&config.llm)?;

    // Create workflow executor
    let executor = WorkflowExecutor::new();

    // Register agents for each unique agent_id in the workflow
    let agent_ids = extract_agent_ids(&workflow);
    for agent_id_str in agent_ids {
        let agent_config = config.agents.get("default").ok_or_else(|| {
            graphbit_core::errors::GraphBitError::config("No default agent configuration found")
        })?;

        let agent = AgentBuilder::new(&agent_id_str, llm_config.clone())
            .system_prompt(&agent_config.system_prompt)
            .max_tokens(agent_config.max_tokens)
            .temperature(agent_config.temperature)
            .build()
            .await?;

        executor.register_agent(Arc::new(agent)).await;
    }

    println!("🚀 Starting workflow execution...");

    // Execute the workflow
    let context = executor.execute(workflow).await?;

    println!("📊 Execution Results:");
    println!("  Status: {:?}", context.state);
    println!(
        "  Duration: {:?}",
        context
            .completed_at
            .unwrap_or(context.started_at)
            .signed_duration_since(context.started_at)
    );

    if !context.variables.is_empty() {
        println!("  Variables:");
        for (key, value) in &context.variables {
            println!("    {}: {}", key, value);
        }
    }

    println!("✅ Workflow completed successfully!");

    Ok(())
}
