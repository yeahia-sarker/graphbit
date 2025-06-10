//! Integration tests for GraphBit
//!
//! This module contains integration tests that test the complete functionality
//! of the GraphBit framework, including CLI commands, workflow execution,
//! and end-to-end scenarios.

pub mod agent_tests;
pub mod cli_tests;
pub mod error_handling_tests;
pub mod file_io_tests;
pub mod graph_tests;
pub mod validation_tests;
pub mod workflow_tests;

use std::path::PathBuf;
use tempfile::TempDir;

/// Get API key from environment or provide test default
pub fn get_test_api_key() -> String {
    std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("TEST_OPENAI_API_KEY"))
        .unwrap_or_else(|_| "test-api-key-placeholder".to_string())
}

/// Get LLM model from environment or provide test default
pub fn get_test_model() -> String {
    std::env::var("TEST_LLM_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string())
}

/// Test utilities and common setup for integration tests
pub struct TestEnvironment {
    #[allow(dead_code)]
    pub temp_dir: TempDir,
    pub config_path: PathBuf,
    pub workflow_path: PathBuf,
}

impl TestEnvironment {
    pub fn new() -> anyhow::Result<Self> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");
        let workflow_path = temp_dir.path().join("workflow.json");

        Ok(Self {
            temp_dir,
            config_path,
            workflow_path,
        })
    }

    pub fn create_test_config(&self) -> anyhow::Result<()> {
        let config_content = serde_json::json!({
            "llm": {
                "provider": "openai",
                "model": get_test_model(),
                "api_key": get_test_api_key(),
                "base_url": null,
                "organization": null
            },
            "agents": {
                "test-agent": {
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "system_prompt": "You are a helpful test agent.",
                    "capabilities": ["text-generation", "analysis"]
                }
            }
        });

        std::fs::write(
            &self.config_path,
            serde_json::to_string_pretty(&config_content)?,
        )?;
        Ok(())
    }

    pub fn create_simple_workflow(&self) -> anyhow::Result<()> {
        let workflow_content = serde_json::json!({
            "name": "Test Workflow",
            "description": "A simple test workflow",
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "name": "Start Agent",
                    "description": "Starting agent for the workflow",
                    "config": {
                        "agent_id": "test-agent",
                        "prompt": "Generate a greeting message"
                    }
                }
            ],
            "edges": [],
            "metadata": {}
        });

        std::fs::write(
            &self.workflow_path,
            serde_json::to_string_pretty(&workflow_content)?,
        )?;
        Ok(())
    }

    pub fn create_complex_workflow(&self) -> anyhow::Result<()> {
        let workflow_content = serde_json::json!({
            "name": "Complex Test Workflow",
            "description": "A complex workflow with multiple nodes and edges",
            "nodes": [
                {
                    "id": "input",
                    "type": "agent",
                    "name": "Input Agent",
                    "description": "Processes input data",
                    "config": {
                        "agent_id": "test-agent",
                        "prompt": "Process this input: {input_data}"
                    }
                },
                {
                    "id": "transformer",
                    "type": "transform",
                    "name": "Data Transformer",
                    "description": "Transforms the processed data",
                    "config": {
                        "transformation": "uppercase"
                    }
                },
                {
                    "id": "condition",
                    "type": "condition",
                    "name": "Quality Check",
                    "description": "Checks data quality",
                    "config": {
                        "expression": "length > 10"
                    }
                },
                {
                    "id": "output",
                    "type": "agent",
                    "name": "Output Agent",
                    "description": "Formats final output",
                    "config": {
                        "agent_id": "test-agent",
                        "prompt": "Format this data for output: {processed_data}"
                    }
                },
                {
                    "id": "delay",
                    "type": "delay",
                    "name": "Processing Delay",
                    "description": "Brief delay for processing",
                    "config": {
                        "duration_seconds": 1
                    }
                }
            ],
            "edges": [
                {
                    "from": "input",
                    "to": "transformer",
                    "type": "data_flow"
                },
                {
                    "from": "transformer",
                    "to": "condition",
                    "type": "data_flow"
                },
                {
                    "from": "condition",
                    "to": "output",
                    "type": "conditional",
                    "condition": "passed"
                },
                {
                    "from": "condition",
                    "to": "delay",
                    "type": "conditional",
                    "condition": "!passed"
                }
            ],
            "metadata": {
                "version": "1.0",
                "author": "test-suite"
            }
        });

        std::fs::write(
            &self.workflow_path,
            serde_json::to_string_pretty(&workflow_content)?,
        )?;
        Ok(())
    }

    pub fn create_invalid_workflow(&self) -> anyhow::Result<()> {
        let workflow_content = serde_json::json!({
            "name": "Invalid Workflow",
            "description": "A workflow with validation errors",
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "name": "Agent 1",
                    "description": "First agent",
                    "config": {
                        "agent_id": "nonexistent-agent",
                        "prompt": "Do something"
                    }
                },
                {
                    "id": "node2",
                    "type": "invalid-type",
                    "name": "Invalid Node",
                    "description": "A node with invalid type",
                    "config": {}
                }
            ],
            "edges": [
                {
                    "from": "node1",
                    "to": "nonexistent-node",
                    "type": "data_flow"
                }
            ],
            "metadata": {}
        });

        std::fs::write(
            &self.workflow_path,
            serde_json::to_string_pretty(&workflow_content)?,
        )?;
        Ok(())
    }
}
