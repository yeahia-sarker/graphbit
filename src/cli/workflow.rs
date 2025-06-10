//! Workflow operations for GraphBit CLI
//!
//! This module handles workflow file operations including loading, parsing,
//! validation, and conversion to internal representations.

use graphbit_core::{
    errors::GraphBitError,
    graph::{NodeType, WorkflowEdge, WorkflowNode},
    types::{AgentId, NodeId},
    GraphBitResult, Workflow,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Workflow file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowFile {
    pub name: String,
    pub description: String,
    pub nodes: Vec<WorkflowNodeData>,
    pub edges: Vec<WorkflowEdgeData>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNodeData {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    pub description: String,
    pub config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdgeData {
    pub from: String,
    pub to: String,
    #[serde(rename = "type", default = "default_edge_type")]
    pub edge_type: String,
    #[serde(default)]
    pub condition: Option<String>,
    #[serde(default)]
    pub transform: Option<String>,
}

fn default_edge_type() -> String {
    "data_flow".to_string()
}

/// Validate a workflow file
pub async fn validate_workflow(workflow_path: &PathBuf) -> GraphBitResult<()> {
    // Load the workflow file
    let workflow_content = fs::read_to_string(workflow_path)?;

    // Parse JSON
    let workflow_data: WorkflowFile = serde_json::from_str(&workflow_content)?;

    println!("✓ Workflow file parsed successfully");
    println!("  Name: {}", workflow_data.name);
    println!("  Description: {}", workflow_data.description);
    println!("  Nodes: {}", workflow_data.nodes.len());
    println!("  Edges: {}", workflow_data.edges.len());

    // Convert to internal workflow representation
    let workflow = convert_workflow_data(workflow_data)?;

    // Validate the workflow structure
    workflow.validate()?;

    println!("✓ Workflow structure is valid");
    println!("✓ Graph is acyclic and properly connected");

    Ok(())
}

/// Convert workflow file data to internal workflow representation
pub fn convert_workflow_data(data: WorkflowFile) -> GraphBitResult<Workflow> {
    let mut workflow = Workflow::new(data.name, data.description);

    // Add metadata
    for (key, value) in data.metadata {
        workflow.set_metadata(key, value);
    }

    // Create a map of string IDs to NodeIds for edge creation
    let mut node_id_map = HashMap::with_capacity(data.nodes.len().max(8));

    // Add nodes
    for node_data in data.nodes {
        let node_id = NodeId::from_string(&node_data.id)
            .map_err(|_| GraphBitError::graph("Invalid node ID format"))?;

        let node_type = match node_data.node_type.as_str() {
            "agent" => {
                let agent_id = node_data
                    .config
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| GraphBitError::graph("Agent node missing agent_id"))?;

                let prompt = node_data
                    .config
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                NodeType::Agent {
                    agent_id: AgentId::from_string(agent_id)
                        .map_err(|_| GraphBitError::graph("Invalid agent ID format"))?,
                    prompt_template: prompt,
                }
            }
            "condition" => {
                let expression = node_data
                    .config
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("true")
                    .to_string();
                NodeType::Condition { expression }
            }
            "transform" => {
                let transformation = node_data
                    .config
                    .get("transformation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("identity")
                    .to_string();
                NodeType::Transform { transformation }
            }
            "delay" => {
                let duration = node_data
                    .config
                    .get("duration_seconds")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);
                NodeType::Delay {
                    duration_seconds: duration,
                }
            }
            _ => {
                return Err(GraphBitError::graph(format!(
                    "Unsupported node type: {}",
                    node_data.node_type
                )))
            }
        };

        let mut node = WorkflowNode::new(node_data.name, node_data.description, node_type);
        node.id = node_id.clone();

        node_id_map.insert(node_data.id, node_id.clone());
        workflow.add_node(node)?;
    }

    // Add edges
    for edge_data in data.edges {
        let from_id = node_id_map
            .get(&edge_data.from)
            .ok_or_else(|| GraphBitError::graph("Invalid from node ID in edge"))?;
        let to_id = node_id_map
            .get(&edge_data.to)
            .ok_or_else(|| GraphBitError::graph("Invalid to node ID in edge"))?;

        let edge = match edge_data.edge_type.as_str() {
            "data_flow" => WorkflowEdge::data_flow(),
            "control_flow" => WorkflowEdge::control_flow(),
            "conditional" => {
                let condition = edge_data.condition.unwrap_or_else(|| "true".to_string());
                WorkflowEdge::conditional(condition)
            }
            _ => {
                return Err(GraphBitError::graph(format!(
                    "Unsupported edge type: {}",
                    edge_data.edge_type
                )))
            }
        };

        workflow.connect_nodes(from_id.clone(), to_id.clone(), edge)?;
    }

    Ok(workflow)
}

/// Extract agent IDs from a workflow
pub fn extract_agent_ids(workflow: &Workflow) -> Vec<String> {
    let mut agent_ids = Vec::new();

    for node in workflow.graph.get_nodes().values() {
        if let NodeType::Agent { agent_id, .. } = &node.node_type {
            agent_ids.push(agent_id.to_string());
        }
    }

    agent_ids.sort();
    agent_ids.dedup();
    agent_ids
}
