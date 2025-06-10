//! Configuration management for GraphBit CLI
//!
//! This module handles loading, parsing, and validating configuration files.

use crate::cli::utils::{substitute_env_vars, validate_api_key, validate_ollama_config};
use graphbit_core::{errors::GraphBitError, GraphBitResult, LlmConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Configuration file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfigData,
    pub agents: HashMap<String, AgentConfigData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfigData {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigData {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

fn default_max_tokens() -> u32 {
    1000
}

fn default_temperature() -> f32 {
    0.7
}

/// Load configuration from file with environment variable substitution
pub async fn load_config(config_path: Option<&PathBuf>) -> GraphBitResult<Config> {
    let config_path = config_path
        .cloned()
        .unwrap_or_else(|| PathBuf::from("config/default.json"));

    if !config_path.exists() {
        return Err(GraphBitError::config(format!(
            "Configuration file not found: {:?}",
            config_path
        )));
    }

    let config_content = fs::read_to_string(config_path)?;

    // Substitute environment variables
    let config_content = substitute_env_vars(&config_content)?;

    let config: Config = serde_json::from_str(&config_content)?;

    // Validate configuration based on provider
    match config.llm.provider.as_str() {
        "openai" | "anthropic" => {
            // Validate API key is not a placeholder for cloud providers
            if let Some(ref api_key) = config.llm.api_key {
                validate_api_key(api_key)?;
            } else {
                return Err(GraphBitError::config(format!(
                    "API key is required for {} provider",
                    config.llm.provider
                )));
            }
        }
        "ollama" => {
            // Validate Ollama is available and model exists
            validate_ollama_config(&config.llm.model, config.llm.base_url.as_deref()).await?;
        }
        _ => {
            return Err(GraphBitError::config(format!(
                "Unsupported LLM provider: {}",
                config.llm.provider
            )));
        }
    }

    Ok(config)
}

/// Convert LlmConfigData to core LlmConfig
pub fn create_llm_config(llm_data: &LlmConfigData) -> GraphBitResult<LlmConfig> {
    match llm_data.provider.as_str() {
        "openai" => {
            let api_key = llm_data.api_key.as_ref().ok_or_else(|| {
                GraphBitError::config("API key is required for OpenAI provider".to_string())
            })?;
            Ok(LlmConfig::openai(api_key, &llm_data.model))
        }
        "anthropic" => {
            let api_key = llm_data.api_key.as_ref().ok_or_else(|| {
                GraphBitError::config("API key is required for Anthropic provider".to_string())
            })?;
            Ok(LlmConfig::anthropic(api_key, &llm_data.model))
        }
        "ollama" => {
            if let Some(ref base_url) = llm_data.base_url {
                Ok(LlmConfig::ollama_with_base_url(&llm_data.model, base_url))
            } else {
                Ok(LlmConfig::ollama(&llm_data.model))
            }
        }
        _ => Err(GraphBitError::config(format!(
            "Unsupported LLM provider: {}",
            llm_data.provider
        ))),
    }
}
