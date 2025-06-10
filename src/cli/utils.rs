//! Utility functions for GraphBit CLI
//!
//! This module contains helper functions used across different CLI modules.

use graphbit_core::{errors::GraphBitError, GraphBitResult};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Substitute environment variables in configuration content
pub fn substitute_env_vars(content: &str) -> GraphBitResult<String> {
    let mut result = content.to_string();
    let mut found_vars = Vec::new();

    // Find all environment variable references
    while let Some(start) = result.find("${") {
        if let Some(end) = result[start..].find('}') {
            let var_name = &result[start + 2..start + end];
            found_vars.push(var_name.to_string());

            match std::env::var(var_name) {
                Ok(env_value) => {
                    if env_value.is_empty() {
                        return Err(GraphBitError::config(format!(
                            "Environment variable '{}' is empty",
                            var_name
                        )));
                    }
                    result.replace_range(start..start + end + 1, &env_value);
                }
                Err(_) => {
                    return Err(GraphBitError::config(format!(
                        "Environment variable '{}' not found. Set it with: export {}=your-value",
                        var_name, var_name
                    )));
                }
            }
        } else {
            return Err(GraphBitError::config(
                "Malformed environment variable reference (missing closing brace)".to_string(),
            ));
        }
    }

    // Report which environment variables were substituted
    if !found_vars.is_empty() {
        println!("✓ Loaded environment variables: {}", found_vars.join(", "));
    }

    Ok(result)
}

/// Validate that an API key is not a placeholder
pub fn validate_api_key(api_key: &str) -> GraphBitResult<()> {
    // Check for common placeholder patterns
    let invalid_patterns = [
        "your-api-key",
        "your-openai-key",
        "your-anthropic-key",
        "your-ollama-key",
        "dummy-key",
        "placeholder",
        "REPLACE_ME",
        "${",
    ];

    let api_key_lower = api_key.to_lowercase();
    for pattern in &invalid_patterns {
        if api_key_lower.contains(pattern) {
            return Err(GraphBitError::config(
                format!(
                    "API key appears to be a placeholder: '{}'. Please set a valid API key in your environment variables.",
                    api_key
                )
            ));
        }
    }

    // Check minimum length (most real API keys are longer)
    if api_key.len() < 10 {
        return Err(GraphBitError::config(
            "API key appears to be too short. Please verify your API key is correct.".to_string(),
        ));
    }

    Ok(())
}

/// Check if Ollama is running and accessible
pub async fn check_ollama_availability(base_url: Option<&str>) -> GraphBitResult<bool> {
    let base_url = base_url.unwrap_or("http://localhost:11434");
    let client = Client::new();
    let url = format!("{}/api/tags", base_url);

    match client.get(&url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// List available Ollama models
pub async fn list_ollama_models(base_url: Option<&str>) -> GraphBitResult<Vec<String>> {
    let base_url = base_url.unwrap_or("http://localhost:11434");
    let client = Client::new();
    let url = format!("{}/api/tags", base_url);

    let response = client.get(&url).send().await.map_err(|e| {
        GraphBitError::config(format!(
            "Failed to connect to Ollama at {}: {}. Please make sure Ollama is running.",
            base_url, e
        ))
    })?;

    if !response.status().is_success() {
        return Err(GraphBitError::config(format!(
            "Ollama returned HTTP {}: Please check that Ollama is running properly.",
            response.status()
        )));
    }

    let models_response: OllamaModelsResponse = response.json().await.map_err(|e| {
        GraphBitError::config(format!("Failed to parse Ollama models response: {}", e))
    })?;

    Ok(models_response.models.into_iter().map(|m| m.name).collect())
}

/// Pull an Ollama model if it doesn't exist
pub async fn ensure_ollama_model(model: &str, base_url: Option<&str>) -> GraphBitResult<()> {
    let base_url = base_url.unwrap_or("http://localhost:11434");

    // First check if model exists
    let models = list_ollama_models(Some(base_url)).await?;
    if models.iter().any(|m| m == model) {
        println!("✓ Ollama model '{}' is available", model);
        return Ok(());
    }

    println!(
        "Model '{}' not found locally. Pulling from Ollama registry...",
        model
    );

    let client = Client::new();
    let url = format!("{}/api/pull", base_url);
    let pull_request = OllamaPullRequest {
        name: model.to_string(),
    };

    let response = client
        .post(&url)
        .json(&pull_request)
        .send()
        .await
        .map_err(|e| {
            GraphBitError::config(format!("Failed to pull Ollama model '{}': {}", model, e))
        })?;

    if !response.status().is_success() {
        return Err(GraphBitError::config(format!(
            "Failed to pull Ollama model '{}': HTTP {}",
            model,
            response.status()
        )));
    }

    println!("✓ Successfully pulled Ollama model '{}'", model);
    Ok(())
}

/// Validate Ollama configuration
pub async fn validate_ollama_config(model: &str, base_url: Option<&str>) -> GraphBitResult<()> {
    // Check if Ollama is available
    if !check_ollama_availability(base_url).await? {
        let base_url = base_url.unwrap_or("http://localhost:11434");
        return Err(GraphBitError::config(format!(
            "Ollama is not available at {}. Please start Ollama by running 'ollama serve'",
            base_url
        )));
    }

    // Ensure the model is available
    ensure_ollama_model(model, base_url).await?;

    Ok(())
}

/// Get popular Ollama model recommendations
pub fn get_ollama_model_recommendations() -> Vec<(&'static str, &'static str)> {
    vec![
        ("llama3.1", "Latest Llama 3.1 model (8B parameters)"),
        ("llama3.1:70b", "Llama 3.1 large model (70B parameters)"),
        ("llama3", "Llama 3 model (8B parameters)"),
        ("gemma2", "Google Gemma 2 model"),
        ("qwen2.5", "Alibaba Qwen 2.5 model"),
        ("codellama", "Code-focused Llama model"),
        ("mistral", "Mistral 7B model"),
        ("mixtral:8x7b", "Mixtral mixture of experts model"),
        ("phi3", "Microsoft Phi-3 model"),
        ("nomic-embed-text", "Text embedding model"),
    ]
}

// Structures for Ollama API
#[derive(Debug, Serialize)]
struct OllamaPullRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}
