//! LLM provider abstraction for GraphBit
//!
//! This module provides a unified interface for working with different
//! LLM providers while maintaining strong type safety and validation.

pub mod anthropic;
pub mod huggingface;
pub mod ollama;
pub mod openai;
pub mod providers;
pub mod response;

pub use providers::{LlmConfig, LlmProvider, LlmProviderTrait};
pub use response::{FinishReason, LlmResponse, LlmUsage};

use crate::errors::{GraphBitError, GraphBitResult};
use crate::validation::{TypeValidator, ValidationResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// LLM request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// The prompt or messages to send to the LLM
    pub messages: Vec<LlmMessage>,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for response randomness (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// List of tools available to the LLM
    pub tools: Vec<LlmTool>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl LlmRequest {
    /// Create a new LLM request with a simple text prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![LlmMessage::user(prompt)],
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Vec::with_capacity(4), // Pre-allocate small capacity
            extra_params: HashMap::with_capacity(4), // Pre-allocate small capacity
        }
    }

    /// Create a new LLM request with multiple messages
    pub fn with_messages(messages: Vec<LlmMessage>) -> Self {
        Self {
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Vec::with_capacity(4),
            extra_params: HashMap::with_capacity(4),
        }
    }

    /// Add a message to the request
    #[inline]
    pub fn with_message(mut self, message: LlmMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Set maximum tokens
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    #[inline]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 1.0));
        self
    }

    /// Set top-p
    #[inline]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// Add a tool
    #[inline]
    pub fn with_tool(mut self, tool: LlmTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools
    pub fn with_tools(mut self, tools: Vec<LlmTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add extra parameters
    #[inline]
    pub fn with_extra_param(mut self, key: String, value: serde_json::Value) -> Self {
        self.extra_params.insert(key, value);
        self
    }

    /// Get total message length estimate for performance planning
    pub fn estimated_token_count(&self) -> usize {
        self.messages
            .iter()
            .map(|msg| msg.content.len() / 4) // Rough estimate: 4 chars per token
            .sum()
    }
}

/// Individual message in an LLM conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    /// Role of the message sender
    pub role: LlmRole,
    /// Content of the message
    pub content: String,
    /// Optional tool calls in this message
    pub tool_calls: Vec<LlmToolCall>,
}

impl LlmMessage {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: LlmRole::User,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: LlmRole::Assistant,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: LlmRole::System,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Add tool calls to the message
    #[inline]
    pub fn with_tool_calls(mut self, tool_calls: Vec<LlmToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Get content length for performance estimation
    #[inline]
    pub fn content_length(&self) -> usize {
        self.content.len()
    }
}

/// Role of a message sender
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmRole {
    User,
    Assistant,
    System,
    Tool,
}

/// Tool definition for LLM function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTool {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema for the tool parameters
    pub parameters: serde_json::Value,
}

impl LlmTool {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Tool call made by the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Name of the tool to call
    pub name: String,
    /// Parameters to pass to the tool
    pub parameters: serde_json::Value,
}

/// Batch request for multiple LLM calls to improve throughput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmBatchRequest {
    /// Multiple requests to process
    pub requests: Vec<LlmRequest>,
    /// Maximum concurrent requests to process
    pub max_concurrency: Option<usize>,
    /// Timeout for the entire batch in milliseconds
    pub timeout_ms: Option<u64>,
}

impl LlmBatchRequest {
    /// Create a new batch request
    pub fn new(requests: Vec<LlmRequest>) -> Self {
        Self {
            requests,
            max_concurrency: Some(5), // Default to reasonable concurrency
            timeout_ms: Some(30000),  // Default 30 second timeout
        }
    }

    /// Set maximum concurrency
    #[inline]
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency);
        self
    }

    /// Set timeout
    #[inline]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Batch response for multiple LLM calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmBatchResponse {
    /// Responses corresponding to the requests
    pub responses: Vec<Result<LlmResponse, GraphBitError>>,
    /// Total processing time in milliseconds
    pub total_duration_ms: u64,
    /// Statistics about the batch processing
    pub stats: LlmBatchStats,
}

/// Statistics for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmBatchStats {
    /// Number of successful requests
    pub successful_requests: usize,
    /// Number of failed requests
    pub failed_requests: usize,
    /// Average response time per request
    pub avg_response_time_ms: f64,
    /// Peak concurrent requests
    pub peak_concurrency: usize,
}

/// Factory for creating LLM providers
pub struct LlmProviderFactory;

impl LlmProviderFactory {
    /// Create a new LLM provider from configuration
    pub fn create_provider(config: LlmConfig) -> GraphBitResult<Box<dyn LlmProviderTrait>> {
        match config {
            LlmConfig::OpenAI { api_key, model, .. } => {
                Ok(Box::new(openai::OpenAiProvider::new(api_key, model)?))
            }
            LlmConfig::Anthropic { api_key, model, .. } => {
                Ok(Box::new(anthropic::AnthropicProvider::new(api_key, model)?))
            }
            LlmConfig::HuggingFace {
                api_key,
                model,
                base_url,
                ..
            } => {
                if let Some(base_url) = base_url {
                    Ok(Box::new(huggingface::HuggingFaceProvider::with_base_url(
                        api_key, model, base_url,
                    )?))
                } else {
                    Ok(Box::new(huggingface::HuggingFaceProvider::new(
                        api_key, model,
                    )?))
                }
            }
            LlmConfig::Ollama {
                model, base_url, ..
            } => {
                if let Some(base_url) = base_url {
                    Ok(Box::new(ollama::OllamaProvider::with_base_url(
                        model, base_url,
                    )?))
                } else {
                    Ok(Box::new(ollama::OllamaProvider::new(model)?))
                }
            }
            LlmConfig::Custom { provider_type, .. } => Err(GraphBitError::config(format!(
                "Unsupported custom provider: {}",
                provider_type
            ))),
        }
    }

    /// Create multiple providers for load balancing
    pub fn create_providers(
        configs: Vec<LlmConfig>,
    ) -> GraphBitResult<Vec<Box<dyn LlmProviderTrait>>> {
        configs
            .into_iter()
            .map(Self::create_provider)
            .collect::<GraphBitResult<Vec<_>>>()
    }
}

/// Enhanced trait for LLM provider with batch processing capabilities
#[async_trait]
pub trait EnhancedLlmProviderTrait: LlmProviderTrait {
    /// Process multiple requests concurrently
    async fn process_batch(&self, batch: LlmBatchRequest) -> GraphBitResult<LlmBatchResponse> {
        let start_time = std::time::Instant::now();
        let max_concurrency = batch.max_concurrency.unwrap_or(5);

        // Use atomic counter instead of semaphore to eliminate bottleneck
        let current_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut tasks = Vec::with_capacity(batch.requests.len());

        for request in batch.requests {
            let current_requests = Arc::clone(&current_requests);
            let provider = self.clone_provider();
            let task = tokio::spawn(async move {
                // Wait for available slot using atomic operations (no semaphore bottleneck)
                loop {
                    let current = current_requests.load(std::sync::atomic::Ordering::Acquire);
                    if current < max_concurrency {
                        match current_requests.compare_exchange(
                            current,
                            current + 1,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        ) {
                            Ok(_) => break,     // Successfully acquired slot
                            Err(_) => continue, // Retry
                        }
                    } else {
                        // Brief yield to avoid busy waiting
                        tokio::task::yield_now().await;
                    }
                }

                // Execute the request
                let result = provider.complete(request).await;

                // Release slot
                current_requests.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);

                result
            });

            tasks.push(task);
        }

        // Wait for all tasks with optional timeout
        let responses = if let Some(timeout_ms) = batch.timeout_ms {
            let timeout_duration = tokio::time::Duration::from_millis(timeout_ms);
            match tokio::time::timeout(timeout_duration, futures::future::join_all(tasks)).await {
                Ok(results) => results,
                Err(_) => return Err(GraphBitError::llm("Batch request timed out".to_string())),
            }
        } else {
            futures::future::join_all(tasks).await
        };

        // Process results
        let mut successful = 0;
        let mut failed = 0;
        let final_responses: Vec<Result<LlmResponse, GraphBitError>> = responses
            .into_iter()
            .map(|task_result| match task_result {
                Ok(llm_result) => match llm_result {
                    Ok(response) => {
                        successful += 1;
                        Ok(response)
                    }
                    Err(e) => {
                        failed += 1;
                        Err(e)
                    }
                },
                Err(e) => {
                    failed += 1;
                    Err(GraphBitError::llm(format!("Task execution failed: {}", e)))
                }
            })
            .collect();

        let total_duration_ms = start_time.elapsed().as_millis() as u64;
        let avg_response_time_ms = if total_duration_ms > 0 && successful > 0 {
            total_duration_ms as f64 / successful as f64
        } else {
            0.0
        };

        Ok(LlmBatchResponse {
            responses: final_responses,
            total_duration_ms,
            stats: LlmBatchStats {
                successful_requests: successful,
                failed_requests: failed,
                avg_response_time_ms,
                peak_concurrency: max_concurrency,
            },
        })
    }

    /// Clone the provider for concurrent usage
    fn clone_provider(&self) -> Box<dyn LlmProviderTrait>;

    /// Get provider statistics for monitoring
    async fn get_provider_stats(&self) -> GraphBitResult<serde_json::Value> {
        Ok(serde_json::json!({
            "provider_type": "unknown",
            "total_requests": 0,
            "avg_response_time_ms": 0.0
        }))
    }
}

/// Trait for LLM response validation
#[async_trait]
pub trait LlmResponseValidator: Send + Sync {
    /// Validate an LLM response against expected schema
    async fn validate_response(
        &self,
        response: &LlmResponse,
        expected_schema: &serde_json::Value,
    ) -> ValidationResult;

    /// Extract structured data from LLM response
    async fn extract_structured_data(
        &self,
        response: &LlmResponse,
        target_type: &str,
    ) -> GraphBitResult<serde_json::Value>;
}

/// Default LLM response validator
pub struct DefaultLlmResponseValidator {
    type_validator: TypeValidator,
}

impl DefaultLlmResponseValidator {
    /// Create a new default validator
    pub fn new() -> Self {
        Self {
            type_validator: TypeValidator::new(),
        }
    }
}

impl Default for DefaultLlmResponseValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmResponseValidator for DefaultLlmResponseValidator {
    async fn validate_response(
        &self,
        response: &LlmResponse,
        expected_schema: &serde_json::Value,
    ) -> ValidationResult {
        self.type_validator
            .validate_against_schema(&response.content, expected_schema)
    }

    async fn extract_structured_data(
        &self,
        response: &LlmResponse,
        target_type: &str,
    ) -> GraphBitResult<serde_json::Value> {
        // Try to parse JSON from the response content
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&response.content) {
            return Ok(json_value);
        }

        // If direct parsing fails, try to extract JSON from text
        self.extract_json_from_text(&response.content, target_type)
    }
}

impl DefaultLlmResponseValidator {
    /// Extract JSON data from text content
    fn extract_json_from_text(
        &self,
        text: &str,
        _target_type: &str,
    ) -> GraphBitResult<serde_json::Value> {
        // Look for JSON blocks in the text
        let json_start_markers = ["```json", "```JSON", "{"];
        let json_end_markers = ["```", "}"];

        for start_marker in &json_start_markers {
            if let Some(start_pos) = text.find(start_marker) {
                let json_start = if start_marker.starts_with("```") {
                    start_pos + start_marker.len()
                } else {
                    start_pos
                };

                for end_marker in &json_end_markers {
                    if let Some(end_pos) = text[json_start..].find(end_marker) {
                        let json_end = json_start + end_pos;
                        let json_str = if *end_marker == "}" {
                            &text[json_start..json_end + 1]
                        } else {
                            &text[json_start..json_end]
                        };

                        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str)
                        {
                            return Ok(json_value);
                        }
                    }
                }
            }
        }

        Err(GraphBitError::validation(
            "content",
            "No valid JSON found in response",
        ))
    }
}
