//! Project initialization for GraphBit CLI
//!
//! This module handles creating new GraphBit projects with proper structure.

use crate::cli::utils::get_ollama_model_recommendations;
use graphbit_core::GraphBitResult;
use std::fs;
use std::path::Path;

/// Create a new GraphBit project structure
pub fn create_project_structure(path: &Path, name: &str) -> GraphBitResult<()> {
    let project_dir = path.join(name);
    fs::create_dir_all(&project_dir)?;

    // Create basic project structure
    fs::create_dir_all(project_dir.join("workflows"))?;
    fs::create_dir_all(project_dir.join("agents"))?;
    fs::create_dir_all(project_dir.join("config"))?;

    // Create example workflow
    let example_workflow = r#"{
  "name": "Example Workflow",
  "description": "A simple example workflow demonstrating agent interaction",
  "nodes": [
    {
      "id": "start",
      "type": "agent",
      "name": "Greeting Agent",
      "description": "Generates a friendly greeting",
      "config": {
        "prompt": "Generate a friendly greeting for a new user of GraphBit",
        "agent_id": "greeting-agent"
      }
    },
    {
      "id": "processor",
      "type": "agent",
      "name": "Processing Agent",
      "description": "Processes the greeting and adds information",
      "config": {
        "prompt": "Take the greeting and add helpful information about GraphBit workflow automation",
        "agent_id": "processor-agent"
      }
    }
  ],
  "edges": [
    {
      "from": "start",
      "to": "processor",
      "type": "data_flow"
    }
  ]
}"#;

    fs::write(
        project_dir.join("workflows").join("example.json"),
        example_workflow,
    )?;

    // Create example config
    let example_config = r#"{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}"
  },
  "agents": {
    "default": {
      "max_tokens": 1000,
      "temperature": 0.7,
      "system_prompt": "You are a helpful AI assistant in the GraphBit workflow automation system.",
      "capabilities": ["text_processing", "data_analysis"]
    }
  }
}"#;

    fs::write(
        project_dir.join("config").join("example.json"),
        example_config,
    )?;

    // Create Ollama config example with model recommendations
    let recommendations = get_ollama_model_recommendations();
    let mut ollama_config = String::from(
        r#"{
  "llm": {
    "provider": "ollama",
    "model": "llama3.1",
    "base_url": "http://localhost:11434"
  },
  "agents": {
    "default": {
      "max_tokens": 1000,
      "temperature": 0.7,
      "system_prompt": "You are a helpful AI assistant running locally via Ollama. Provide clear, accurate, and actionable responses.",
      "capabilities": ["text_processing", "data_analysis"]
    }
  }
}

// Popular Ollama models you can use:
"#,
    );

    for (model, description) in recommendations {
        ollama_config.push_str(&format!("// • {} - {}\n", model, description));
    }

    ollama_config.push_str(
        r#"//
// To use any of these models:
// 1. Start Ollama: ollama serve
// 2. Pull the model: ollama pull <model-name>
// 3. Update the "model" field above with your chosen model
"#,
    );

    fs::write(
        project_dir.join("config").join("ollama.json"),
        ollama_config,
    )?;

    // Create .env.example file for development
    let env_example = r#"# GraphBit Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (required for OpenAI provider)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API Key (required for Anthropic provider)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Optional: Custom base URLs for API providers
# OPENAI_BASE_URL=https://api.openai.com/v1
# ANTHROPIC_BASE_URL=https://api.anthropic.com

# Ollama Configuration (no API key needed - runs locally)
# OLLAMA_BASE_URL=http://localhost:11434
"#;

    fs::write(project_dir.join(".env.example"), env_example)?;

    // Create .gitignore
    let gitignore = r#"# Environment files (contains sensitive API keys)
.env
.env.local
.env.*.local

# Logs
*.log
logs/

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Build outputs
target/
dist/
build/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"#;

    fs::write(project_dir.join(".gitignore"), gitignore)?;

    // Create README.md
    let readme = format!(
        r#"# {}

A GraphBit workflow automation project.

## Setup

1. Copy the environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. Test the example workflow:
   ```bash
   graphbit validate workflows/example.json
   graphbit run workflows/example.json --config config/example.json
   ```

## Project Structure

- `workflows/` - Workflow definition files (JSON)
- `config/` - Configuration files for different environments
- `agents/` - Custom agent implementations (if any)

## Environment Variables

Set these environment variables in your `.env` file:

- `OPENAI_API_KEY` - Your OpenAI API key (for OpenAI provider)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (for Anthropic provider)

## Getting Started

1. Create your workflow in the `workflows/` directory
2. Configure your agents and LLM settings in `config/`
3. Run your workflow with `graphbit run`

For more information, see the [GraphBit documentation](https://github.com/InfinitiBit/graphbit).
"#,
        name
    );

    fs::write(project_dir.join("README.md"), readme)?;

    Ok(())
}
