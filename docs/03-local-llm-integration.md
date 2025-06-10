---
title: Ollama Integration Guide
description: Setting up and using Ollama for local LLM inference with GraphBit
---

# Ollama Integration Guide

This guide covers setting up and using Ollama for local LLM inference with GraphBit.

## What is Ollama?

Ollama is a tool for running large language models locally on your machine. It provides:
- **Privacy**: Models run entirely on your hardware
- **No API costs**: No per-token charges
- **Offline capability**: Works without internet connection
- **Speed**: Direct hardware access for faster inference
- **Model variety**: Support for many popular open-source models

## Installation

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai) and download the installer for your platform:

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and run the installer from the website.

### 2. Start Ollama Server

```bash
ollama serve
```

The server will start on `http://localhost:11434` by default.

## Model Management

### Popular Models for GraphBit

| Model | Size | Use Case | Command |
|-------|------|----------|---------|
| llama3.1 | 4.7GB | General purpose | `ollama pull llama3.1` |
| llama3.1:70b | 40GB | High-quality responses | `ollama pull llama3.1:70b` |
| codellama | 3.8GB | Code generation | `ollama pull codellama` |
| mistral | 4.1GB | Fast responses | `ollama pull mistral` |
| gemma2 | 5.4GB | Google's model | `ollama pull gemma2` |
| qwen2.5 | 4.4GB | Multilingual | `ollama pull qwen2.5` |
| phi3 | 2.3GB | Microsoft's compact model | `ollama pull phi3` |

### Pulling Models

```bash
# Pull a specific model
ollama pull llama3.1

# List available models
ollama list

# Remove a model
ollama rm llama3.1
```

### Model Variants

Models come in different sizes:
```bash
# Default size (usually 7-8B parameters)
ollama pull llama3.1

# Specific sizes
ollama pull llama3.1:70b    # 70 billion parameters
ollama pull llama3.1:8b     # 8 billion parameters
ollama pull llama3.1:13b    # 13 billion parameters
```

## GraphBit Configuration

### Basic Configuration

Create or modify your `config/default.json`:

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3.1",
    "base_url": "http://localhost:11434"
  },
  "agents": {
    "default": {
      "max_tokens": 1000,
      "temperature": 0.7,
      "system_prompt": "You are a helpful AI assistant running locally.",
      "capabilities": ["text_processing", "data_analysis"]
    }
  }
}
```

### Advanced Configuration

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3.1",
    "base_url": "http://localhost:11434",
    "timeout_seconds": 60,
    "retry_attempts": 3
  },
  "agents": {
    "coder": {
      "max_tokens": 2000,
      "temperature": 0.3,
      "system_prompt": "You are an expert programmer. Write clean, efficient code.",
      "capabilities": ["code_generation", "debugging"]
    },
    "analyst": {
      "max_tokens": 1500,
      "temperature": 0.1,
      "system_prompt": "You are a data analyst. Provide accurate, factual analysis.",
      "capabilities": ["data_analysis", "reporting"]
    }
  }
}
```

### Multi-Model Configuration

You can configure different models for different agents:

```json
{
  "llm": {
    "provider": "ollama",
    "base_url": "http://localhost:11434"
  },
  "agents": {
    "fast_agent": {
      "model": "phi3",
      "max_tokens": 500,
      "temperature": 0.8
    },
    "quality_agent": {
      "model": "llama3.1:70b",
      "max_tokens": 2000,
      "temperature": 0.3
    },
    "code_agent": {
      "model": "codellama",
      "max_tokens": 1500,
      "temperature": 0.2
    }
  }
}
```

## Performance Optimization

### Hardware Requirements

| Model Size | RAM Required | VRAM (GPU) | Performance |
|------------|--------------|-------------|-------------|
| 3B params  | 4GB          | 2GB         | Fast        |
| 7B params  | 8GB          | 4GB         | Good        |
| 13B params | 16GB         | 8GB         | Better      |
| 30B params | 32GB         | 16GB        | High        |
| 70B params | 64GB         | 32GB        | Best        |

### Optimization Settings

#### CPU Optimization
```bash
# Set number of CPU threads
export OLLAMA_NUM_THREADS=8

# Enable CPU optimizations
export OLLAMA_CPU_TARGET="native"
```

#### GPU Acceleration
```bash
# For NVIDIA GPUs
export OLLAMA_GPU_TARGET="cuda"

# For AMD GPUs
export OLLAMA_GPU_TARGET="rocm"

# For Metal (macOS)
export OLLAMA_GPU_TARGET="metal"
```

#### Memory Management
```bash
# Set context window size
export OLLAMA_MAX_TOKENS=4096

# Control memory usage
export OLLAMA_MAX_MEMORY="8GB"
```

## Usage Examples

### Simple Workflow

```json
{
  "name": "Local Analysis",
  "description": "Data analysis using local Ollama model",
  "nodes": [
    {
      "id": "analyzer",
      "type": "agent",
      "name": "Data Analyzer",
      "config": {
        "agent_id": "default",
        "prompt": "Analyze this data and provide insights: {input}"
      }
    }
  ]
}
```

### Multi-Model Workflow

```json
{
  "name": "Code Review Pipeline",
  "description": "Code review using specialized models",
  "nodes": [
    {
      "id": "checker",
      "type": "agent",
      "name": "Syntax Checker",
      "config": {
        "agent_id": "code_agent",
        "prompt": "Check this code for syntax errors: {code}"
      }
    },
    {
      "id": "reviewer",
      "type": "agent", 
      "name": "Code Reviewer",
      "config": {
        "agent_id": "quality_agent",
        "prompt": "Review this code for best practices and improvements: {code}"
      }
    }
  ],
  "edges": [
    {"from": "checker", "to": "reviewer", "type": "data_flow"}
  ]
}
```

## Troubleshooting

### Common Issues

#### Ollama Server Not Running
```bash
# Check if server is running
curl http://localhost:11434/api/tags

# Start server if not running
ollama serve
```

#### Model Not Found
```bash
# Check available models
ollama list

# Pull missing model
ollama pull llama3.1
```

#### Connection Timeouts
```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3.1",
    "base_url": "http://localhost:11434",
    "timeout_seconds": 120,
    "retry_attempts": 5
  }
}
```

#### Out of Memory Errors
```bash
# Use smaller model
ollama pull phi3

# Or reduce context size
export OLLAMA_MAX_TOKENS=2048
```

### Performance Issues

#### Slow Responses
1. **Use smaller model**: Switch from 70B to 7B model
2. **Enable GPU**: Configure GPU acceleration
3. **Reduce context**: Lower max_tokens setting
4. **Optimize prompt**: Use shorter, more focused prompts

#### High Memory Usage
1. **Use quantized models**: Choose 4-bit or 8-bit variants
2. **Reduce concurrent requests**: Limit parallel workflows
3. **Clear model cache**: `ollama rm <model>` and re-pull

### Debug Mode

Enable verbose logging:

```bash
# Set debug level
export OLLAMA_DEBUG=1
export RUST_LOG=debug

# Run GraphBit with debug info
graphbit run workflows/example.json --verbose
```

## Best Practices

### Model Selection
- **Development**: Use smaller models (phi3, mistral) for faster iteration
- **Production**: Use larger models (llama3.1:70b) for better quality
- **Code tasks**: Use specialized models (codellama)
- **Analysis**: Use instruction-tuned models (llama3.1)

### Configuration
- Start with lower temperature (0.1-0.3) for consistent outputs
- Use appropriate max_tokens for your use case
- Set reasonable timeout values (30-120 seconds)
- Configure retry logic for robustness

### Resource Management
- Monitor CPU/GPU usage during inference
- Use model caching to avoid re-loading
- Scale horizontally with multiple Ollama instances if needed
- Consider using different models for different workflow stages

### Security
- Run Ollama in isolated environment for production
- Restrict network access to Ollama port (11434)
- Use local models to avoid data leaving your infrastructure
- Regularly update Ollama and models for security patches

## Advanced Topics

### Custom Models

You can use custom fine-tuned models with Ollama:

```bash
# Create custom model from Modelfile
ollama create mymodel -f Modelfile

# Use in GraphBit
{
  "llm": {
    "provider": "ollama",
    "model": "mymodel"
  }
}
```

### Distributed Setup

Run Ollama on multiple machines:

```json
{
  "llm": {
    "provider": "ollama",
    "base_url": "http://ollama-server-1:11434"
  },
  "agents": {
    "heavy_agent": {
      "base_url": "http://ollama-server-2:11434",
      "model": "llama3.1:70b"
    }
  }
}
```

### Monitoring

Monitor Ollama performance:

```bash
# Check model status
curl http://localhost:11434/api/ps

# Monitor resource usage
htop
nvidia-smi  # For GPU monitoring
```

---

For more information, visit the [Ollama documentation](https://github.com/ollama/ollama) or join our [Discord community](https://discord.gg/graphbit). 
