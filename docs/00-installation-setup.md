# 🚀 Installation & Setup

Get GraphBit running in your environment quickly with this step-by-step setup guide.

## ⚡ Quick Start

### 1. Install GraphBit
```bash
pip install graphbit
```

### 2. Set Up Your API Key
```bash
# For OpenAI (most common)
export OPENAI_API_KEY="your-openai-api-key"

# Or create a .env file
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

### 3. Verify Installation
```python
import graphbit
import os

# Initialize and test
graphbit.init()
config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")
print("✅ GraphBit is ready!")
```

**Ready to build?** → [Start with your first workflow](01-getting-started-workflows.md)

---

## 📦 Installation Methods

### From PyPI (Recommended)
Best for most users who want stable releases:

```bash
pip install graphbit
```

### From Conda
If you're using conda environments:

```bash
conda install -c conda-forge graphbit
```

### Development Installation
For contributors or bleeding-edge features:

```bash
git clone https://github.com/InfinitiBit/graphbit.git
cd graphbit

# Install Rust toolchain (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install with maturin
pip install maturin
maturin develop --release
```

---

## 🔑 LLM Provider Setup

GraphBit works with multiple LLM providers. Choose what works best for you:

### OpenAI (Recommended)

> **🔒 Security Warning**: Never commit API keys to version control. Always use environment variables or secure secret management systems.

```bash
# Get your API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_actual_api_key_here"

# Test the connection
python -c "
import graphbit
import os
graphbit.init()
config = graphbit.PyLlmConfig.openai(os.getenv('OPENAI_API_KEY'), 'gpt-4')
print('OpenAI connection ready!')
"
```

### Anthropic Claude
```bash
# Get your API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-..."

# Test the connection
python -c "
import graphbit
import os
graphbit.init()
config = graphbit.PyLlmConfig.anthropic(os.getenv('ANTHROPIC_API_KEY'), 'claude-3-sonnet')
print('Anthropic connection ready!')
"
```

### HuggingFace Models
Access thousands of open-source models:

```bash
# Get your API token from https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="hf_..."

# Test the connection
python -c "
import graphbit
import os
graphbit.init()
config = graphbit.PyLlmConfig.huggingface(os.getenv('HUGGINGFACE_API_KEY'), 'microsoft/DialoGPT-medium')
print('HuggingFace connection ready!')
"
```

**Popular Models:**
- `microsoft/DialoGPT-medium` - Conversational AI
- `gpt2` - Text generation  
- `facebook/blenderbot-400M-distill` - Chatbot
- `Salesforce/codegen-350M-mono` - Code generation

### Local Models with Ollama
Perfect for privacy-sensitive applications:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1

# Test with GraphBit (Note: Ollama support via custom configuration)
python -c "
import graphbit
graphbit.init()
# Note: Ollama requires custom implementation
print('GraphBit ready - configure Ollama separately!')
"
```

**Learn more**: [Local LLM Integration Guide](03-local-llm-integration.md)

---

## 🐍 Python Environment Setup

### Recommended: Conda Environment
For this project, we recommend using conda:

```bash
# Create and activate environment
conda create -n graphbit python=3.9
conda activate graphbit

# Install GraphBit
pip install graphbit

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Alternative: Virtual Environment
```bash
# Create virtual environment
python -m venv graphbit-env
source graphbit-env/bin/activate  # On Windows: graphbit-env\Scripts\activate

# Install GraphBit
pip install graphbit
```

### Development Setup
For GraphBit development:

```bash
# Clone and setup
git clone https://github.com/InfinitiBit/graphbit.git
cd graphbit

# Activate your conda environment
conda activate graphbit

# Set up development environment
pip install maturin
maturin develop --release

# Optional: Install development dependencies
pip install graphbit[dev]
```

---

## 🔧 Configuration Examples

### Environment Variables
```bash
# Essential
export OPENAI_API_KEY="your-openai-key"

# Optional: Multiple providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export HUGGINGFACE_API_KEY="your-hf-token"
export OPENAI_ORG_ID="your-org-id"

# Advanced: Custom endpoints
export OPENAI_BASE_URL="https://your-custom-endpoint.com"
```

### .env File Configuration
Create a `.env` file in your project:

```bash
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
HUGGINGFACE_API_KEY=your-hf-token

# Optional settings
GRAPHBIT_LOG_LEVEL=INFO
GRAPHBIT_MAX_RETRIES=3
```

### Python Configuration
```python
import graphbit
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize GraphBit
graphbit.init()

# Configure multiple providers
openai_config = graphbit.PyLlmConfig.openai(
    os.getenv("OPENAI_API_KEY"), 
    "gpt-4"
)

anthropic_config = graphbit.PyLlmConfig.anthropic(
    os.getenv("ANTHROPIC_API_KEY"), 
    "claude-3-sonnet-20240229"
)

huggingface_config = graphbit.PyLlmConfig.huggingface(
    os.getenv("HUGGINGFACE_API_KEY"), 
    "microsoft/DialoGPT-medium"
)

# Use in workflows as needed
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### "ModuleNotFoundError: No module named 'graphbit'"
```bash
# Check if GraphBit is installed
pip list | grep graphbit

# If not found, install it
pip install graphbit

# If using conda, make sure you're in the right environment
conda activate graphbit
```

#### Build Errors (Development)
```bash
# Update Rust toolchain
rustup update

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install build-essential

# macOS:
xcode-select --install

# Clean and rebuild
cargo clean
maturin develop --release
```

#### API Key Issues
```bash
# Verify your API key is set
echo $OPENAI_API_KEY

# Test API connection
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.models.list()
print('API key works!')
"
```

#### Permission Errors
```bash
# Use user installation
pip install --user graphbit

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install graphbit
```

### Performance Optimization

#### For Large Workflows
```python
# Configure high-performance execution
config = graphbit.PyLlmConfig.openai(api_key, "gpt-4")
executor = graphbit.PyWorkflowExecutor(config)

# Optimize pool settings
pool_config = graphbit.PyPoolConfig(
    max_size=50,
    min_size=10
)
executor.configure_pools(pool_config)

# Configure retries
retry_config = graphbit.PyRetryConfig(
    max_attempts=3,
    backoff_ms=1000
)
executor.configure_retries(retry_config)
```

---

## ✅ Verification Checklist

Make sure everything is working:

- [ ] GraphBit installed: `pip list | grep graphbit`
- [ ] Python can import: `python -c "import graphbit; print('✅ Import works')"`
- [ ] API key set: `echo $OPENAI_API_KEY` (should show your key)
- [ ] LLM connection works: Run the test scripts above
- [ ] First workflow runs: Try the [Getting Started guide](01-getting-started-workflows.md)

---

## 🎯 Next Steps

Once you have GraphBit installed and configured:

1. **[Build Your First Workflow](01-getting-started-workflows.md)** - 5-minute tutorial
2. **[Explore Use Cases](02-use-case-examples.md)** - Real-world examples
3. **[Set Up Local LLMs](03-local-llm-integration.md)** - Use Ollama for privacy
4. **[Complete API Reference](05-complete-api-reference.md)** - Full documentation

---

## 🆘 Need Help?

- **Issues & Bugs**: [GitHub Issues](https://github.com/InfinitiBit/graphbit/issues)
- **Questions**: [GitHub Discussions](https://github.com/InfinitiBit/graphbit/discussions)
- **Documentation**: Browse other guides in this docs folder 
