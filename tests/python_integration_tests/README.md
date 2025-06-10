# GraphBit Python Integration Tests

This directory contains comprehensive integration tests for the GraphBit Python library, covering all major functionality areas.

## Test Categories

### 1. Embedding Integration Tests (`tests_embeddings.py`)
- **OpenAI Embeddings**: Configuration, single/multiple text embedding, dimensions, consistency
- **HuggingFace Embeddings**: Configuration, text embedding, API integration
- **Utility Functions**: Cosine similarity, embedding requests/responses
- **Cross-Provider Comparison**: Semantic similarity across different providers

### 2. LLM Integration Tests (`tests_llm.py`)
- **OpenAI Models**: GPT-4, GPT-3.5 configuration and workflow execution
- **Anthropic Models**: Claude model configuration and integration
- **HuggingFace Models**: Open-source model integration
- **Configuration**: Retry policies, circuit breakers, pool configurations
- **Agent Capabilities**: Predefined and custom capabilities
- **Cross-Provider**: Multi-provider workflow validation

### 3. Static Workflow Integration Tests (`tests_static_workflow.py`)
- **Workflow Creation**: Basic workflow construction with various node types
- **Node Types**: Agent, condition, transform, delay, document loader nodes
- **Connections**: Data flow, control flow, conditional connections
- **Validation**: Workflow structure validation
- **Serialization**: JSON serialization/deserialization
- **Builder Pattern**: Workflow construction using builder pattern
- **Execution**: Synchronous and asynchronous workflow execution

### 4. Dynamic Workflow Integration Tests (`tests_dynamic_workflow.py`)
- **Dynamic Configuration**: Auto-node generation settings
- **Graph Manager**: Dynamic node generation and workflow completion
- **Analytics**: Performance tracking and generation statistics
- **Auto-Completion**: Intelligent workflow completion
- **Integration**: Hybrid static/dynamic workflows
- **Performance**: Scalability and optimization configurations

## Prerequisites

### Required Dependencies
```bash
# Install GraphBit with development dependencies
pip install -e .
pip install pytest pytest-asyncio
```

### API Keys (Optional but Recommended)
Set environment variables for full test coverage:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  
export HUGGINGFACE_API_KEY="your-huggingface-api-key"
```

**Note**: Tests will skip API-dependent functionality if keys are not provided.

## Running Tests

### Run All Integration Tests
```bash
# Using the test runner (recommended)
python test_runner.py

# Using pytest directly
pytest -v
```

### Run Specific Test Categories
```bash
# Embeddings only
python test_runner.py embedding

# LLM only  
python test_runner.py llm

# Static workflows only
python test_runner.py static

# Dynamic workflows only
python test_runner.py dynamic
```

### Run Individual Test Files
```bash
# Embedding tests
pytest tests_embeddings.py -v

# LLM tests
pytest tests_llm.py -v

# Static workflow tests
pytest tests_static_workflow.py -v

# Dynamic workflow tests
pytest tests_dynamic_workflow.py -v
```

### Run Specific Test Classes or Methods
```bash
# Run specific test class
pytest tests_embeddings.py::TestOpenAIEmbeddings -v

# Run specific test method
pytest tests_llm.py::TestOpenAILLM::test_openai_gpt4_config_creation -v

# Run tests matching pattern
pytest -k "embedding" -v
pytest -k "workflow and static" -v
```

## Build and Environment Setup

Before running tests, ensure GraphBit is properly built:

```bash
# Activate conda environment
conda activate graphbit

# Set OpenAI API key
export OPENAI_API_KEY=your_actual_api_key_here

# Build Rust components
cargo build --release

# Build Python bindings  
maturin develop --release
```

## Test Features

### Comprehensive Coverage
- **Configuration Testing**: All provider configurations and customizations
- **Functionality Testing**: Core features like embedding generation, LLM interactions
- **Integration Testing**: Cross-component interaction and workflow execution
- **Error Handling**: Graceful handling of API failures and missing keys
- **Performance Testing**: Basic performance and scalability considerations

### Smart Skipping
Tests automatically skip when:
- Required API keys are not available
- Provider services are unreachable
- Dependencies are missing
- Environment setup fails

### Detailed Reporting
- Individual test status and timing
- API key availability status
- Comprehensive error reporting
- Test execution summary

## Example Output

```
============================================================
GraphBit Python Integration Test Suite
============================================================
✓ GraphBit initialized (version: 0.1.0)

API Key Status:
  OPENAI_API_KEY: ✓ Available
  ANTHROPIC_API_KEY: ✗ Not set
  HUGGINGFACE_API_KEY: ✗ Not set

============================================================
INTEGRATION TEST SUMMARY
============================================================
  ✓ PASSED |     5.23s | Embedding Integration Tests
  ✓ PASSED |     3.45s | LLM Integration Tests  
  ✓ PASSED |     2.17s | Static Workflow Integration Tests
  ✓ PASSED |     4.89s | Dynamic Workflow Integration Tests
------------------------------------------------------------
Total: 4/4 modules passed
Total execution time: 15.74 seconds

🎉 All integration tests completed successfully!
```

## Troubleshooting

### Common Issues

1. **GraphBit not found**: Ensure `maturin develop --release` was run
2. **API key errors**: Set required environment variables
3. **Import errors**: Activate the correct conda environment
4. **Build errors**: Run `cargo build --release` first

### Debug Mode
```bash
# Run with verbose output
pytest -v -s

# Run with debug information
pytest --tb=long -v

# Run specific failing test
pytest tests_embeddings.py::TestOpenAIEmbeddings::test_openai_single_embedding -v -s
```

## Contributing

When adding new integration tests:

1. Follow the existing pattern of test classes and fixtures
2. Use appropriate pytest markers (`@pytest.mark.asyncio`, `@pytest.mark.integration`)
3. Handle missing API keys gracefully with `pytest.skip()`
4. Add comprehensive docstrings
5. Update this README with new test categories

## Notes

- Tests focus on **integration** rather than output validation
- Many tests verify API structure and successful execution rather than specific results
- Cross-provider tests validate compatibility and consistency
- Performance tests provide basic metrics without strict assertions 
