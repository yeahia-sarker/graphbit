---
title: GraphBit Python API Summary
description: Complete overview of all Python classes and methods available in GraphBit
---

# GraphBit Python API Summary

Complete reference of all classes, methods, and functions available in the GraphBit Python library.

## Quick Navigation

- [Module Functions](#module-functions)
- [Core Workflow Classes](#core-workflow-classes)
- [LLM Configuration](#llm-configuration)
- [Embedding Classes](#embedding-classes)
- [Dynamic Workflow Classes](#dynamic-workflow-classes)
- [Configuration Classes](#configuration-classes)
- [Node and Edge Types](#node-and-edge-types)

## Module Functions

```python
import graphbit

# Library initialization
graphbit.init()                    # Initialize the library (required)
graphbit.version()                 # Get library version string
```

## Core Workflow Classes

### PyWorkflowBuilder

```python
# Constructor
builder = graphbit.PyWorkflowBuilder("Workflow Name")

# Configuration
builder.description(description)   # Set workflow description

# Node management
node_id = builder.add_node(node)   # Add node, returns node ID
builder.connect(from_id, to_id, edge)  # Connect nodes with edge

# Build
workflow = builder.build()         # Create final workflow
```

### PyWorkflow

```python
# Constructor
workflow = graphbit.PyWorkflow("Name", "Description")

# Direct node management
node_id = workflow.add_node(node)  # Add node directly
workflow.connect_nodes(from_id, to_id, edge)  # Connect nodes

# Properties
workflow.id()                      # Get workflow ID
workflow.name()                    # Get workflow name
workflow.description()             # Get description
workflow.node_count()              # Number of nodes
workflow.edge_count()              # Number of edges

# Validation and serialization
workflow.validate()                # Validate workflow (raises on error)
json_str = workflow.to_json()      # Serialize to JSON
workflow = graphbit.PyWorkflow.from_json(json_str)  # Deserialize
```

### PyWorkflowExecutor

```python
# Constructors
executor = graphbit.PyWorkflowExecutor(llm_config)
executor = graphbit.PyWorkflowExecutor.new_high_throughput(llm_config)
executor = graphbit.PyWorkflowExecutor.new_low_latency(llm_config)
executor = graphbit.PyWorkflowExecutor.new_memory_optimized(llm_config)

# Configuration
executor = executor.with_max_node_execution_time(timeout_ms)
executor = executor.with_fail_fast(enabled)
executor = executor.with_retry_config(retry_config)
executor = executor.with_circuit_breaker_config(circuit_config)
executor = executor.without_retries()

# Execution
context = executor.execute(workflow)                    # Sync execution
context = await executor.execute_async(workflow)        # Async execution
contexts = executor.execute_concurrent(workflows)       # Concurrent workflows
contexts = executor.execute_batch(workflows)            # Batch workflows
results = executor.execute_concurrent_agent_tasks(prompts, agent_id)  # Concurrent prompts
```

### PyWorkflowContext

```python
# Status
context.workflow_id()              # Get workflow ID
context.state()                    # Get current state
context.is_completed()             # Check if completed
context.is_failed()                # Check if failed
context.execution_time_ms()        # Execution time

# Variables
value = context.get_variable(key)  # Get variable value
variables = context.variables()    # Get all variables as [(key, value)]
```

### PyValidationResult

```python
result.is_valid()                  # Check if validation passed
errors = result.errors()           # Get list of error messages
```

## LLM Configuration

### PyLlmConfig

```python
# Provider configurations
config = graphbit.PyLlmConfig.openai(api_key, model)
config = graphbit.PyLlmConfig.anthropic(api_key, model)
config = graphbit.PyLlmConfig.huggingface(api_key, model)

# Properties
config.provider_name()             # Get provider name
config.model_name()                # Get model name
```

### PyAgentCapability

```python
# Built-in capabilities
capability = graphbit.PyAgentCapability.text_processing()
capability = graphbit.PyAgentCapability.data_analysis()
capability = graphbit.PyAgentCapability.tool_execution()
capability = graphbit.PyAgentCapability.decision_making()
capability = graphbit.PyAgentCapability.custom("custom_name")

# String representation
str(capability)                    # Get string representation
```

## Embedding Classes

### PyEmbeddingConfig

```python
# Provider configurations
config = graphbit.PyEmbeddingConfig.openai(api_key, model)
config = graphbit.PyEmbeddingConfig.huggingface(api_key, model)

# Configuration
config = config.with_base_url(url)
config = config.with_timeout(seconds)
config = config.with_max_batch_size(size)

# Properties
config.provider_name()             # Get provider name
config.model_name()                # Get model name
str(config)                        # String representation
```

### PyEmbeddingService

```python
# Constructor
service = graphbit.PyEmbeddingService(config)

# Embedding generation
embedding = await service.embed_text(text)              # Single text
embeddings = await service.embed_texts(texts)           # Multiple texts

# Utilities
similarity = graphbit.PyEmbeddingService.cosine_similarity(vec1, vec2)
dimensions = await service.get_dimensions()
provider, model = service.get_provider_info()
str(service)                       # String representation
```

### PyEmbeddingRequest

```python
# Constructors
request = graphbit.PyEmbeddingRequest(text)
request = graphbit.PyEmbeddingRequest.multiple(texts)

# Configuration
request = request.with_user(user_id)

# Properties
request.input_count()              # Number of inputs
str(request)                       # String representation
```

### PyEmbeddingResponse

```python
# Results
embeddings = response.embeddings()      # List of embedding vectors
model = response.model()                # Model used
dimensions = response.dimensions()      # Embedding dimensions
count = response.embedding_count()      # Number of embeddings

# Token usage
prompt_tokens = response.prompt_tokens()
total_tokens = response.total_tokens()

str(response)                      # String representation
```

## Dynamic Workflow Classes

### PyDynamicGraphConfig

```python
# Constructor
config = graphbit.PyDynamicGraphConfig()

# Configuration
config = config.with_max_auto_nodes(max_nodes)
config = config.with_confidence_threshold(threshold)
config = config.with_generation_temperature(temperature)
config = config.with_max_generation_depth(depth)
config = config.with_completion_objectives(objectives)
config = config.enable_validation()
config = config.disable_validation()

# Properties
config.max_auto_nodes()            # Get max auto nodes
config.confidence_threshold()      # Get confidence threshold
config.generation_temperature()    # Get generation temperature
str(config)                        # String representation
```

### PyDynamicGraphManager

```python
# Constructor and initialization
manager = graphbit.PyDynamicGraphManager(llm_config, dynamic_config)
manager.initialize(llm_config, dynamic_config)

# Node generation
response = manager.generate_node(workflow_id, context, objective, allowed_types)
result = manager.auto_complete_workflow(workflow, context, objectives)

# Analytics
analytics = manager.get_analytics()
```

### PyDynamicNodeResponse

```python
# Generated content
node = response.get_node()              # Get generated node
confidence = response.confidence()      # Confidence score
reasoning = response.reasoning()        # Generation reasoning
completes = response.completes_workflow()  # If workflow is complete

# Connections
connections = response.get_suggested_connections()  # List of suggestions
str(response)                          # String representation
```

### PySuggestedConnection

```python
# Connection details
from_id = connection.source_node_id()     # Source node ID
to_id = connection.to_node()              # Target node ID
edge_type = connection.edge_type()        # Edge type
confidence = connection.confidence()      # Confidence score
reasoning = connection.reasoning()        # Connection reasoning
str(connection)                           # String representation
```

### PyWorkflowAutoCompletion

```python
# Constructor and initialization
engine = graphbit.PyWorkflowAutoCompletion(manager)
engine.initialize(manager)

# Configuration
engine = engine.with_max_iterations(max_iterations)
engine = engine.with_timeout_ms(timeout_ms)

# Auto-completion
result = engine.complete_workflow(workflow, context, objectives)
```

### PyAutoCompletionResult

```python
# Results
nodes = result.get_generated_nodes()     # List of generated node IDs
time_ms = result.completion_time_ms()    # Completion time
success = result.success()               # Success status
count = result.node_count()              # Number of nodes generated

# Analytics
analytics = result.get_analytics()
str(result)                              # String representation
```

### PyDynamicGraphAnalytics

```python
# Generation statistics
total = analytics.total_nodes_generated()      # Total nodes generated
successful = analytics.successful_generations() # Successful generations
failed = analytics.failed_generations()        # Failed generations
success_rate = analytics.success_rate()        # Success rate

# Performance metrics
avg_confidence = analytics.avg_confidence()    # Average confidence
cache_rate = analytics.cache_hit_rate()        # Cache hit rate
avg_time = analytics.avg_generation_time()     # Average generation time
times = analytics.get_generation_times()       # List of generation times
str(analytics)                                 # String representation
```

## Configuration Classes

### PyRetryConfig

```python
# Constructors
config = graphbit.PyRetryConfig(max_attempts)
config = graphbit.PyRetryConfig.default()

# Configuration
config = config.with_exponential_backoff(initial_ms, multiplier, max_ms)
config = config.with_jitter(jitter_factor)

# Properties
config.max_attempts()              # Get max attempts
config.initial_delay_ms()          # Get initial delay
```

### PyCircuitBreakerConfig

```python
# Constructors
config = graphbit.PyCircuitBreakerConfig(failure_threshold, recovery_timeout_ms)
config = graphbit.PyCircuitBreakerConfig.default()

# Properties
config.failure_threshold()         # Get failure threshold
config.recovery_timeout_ms()       # Get recovery timeout
```

## Node and Edge Types

### PyWorkflowNode

```python
# Agent nodes
node = graphbit.PyWorkflowNode.agent_node(name, description, agent_id, prompt)

# Condition nodes
node = graphbit.PyWorkflowNode.condition_node(name, description, expression)

# Transform nodes
node = graphbit.PyWorkflowNode.transform_node(name, description, transformation)

# Delay nodes
node = graphbit.PyWorkflowNode.delay_node(name, description, duration_seconds)

# Document loader nodes
node = graphbit.PyWorkflowNode.document_loader_node(name, description, doc_type, path)

# Properties
node.id()                          # Get node ID
node.name()                        # Get node name
node.description()                 # Get node description
```

### PyWorkflowEdge

```python
# Edge types
edge = graphbit.PyWorkflowEdge.data_flow()       # Data flow edge
edge = graphbit.PyWorkflowEdge.control_flow()    # Control flow edge
edge = graphbit.PyWorkflowEdge.conditional(condition)  # Conditional edge
```

## Common Patterns

### Basic Workflow Creation

```python
import graphbit
import os

# Initialize
graphbit.init()

# Configure LLM
config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")

# Build workflow
builder = graphbit.PyWorkflowBuilder("My Workflow")
node = graphbit.PyWorkflowNode.agent_node("Agent", "Description", "agent_id", "Prompt")
node_id = builder.add_node(node)
workflow = builder.build()

# Execute
executor = graphbit.PyWorkflowExecutor(config)
context = executor.execute(workflow)
```

### Error Handling

```python
try:
    workflow.validate()
    context = executor.execute(workflow)
    if context.is_completed():
        result = context.get_variable("output")
    else:
        print("Workflow failed")
except Exception as e:
    print(f"Error: {e}")
```

### Dynamic Workflow Generation

```python
# Setup dynamic manager
dynamic_config = (graphbit.PyDynamicGraphConfig()
    .with_max_auto_nodes(5)
    .with_confidence_threshold(0.8))

manager = graphbit.PyDynamicGraphManager(llm_config, dynamic_config)
manager.initialize(llm_config, dynamic_config)

# Generate nodes
response = manager.generate_node(workflow_id, context, "objective")
if response.confidence() > 0.8:
    new_node = response.get_node()
    workflow.add_node(new_node)
```

### Concurrent Execution

```python
# Multiple workflows
contexts = executor.execute_concurrent([workflow1, workflow2, workflow3])

# Multiple prompts with same agent
prompts = ["Task 1", "Task 2", "Task 3"]
results = executor.execute_concurrent_agent_tasks(prompts, "agent_id")
```

### Embeddings

```python
# Setup embedding service
embed_config = graphbit.PyEmbeddingConfig.openai(api_key, "text-embedding-3-small")
service = graphbit.PyEmbeddingService(embed_config)

# Generate embeddings
embedding = await service.embed_text("Hello world")
embeddings = await service.embed_texts(["Text 1", "Text 2"])

# Calculate similarity
similarity = graphbit.PyEmbeddingService.cosine_similarity(embeddings[0], embeddings[1])
```

---

This reference covers all available classes and methods in the GraphBit Python library. For detailed examples and usage patterns, see the complete API documentation. 