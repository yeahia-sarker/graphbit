# Document Loader Guide

## Overview

The GraphBit Document Loader functionality enables workflow steps to automatically load and process various document types (PDF, TXT, Word, JSON, CSV, XML, HTML) as part of their execution. Each workflow node can now include document loading capabilities, allowing for sophisticated document analysis pipelines.

## Features

- **Multi-format Support**: Load PDF, TXT, DOCX, JSON, CSV, XML, and HTML documents
- **Automatic Content Extraction**: Extract text and structured data from documents
- **Metadata Preservation**: Maintain document metadata during processing
- **File Validation**: Validate document types and file sizes
- **Error Handling**: Robust error handling with retry capabilities
- **Workflow Integration**: Seamlessly integrate with existing workflow steps

## Supported Document Types

| Type  | Description                    | Status        |
|-------|--------------------------------|---------------|
| TXT   | Plain text files              | ✅ Implemented |
| JSON  | JavaScript Object Notation    | ✅ Implemented |
| CSV   | Comma-separated values        | ✅ Implemented |
| XML   | Extensible Markup Language    | ✅ Implemented |
| HTML  | HyperText Markup Language     | ✅ Implemented |
| PDF   | Portable Document Format      | 🚧 Planned     |
| DOCX  | Microsoft Word documents      | 🚧 Planned     |

## Basic Usage

### Creating a Document Loader Node

```python
import graphbit

# Create a document loader node
loader_node = graphbit.PyWorkflowNode.document_loader_node(
    name="PDF Document Loader",
    description="Loads and extracts content from PDF documents",
    document_type="txt",  # or "pdf", "json", "csv", etc.
    source_path="/path/to/document.txt"
)
```

### Adding to a Workflow

```python
# Create workflow builder
builder = graphbit.PyWorkflowBuilder("Document Processing Workflow")

# Add document loader node
loader_id = builder.add_node(loader_node)

# Add analysis node that processes the loaded document
analyzer_node = graphbit.PyWorkflowNode.agent_node(
    name="Document Analyzer",
    description="Analyzes the loaded document content",
    agent_id="analyzer",
    prompt="Analyze this document: {document_content}"
)
analyzer_id = builder.add_node(analyzer_node)

# Connect loader to analyzer
builder.connect(loader_id, analyzer_id, graphbit.PyWorkflowEdge.data_flow())

# Build workflow
workflow = builder.build()
```

## Document Processing Pipeline Example

Here's a complete example of a document analysis workflow:

```python
import graphbit
import os

def create_document_analysis_workflow():
    # Initialize GraphBit
    graphbit.init()
    
    # Configure LLM
    llm_config = graphbit.PyLlmConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    # Create workflow builder
    builder = graphbit.PyWorkflowBuilder("Multi-Document Analysis")
    
    # Step 1: Load text document
    text_loader = graphbit.PyWorkflowNode.document_loader_node(
        name="Text Loader",
        description="Loads text document",
        document_type="txt",
        source_path="report.txt"
    )
    
    # Step 2: Analyze text content
    text_analyzer = graphbit.PyWorkflowNode.agent_node(
        name="Text Analyzer",
        description="Analyzes text content",
        agent_id="text_analyzer",
        prompt="Summarize key points from: {content}"
    )
    
    # Step 3: Load JSON data
    json_loader = graphbit.PyWorkflowNode.document_loader_node(
        name="JSON Loader",
        description="Loads JSON data",
        document_type="json",
        source_path="data.json"
    )
    
    # Step 4: Analyze JSON data
    data_analyzer = graphbit.PyWorkflowNode.agent_node(
        name="Data Analyzer",
        description="Analyzes JSON data",
        agent_id="data_analyzer",
        prompt="Extract insights from: {data}"
    )
    
    # Step 5: Synthesis
    synthesizer = graphbit.PyWorkflowNode.agent_node(
        name="Synthesizer",
        description="Combines all analyses",
        agent_id="synthesizer",
        prompt="Create executive summary from: {text_analysis} and {data_analysis}"
    )
    
    # Add nodes
    text_loader_id = builder.add_node(text_loader)
    text_analyzer_id = builder.add_node(text_analyzer)
    json_loader_id = builder.add_node(json_loader)
    data_analyzer_id = builder.add_node(data_analyzer)
    synthesizer_id = builder.add_node(synthesizer)
    
    # Connect workflow
    builder.connect(text_loader_id, text_analyzer_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(json_loader_id, data_analyzer_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(text_analyzer_id, synthesizer_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(data_analyzer_id, synthesizer_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build(), llm_config

# Execute the workflow
workflow, llm_config = create_document_analysis_workflow()
executor = graphbit.PyWorkflowExecutor.new_high_throughput(llm_config)
context = executor.execute(workflow)
```

## Document Content Structure

When a document is loaded, it's converted to a structured format:

```json
{
  "source": "/path/to/document.txt",
  "document_type": "txt",
  "content": "Extracted text content...",
  "metadata": {
    "file_size": 1024,
    "file_path": "/path/to/document.txt"
  },
  "file_size": 1024,
  "extracted_at": "2024-01-01T12:00:00Z"
}
```

## Configuration Options

### Document Loader Configuration

You can configure document loading behavior:

```python
# Configure maximum file size (10MB default)
# Configure encoding (UTF-8 default)
# Configure formatting preservation
```

### Workflow Executor Configuration

For document-heavy workflows, use optimized settings:

```python
# High-throughput configuration for document processing
executor = graphbit.PyWorkflowExecutor.new_high_throughput(llm_config)
executor = executor.with_pool_config(graphbit.PyPoolConfig.memory_optimized())
executor = executor.with_max_node_execution_time(120000)  # 2 minutes
executor = executor.with_fail_fast(False)

# Add retry configuration
retry_config = graphbit.PyRetryConfig.new(3)
retry_config = retry_config.with_exponential_backoff(1000, 2.0, 30000)
executor = executor.with_retry_config(retry_config)
```

## Best Practices

### 1. File Validation
Always validate document paths and types before creating workflows:

```python
import os

def validate_document(path, doc_type):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")
    
    supported_types = ["pdf", "txt", "docx", "json", "csv", "xml", "html"]
    if doc_type.lower() not in supported_types:
        raise ValueError(f"Unsupported type: {doc_type}")
    
    return True
```

### 2. Error Handling
Implement robust error handling for document processing:

```python
try:
    context = executor.execute(workflow)
    if context.is_failed():
        print(f"Workflow failed: {context.state()}")
    else:
        print("Workflow completed successfully")
except Exception as e:
    print(f"Execution error: {e}")
```

### 3. Large Documents
For large documents, consider:
- Splitting documents into chunks
- Using streaming processing
- Implementing progress monitoring
- Setting appropriate timeouts

### 4. Memory Management
Enable memory optimization for document-heavy workflows:

```python
executor = executor.with_pool_config(graphbit.PyPoolConfig.memory_optimized())
```

## Advanced Features

### Custom Document Processing

You can extend document processing by:

1. **Custom Extraction Logic**: Implement custom extractors for specific formats
2. **Pre-processing**: Clean and normalize document content
3. **Metadata Enhancement**: Add custom metadata fields
4. **Content Chunking**: Split large documents into manageable pieces

### Parallel Document Processing

Process multiple documents in parallel:

```python
# Create multiple document loader nodes
documents = ["doc1.txt", "doc2.json", "doc3.csv"]
loader_nodes = []

for i, doc_path in enumerate(documents):
    doc_type = doc_path.split('.')[-1]
    loader = graphbit.PyWorkflowNode.document_loader_node(
        name=f"Loader {i+1}",
        description=f"Loads {doc_path}",
        document_type=doc_type,
        source_path=doc_path
    )
    loader_nodes.append(loader)

# Add all loaders to workflow (they'll run in parallel)
```

## Troubleshooting

### Common Issues

1. **File Not Found**
   ```
   Error: File not found: /path/to/document.txt
   Solution: Check file path and permissions
   ```

2. **Unsupported Document Type**
   ```
   Error: Unsupported document type: xyz
   Solution: Use supported types: pdf, txt, docx, json, csv, xml, html
   ```

3. **File Too Large**
   ```
   Error: File size exceeds maximum allowed size
   Solution: Reduce file size or increase limit in configuration
   ```

4. **Memory Issues**
   ```
   Error: Out of memory processing large document
   Solution: Enable memory optimization or split document
   ```

### Debug Mode

Enable debug logging for document processing:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your workflow code here
```

## Future Enhancements

Planned improvements include:

- **PDF Text Extraction**: Full PDF content extraction with layout preservation
- **DOCX Processing**: Complete Microsoft Word document support  
- **URL Loading**: Load documents directly from URLs
- **Cloud Storage**: Integration with cloud storage providers
- **OCR Support**: Optical character recognition for scanned documents
- **Document Comparison**: Compare multiple documents automatically
- **Version Control**: Track document changes across workflow runs

## API Reference

### PyWorkflowNode.document_loader_node()

```python
@staticmethod
def document_loader_node(
    name: str,
    description: str, 
    document_type: str,
    source_path: str
) -> PyWorkflowNode
```

**Parameters:**
- `name`: Human-readable name for the node
- `description`: Description of what the node does
- `document_type`: Type of document ("txt", "json", "csv", etc.)
- `source_path`: Path to the document file

**Returns:**
- `PyWorkflowNode`: Document loader node ready for workflow integration

**Raises:**
- `ValueError`: If document type is not supported

## Contributing

To contribute to document loader functionality:

1. Add support for new document types in `core/src/document_loader.rs`
2. Update validation in `core/src/graph.rs`
3. Add Python bindings in `python/src/lib.rs`
4. Write tests and update documentation
5. Submit a pull request

For detailed contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). 
