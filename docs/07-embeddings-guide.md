# GraphBit Embeddings Guide

**Version**: Latest  
**Description**: Complete guide to using embeddings in GraphBit for semantic search, similarity analysis, and content understanding

## Overview

GraphBit provides powerful embeddings support for both OpenAI and HuggingFace embedding models. This enables you to:

- **Generate semantic embeddings** for text content
- **Calculate similarity** between documents or messages
- **Build vector databases** for semantic search
- **Enhance workflows** with content understanding
- **Process documents** with semantic analysis

## Quick Start

```python
import graphbit
import asyncio
import os

async def basic_embeddings_example():
    # Initialize GraphBit
    graphbit.init()
    
    # Configure OpenAI embeddings
    config = graphbit.PyEmbeddingConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    
    # Create embedding service
    service = graphbit.PyEmbeddingService(config)
    
    # Generate embeddings for single text
    embedding = await service.embed_text("Hello, world!")
    print(f"Embedding dimensions: {len(embedding)}")
    
    # Generate embeddings for multiple texts
    texts = ["Machine learning", "Artificial intelligence", "Data science"]
    embeddings = await service.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Calculate similarity
    similarity = graphbit.PyEmbeddingService.cosine_similarity(
        embeddings[0], embeddings[1]
    )
    print(f"Similarity between texts: {similarity}")

# Run the example
asyncio.run(basic_embeddings_example())
```

## Provider Configuration

### OpenAI Embeddings

GraphBit supports OpenAI's embedding models including the latest v3 models:

```python
# Basic OpenAI configuration
config = graphbit.PyEmbeddingConfig.openai(
    api_key="your-openai-api-key",
    model="text-embedding-3-small"  # or "text-embedding-3-large"
)

# Advanced configuration with custom settings
config = (graphbit.PyEmbeddingConfig.openai("your-api-key", "text-embedding-3-large")
          .with_timeout(60)  # 60 seconds timeout
          .with_max_batch_size(100)  # Process up to 100 texts at once
          .with_base_url("https://api.openai.com/v1"))  # Custom endpoint
```

**Supported OpenAI Models:**
- `text-embedding-3-small` (1536 dimensions) - Cost-effective, good performance
- `text-embedding-3-large` (3072 dimensions) - Highest performance
- `text-embedding-ada-002` (1536 dimensions) - Legacy model, still supported

### HuggingFace Embeddings

Access thousands of embedding models from HuggingFace:

```python
# Basic HuggingFace configuration
config = graphbit.PyEmbeddingConfig.huggingface(
    api_key="your-huggingface-token",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Advanced configuration
config = (graphbit.PyEmbeddingConfig.huggingface("your-token", "your-model")
          .with_timeout(120)  # Longer timeout for slower models
          .with_max_batch_size(50)
          .with_base_url("https://api-inference.huggingface.co"))
```

**Popular HuggingFace Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` - High quality (768 dimensions)
- `intfloat/e5-large-v2` - State-of-the-art performance (1024 dimensions)
- `BAAI/bge-large-en-v1.5` - Excellent for retrieval tasks (1024 dimensions)

## Core Classes

### PyEmbeddingConfig

Configuration for embedding providers with fluent API design.

#### Static Methods

##### `openai(api_key: str, model: str) -> PyEmbeddingConfig`
Creates OpenAI embedding configuration.

**Parameters:**
- `api_key` (str): Your OpenAI API key
- `model` (str): OpenAI embedding model name

**Example:**
```python
config = graphbit.PyEmbeddingConfig.openai(
    "sk-...",
    "text-embedding-3-small"
)
```

##### `huggingface(api_key: str, model: str) -> PyEmbeddingConfig`
Creates HuggingFace embedding configuration.

**Parameters:**
- `api_key` (str): Your HuggingFace API token
- `model` (str): HuggingFace model identifier

**Example:**
```python
config = graphbit.PyEmbeddingConfig.huggingface(
    "hf_...",
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

#### Instance Methods

##### `with_timeout(timeout_seconds: int) -> PyEmbeddingConfig`
Sets request timeout in seconds.

##### `with_max_batch_size(size: int) -> PyEmbeddingConfig`
Sets maximum batch size for processing multiple texts.

##### `with_base_url(url: str) -> PyEmbeddingConfig`
Sets custom API endpoint URL.

##### `provider_name() -> str`
Returns the provider name ("openai" or "huggingface").

##### `model_name() -> str`
Returns the configured model name.

### PyEmbeddingService

Main service class for generating embeddings.

#### Constructor

##### `PyEmbeddingService(config: PyEmbeddingConfig)`
Creates an embedding service with the given configuration.

#### Async Methods

##### `embed_text(text: str) -> List[float]`
Generates embedding for a single text.

**Parameters:**
- `text` (str): Input text to embed

**Returns:** List of floats representing the embedding vector

**Example:**
```python
embedding = await service.embed_text("Hello world")
print(f"Embedding has {len(embedding)} dimensions")
```

##### `embed_texts(texts: List[str]) -> List[List[float]]`
Generates embeddings for multiple texts efficiently.

**Parameters:**
- `texts` (List[str]): List of input texts

**Returns:** List of embedding vectors

**Example:**
```python
embeddings = await service.embed_texts([
    "First document",
    "Second document",
    "Third document"
])
```

##### `get_dimensions() -> int`
Returns the embedding dimension size for the configured model.

#### Static Methods

##### `cosine_similarity(a: List[float], b: List[float]) -> float`
Calculates cosine similarity between two embedding vectors.

**Parameters:**
- `a` (List[float]): First embedding vector
- `b` (List[float]): Second embedding vector

**Returns:** Similarity score between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)

**Example:**
```python
similarity = graphbit.PyEmbeddingService.cosine_similarity(
    embedding1, embedding2
)
print(f"Similarity: {similarity:.3f}")
```

#### Instance Methods

##### `get_provider_info() -> (str, str)`
Returns tuple of (provider_name, model_name).

## Practical Use Cases

### 1. Document Similarity Analysis

Compare documents to find the most similar ones:

```python
import asyncio
import graphbit

async def document_similarity_pipeline():
    # Setup
    config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-key",
        model="text-embedding-3-small"
    )
    service = graphbit.PyEmbeddingService(config)
    
    # Document corpus
    documents = [
        "Machine learning algorithms for data analysis",
        "Deep learning neural networks and AI",
        "Cooking recipes for Italian cuisine",
        "Python programming best practices",
        "Artificial intelligence and machine learning",
        "Traditional Italian pasta dishes"
    ]
    
    # Generate embeddings
    embeddings = await service.embed_texts(documents)
    
    # Find most similar pairs
    similarities = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                embeddings[i], embeddings[j]
            )
            similarities.append((i, j, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Print top 3 most similar pairs
    print("Most similar document pairs:")
    for i, j, sim in similarities[:3]:
        print(f"Doc {i} & Doc {j}: {sim:.3f}")
        print(f"  '{documents[i]}'")
        print(f"  '{documents[j]}'")
        print()

asyncio.run(document_similarity_pipeline())
```

### 2. Semantic Search Engine

Build a semantic search system to find relevant documents:

```python
import asyncio
import graphbit
from typing import List, Tuple

class SemanticSearchEngine:
    def __init__(self, embedding_service: graphbit.PyEmbeddingService):
        self.service = embedding_service
        self.documents = []
        self.embeddings = []
    
    async def add_documents(self, documents: List[str]):
        """Add documents to the search index"""
        self.documents.extend(documents)
        new_embeddings = await self.service.embed_texts(documents)
        self.embeddings.extend(new_embeddings)
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = await self.service.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                query_embedding, doc_embedding
            )
            similarities.append((self.documents[i], similarity))
        
        # Sort and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

async def semantic_search_example():
    # Setup
    config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-key",
        model="text-embedding-3-small"
    )
    service = graphbit.PyEmbeddingService(config)
    search_engine = SemanticSearchEngine(service)
    
    # Add knowledge base
    knowledge_base = [
        "Python is a programming language known for its simplicity",
        "Machine learning uses algorithms to find patterns in data",
        "Neural networks are inspired by biological brain structures",
        "JavaScript is primarily used for web development",
        "Data science combines statistics and programming",
        "Artificial intelligence aims to create intelligent machines",
        "React is a JavaScript library for building user interfaces",
        "Deep learning is a subset of machine learning"
    ]
    
    await search_engine.add_documents(knowledge_base)
    
    # Search queries
    queries = [
        "programming languages",
        "AI and ML concepts",
        "web development tools"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = await search_engine.search(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. {score:.3f}: {doc}")

asyncio.run(semantic_search_example())
```

### 3. Content Clustering

Group similar content automatically:

```python
import asyncio
import graphbit
from typing import List, Dict
import numpy as np

async def content_clustering_example():
    # Setup
    config = graphbit.PyEmbeddingConfig.huggingface(
        api_key="your-token",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    service = graphbit.PyEmbeddingService(config)
    
    # Content to cluster
    content = [
        # Technology cluster
        "Latest AI breakthroughs in 2024",
        "Machine learning transforms healthcare",
        "Quantum computing advances rapidly",
        
        # Cooking cluster
        "Best Italian pasta recipes",
        "French cuisine cooking techniques",
        "Traditional Mediterranean dishes",
        
        # Finance cluster
        "Stock market trends this year",
        "Cryptocurrency investment strategies",
        "Personal finance planning tips"
    ]
    
    # Generate embeddings
    embeddings = await service.embed_texts(content)
    
    # Simple clustering based on similarity threshold
    clusters: Dict[int, List[int]] = {}
    assigned = set()
    cluster_id = 0
    threshold = 0.3  # Similarity threshold for clustering
    
    for i, embedding_i in enumerate(embeddings):
        if i in assigned:
            continue
        
        # Start new cluster
        clusters[cluster_id] = [i]
        assigned.add(i)
        
        # Find similar items
        for j, embedding_j in enumerate(embeddings):
            if j <= i or j in assigned:
                continue
            
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                embedding_i, embedding_j
            )
            
            if similarity > threshold:
                clusters[cluster_id].append(j)
                assigned.add(j)
        
        cluster_id += 1
    
    # Display clusters
    print("Content Clusters:")
    for cluster_id, indices in clusters.items():
        print(f"\nCluster {cluster_id + 1}:")
        for idx in indices:
            print(f"  - {content[idx]}")

asyncio.run(content_clustering_example())
```

### 4. RAG (Retrieval-Augmented Generation) Pipeline

Combine embeddings with workflow execution for intelligent content generation:

```python
import asyncio
import graphbit
from typing import List, Tuple

class RAGPipeline:
    def __init__(self, embedding_service: graphbit.PyEmbeddingService, 
                 llm_config: graphbit.PyLlmConfig):
        self.embedding_service = embedding_service
        self.llm_config = llm_config
        self.knowledge_base = []
        self.embeddings = []
    
    async def add_knowledge(self, documents: List[str]):
        """Add documents to knowledge base"""
        self.knowledge_base.extend(documents)
        new_embeddings = await self.embedding_service.embed_texts(documents)
        self.embeddings.extend(new_embeddings)
    
    async def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant documents for query"""
        query_embedding = await self.embedding_service.embed_text(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                query_embedding, doc_embedding
            )
            similarities.append((i, similarity))
        
        # Sort and get top documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.knowledge_base[i] for i, _ in similarities[:top_k]]
    
    async def generate_answer(self, question: str) -> str:
        """Generate answer using retrieved documents"""
        # Retrieve relevant context
        relevant_docs = await self.retrieve_relevant_docs(question)
        context = "\n\n".join(relevant_docs)
        
        # Create RAG workflow
        builder = graphbit.PyWorkflowBuilder("RAG Answer Generation")
        
        rag_node = graphbit.PyWorkflowNode.agent_node(
            "RAG Agent",
            "Answers questions using provided context",
            "rag_agent",
            f"""Based on the following context, answer the question accurately.
            
Context:
{context}

Question: {question}

Answer:"""
        )
        
        node_id = builder.add_node(rag_node)
        workflow = builder.build()
        
        # Execute workflow
        executor = graphbit.PyWorkflowExecutor(self.llm_config)
        context = executor.execute(workflow)
        
        # Extract answer (this would depend on your actual response format)
        return "Generated answer based on context"

async def rag_example():
    # Setup embedding service
    embedding_config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-openai-key",
        model="text-embedding-3-small"
    )
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    
    # Setup LLM config
    llm_config = graphbit.PyLlmConfig.openai("your-openai-key", "gpt-4")
    
    # Create RAG pipeline
    rag = RAGPipeline(embedding_service, llm_config)
    
    # Add knowledge base
    knowledge_docs = [
        "GraphBit is a declarative agentic workflow automation framework.",
        "GraphBit supports both OpenAI and HuggingFace embedding models.",
        "The framework enables building complex multi-agent workflows.",
        "GraphBit provides performance optimization through memory pools.",
        "Workflows in GraphBit can be executed synchronously or asynchronously."
    ]
    
    await rag.add_knowledge(knowledge_docs)
    
    # Ask questions
    questions = [
        "What embedding providers does GraphBit support?",
        "How does GraphBit optimize performance?",
        "What types of workflows can GraphBit handle?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        relevant_docs = await rag.retrieve_relevant_docs(question)
        print("Relevant documents:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"  {i}. {doc}")

asyncio.run(rag_example())
```

## Performance Considerations

### Batch Processing

Process multiple texts efficiently:

```python
async def efficient_batch_processing():
    config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-key",
        model="text-embedding-3-small"
    ).with_max_batch_size(100)  # Process up to 100 texts at once
    
    service = graphbit.PyEmbeddingService(config)
    
    # Large dataset
    large_dataset = [f"Document number {i}" for i in range(1000)]
    
    # Process in chunks for memory efficiency
    chunk_size = 50
    all_embeddings = []
    
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset[i:i + chunk_size]
        chunk_embeddings = await service.embed_texts(chunk)
        all_embeddings.extend(chunk_embeddings)
        print(f"Processed {i + len(chunk)}/{len(large_dataset)} documents")
    
    print(f"Generated {len(all_embeddings)} embeddings total")
```

### Caching Embeddings

Save computational costs by caching embeddings:

```python
import json
import os
from typing import Dict, List

class EmbeddingCache:
    def __init__(self, cache_file: str = "embeddings_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, List[float]] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    async def get_embedding(self, text: str, service: graphbit.PyEmbeddingService) -> List[float]:
        if text in self.cache:
            return self.cache[text]
        
        embedding = await service.embed_text(text)
        self.cache[text] = embedding
        return embedding
    
    async def get_embeddings(self, texts: List[str], service: graphbit.PyEmbeddingService) -> List[List[float]]:
        # Check cache for existing embeddings
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await service.embed_texts(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                self.cache[texts[idx]] = embedding
        
        return embeddings

# Usage example
async def cached_embedding_example():
    config = graphbit.PyEmbeddingConfig.openai("your-key", "text-embedding-3-small")
    service = graphbit.PyEmbeddingService(config)
    cache = EmbeddingCache()
    
    texts = ["Hello world", "Machine learning", "Data science"]
    
    # First call - generates embeddings
    embeddings1 = await cache.get_embeddings(texts, service)
    print("Generated embeddings for first time")
    
    # Second call - uses cache
    embeddings2 = await cache.get_embeddings(texts, service)
    print("Used cached embeddings")
    
    # Save cache for future runs
    cache.save_cache()
```

## Error Handling

Handle common errors gracefully:

```python
import asyncio
import graphbit

async def robust_embedding_example():
    config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-key",
        model="text-embedding-3-small"
    ).with_timeout(30)
    
    try:
        service = graphbit.PyEmbeddingService(config)
        
        # Test with potentially problematic input
        texts = [
            "Normal text",
            "",  # Empty string
            "Very long text " * 1000,  # Very long text
            "Special characters: !@#$%^&*()"
        ]
        
        for i, text in enumerate(texts):
            try:
                embedding = await service.embed_text(text)
                print(f"Text {i}: Success ({len(embedding)} dimensions)")
            except Exception as e:
                print(f"Text {i}: Failed - {e}")
        
    except Exception as e:
        print(f"Service creation failed: {e}")

asyncio.run(robust_embedding_example())
```

## Integration with Workflows

Combine embeddings with GraphBit workflows for powerful content processing:

```python
import asyncio
import graphbit

async def embedding_workflow_integration():
    # Setup services
    embedding_config = graphbit.PyEmbeddingConfig.openai("your-key", "text-embedding-3-small")
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    llm_config = graphbit.PyLlmConfig.openai("your-key", "gpt-4")
    
    # Content to process
    articles = [
        "Introduction to machine learning and its applications",
        "The future of artificial intelligence in healthcare",
        "Climate change impact on global food security"
    ]
    
    # Generate embeddings for content analysis
    embeddings = await embedding_service.embed_texts(articles)
    
    # Find most similar pair for further processing
    max_similarity = -1
    best_pair = (0, 1)
    
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                embeddings[i], embeddings[j]
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (i, j)
    
    # Create workflow to analyze the most similar articles
    builder = graphbit.PyWorkflowBuilder("Similar Content Analysis")
    
    analyzer_node = graphbit.PyWorkflowNode.agent_node(
        "Content Analyzer",
        "Analyzes similar content",
        "analyzer",
        f"""Analyze these two similar articles and explain their relationship:

Article 1: {articles[best_pair[0]]}

Article 2: {articles[best_pair[1]]}

Similarity Score: {max_similarity:.3f}

Analysis:"""
    )
    
    builder.add_node(analyzer_node)
    workflow = builder.build()
    
    # Execute analysis workflow
    executor = graphbit.PyWorkflowExecutor(llm_config)
    context = executor.execute(workflow)
    
    print(f"Analysis completed: {context.is_completed()}")

asyncio.run(embedding_workflow_integration())
```

## Best Practices

### 1. Choose the Right Model

- **OpenAI text-embedding-3-small**: Best for cost-effective general use
- **OpenAI text-embedding-3-large**: Best for highest accuracy requirements
- **HuggingFace all-MiniLM-L6-v2**: Fast and efficient for basic similarity tasks
- **HuggingFace all-mpnet-base-v2**: Good balance of speed and quality

### 2. Optimize Batch Processing

```python
# Good: Process texts in batches
texts = ["text1", "text2", "text3", ...]
embeddings = await service.embed_texts(texts)

# Avoid: Processing texts one by one
embeddings = []
for text in texts:
    embedding = await service.embed_text(text)
    embeddings.append(embedding)
```

### 3. Handle Rate Limits

```python
import asyncio

async def rate_limited_processing(texts, service, delay=1.0):
    embeddings = []
    for i in range(0, len(texts), 50):  # Process 50 at a time
        batch = texts[i:i+50]
        batch_embeddings = await service.embed_texts(batch)
        embeddings.extend(batch_embeddings)
        
        if i + 50 < len(texts):  # Don't delay after last batch
            await asyncio.sleep(delay)
    
    return embeddings
```

### 4. Normalize Embeddings for Consistency

```python
import math

def normalize_embedding(embedding):
    """Normalize embedding to unit length"""
    magnitude = math.sqrt(sum(x * x for x in embedding))
    return [x / magnitude for x in embedding] if magnitude > 0 else embedding

# Use normalized embeddings for consistent similarity calculations
embedding = await service.embed_text("Hello world")
normalized_embedding = normalize_embedding(embedding)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```python
   # Make sure your API key is valid and has proper permissions
   config = graphbit.PyEmbeddingConfig.openai("sk-...", "text-embedding-3-small")
   ```

2. **Timeout Issues**
   ```python
   # Increase timeout for slower models or large batches
   config = config.with_timeout(120)  # 2 minutes
   ```

3. **Rate Limit Exceeded**
   ```python
   # Reduce batch size and add delays
   config = config.with_max_batch_size(20)
   await asyncio.sleep(1.0)  # Add delay between requests
   ```

4. **Memory Issues with Large Datasets**
   ```python
   # Process in smaller chunks
   chunk_size = 100
   for i in range(0, len(large_dataset), chunk_size):
       chunk = large_dataset[i:i + chunk_size]
       embeddings = await service.embed_texts(chunk)
       # Process embeddings immediately rather than storing all
   ```

## Next Steps

- Learn about [Workflow Integration](01-getting-started-workflows.md)
- Explore [Document Processing](04-document-processing-guide.md)
- Check out [Complete API Reference](05-complete-api-reference.md)
- See [Advanced Use Cases](02-use-case-examples.md) 
