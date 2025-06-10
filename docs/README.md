# GraphBit Documentation

Welcome to GraphBit - a declarative agentic workflow automation framework that lets you build powerful AI-driven workflows with ease. This documentation is organized by topics and use cases, with practical Python examples to get you started quickly.

## 🚀 Getting Started

### Quick Setup
```python
import graphbit
import os

# Initialize and create your first workflow
graphbit.init()
config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")

builder = graphbit.PyWorkflowBuilder("Hello World")
node = graphbit.PyWorkflowNode.agent_node(
    "Greeter", "Generates greetings", "greeter",
    "Say hello to {name}"
)

workflow = builder.add_node(node).build()
executor = graphbit.PyWorkflowExecutor(config)
context = executor.execute(workflow)
```

**Learn more**: [Installation Guide](00-installation-setup.md) | [Quick Start Tutorial](01-getting-started-workflows.md)

---

## 🧠 Embeddings & Semantic Analysis

GraphBit provides powerful embeddings support for semantic search, similarity analysis, and content understanding.

### Quick Embeddings Example
```python
import asyncio
import graphbit

async def embeddings_example():
    # Configure OpenAI embeddings
    config = graphbit.PyEmbeddingConfig.openai(
        api_key="your-openai-key",
        model="text-embedding-3-small"
    )
    
    # Create embedding service
    service = graphbit.PyEmbeddingService(config)
    
    # Generate embeddings
    texts = ["Machine learning", "Artificial intelligence", "Data science"]
    embeddings = await service.embed_texts(texts)
    
    # Calculate similarity
    similarity = graphbit.PyEmbeddingService.cosine_similarity(
        embeddings[0], embeddings[1]
    )
    print(f"Similarity: {similarity:.3f}")

asyncio.run(embeddings_example())
```

### Semantic Search Engine
Build intelligent search systems that understand meaning:

```python
class SemanticSearch:
    def __init__(self, embedding_service):
        self.service = embedding_service
        self.documents = []
        self.embeddings = []
    
    async def add_documents(self, documents):
        self.documents.extend(documents)
        new_embeddings = await self.service.embed_texts(documents)
        self.embeddings.extend(new_embeddings)
    
    async def search(self, query, top_k=5):
        query_embedding = await self.service.embed_text(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                query_embedding, doc_embedding
            )
            similarities.append((self.documents[i], similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

### RAG (Retrieval-Augmented Generation)
Combine embeddings with workflows for intelligent content generation:

```python
async def rag_pipeline(question, knowledge_base):
    # Setup embedding service
    embedding_config = graphbit.PyEmbeddingConfig.openai("your-key", "text-embedding-3-small")
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    
    # Create search engine
    search = SemanticSearch(embedding_service)
    await search.add_documents(knowledge_base)
    
    # Find relevant context
    relevant_docs = await search.search(question, top_k=3)
    context = "\n\n".join([doc for doc, _ in relevant_docs])
    
    # Create RAG workflow
    llm_config = graphbit.PyLlmConfig.openai("your-key", "gpt-4")
    builder = graphbit.PyWorkflowBuilder("RAG Pipeline")
    
    rag_node = graphbit.PyWorkflowNode.agent_node(
        "RAG Agent",
        "Answers using context",
        "rag_agent",
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    builder.add_node(rag_node)
    workflow = builder.build()
    
    # Execute and get answer
    executor = graphbit.PyWorkflowExecutor(llm_config)
    result = executor.execute(workflow)
    return result
```

**Learn more**: [Embeddings Guide](07-embeddings-guide.md) | [API Reference](05-complete-api-reference.md#embeddings-classes)

---

## 🏗️ Static vs Dynamic Workflows

GraphBit supports both static workflows (predefined structure) and dynamic workflows (generated at runtime).

### Static Workflows
Perfect for predictable, repeatable processes:

```python
def create_blog_workflow():
    """Static workflow - structure defined at design time"""
    builder = graphbit.PyWorkflowBuilder("Static Blog Generator")
    
    # Predefined pipeline: Research → Write → Edit
    researcher = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches topics", "research",
        "Research key points about: {topic}"
    ))
    
    writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Writer", "Writes content", "write", 
        "Write a blog post using research: {research}"
    ))
    
    editor = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Editor", "Edits content", "edit",
        "Edit and improve: {content}"
    ))
    
    builder.connect(researcher, writer, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(writer, editor, graphbit.PyWorkflowEdge.data_flow())
    return builder.build()
```

### Dynamic Workflows
Generated programmatically based on runtime conditions:

```python
def create_dynamic_content_workflow(content_type, complexity_level, target_audience):
    """Dynamic workflow - structure determined at runtime"""
    builder = graphbit.PyWorkflowBuilder(f"Dynamic-{content_type}-{complexity_level}")
    
    # Base research node (always included)
    researcher = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches content", "research",
        f"Research {content_type} topics for {target_audience}: {{topic}}"
    ))
    
    previous_node = researcher
    
    # Add nodes based on complexity
    if complexity_level >= 2:
        outliner = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Outliner", "Creates outline", "outline",
            f"Create detailed outline for {content_type}: {{research}}"
        ))
        builder.connect(previous_node, outliner, graphbit.PyWorkflowEdge.data_flow())
        previous_node = outliner
    
    # Content creation varies by type
    if content_type == "blog":
        writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Blog Writer", "Writes blog posts", "blog_writer",
            "Write comprehensive blog post: {research}"
        ))
    elif content_type == "social":
        writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Social Writer", "Writes social content", "social_writer",
            "Create engaging social media content: {research}"
        ))
    else:
        writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "General Writer", "Writes content", "writer",
            "Write content based on: {research}"
        ))
    
    builder.connect(previous_node, writer, graphbit.PyWorkflowEdge.data_flow())
    previous_node = writer
    
    # Add quality control for complex content
    if complexity_level >= 3:
        reviewer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Quality Reviewer", "Reviews content quality", "reviewer",
            f"Review and improve {content_type} for {target_audience}: {{content}}"
        ))
        builder.connect(previous_node, reviewer, graphbit.PyWorkflowEdge.data_flow())
        previous_node = reviewer
    
    return builder.build()

# Usage examples
blog_workflow = create_dynamic_content_workflow("blog", 3, "developers")
social_workflow = create_dynamic_content_workflow("social", 2, "general audience")
```

---

## 📝 Content Creation & Writing

Build AI workflows for automated content generation, from blog posts to social media campaigns.

### Static Blog Post Generation Pipeline
```python
def create_blog_workflow():
    builder = graphbit.PyWorkflowBuilder("Blog Generator")
    
    # Research → Outline → Write → Optimize
    researcher = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches topics", "research",
        "Research key points about: {topic}"
    ))
    
    writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Writer", "Writes content", "write", 
        "Write a blog post using research: {research}"
    ))
    
    builder.connect(researcher, writer, graphbit.PyWorkflowEdge.data_flow())
    return builder.build()
```

### Dynamic Social Media Campaigns
Create platform-specific content that adapts to campaign requirements:

```python
def create_adaptive_social_campaign(platforms, campaign_style, budget_level):
    """Dynamic social media workflow based on requirements"""
    builder = graphbit.PyWorkflowBuilder(f"Social-{campaign_style}-{budget_level}")
    
    # Strategy varies by budget and style
    if budget_level == "premium":
        strategist = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Premium Strategist", "Creates premium strategy", "premium_strategy",
            f"Create {campaign_style} premium social strategy for: {{product}}"
        ))
    else:
        strategist = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Standard Strategist", "Creates standard strategy", "standard_strategy",
            f"Create {campaign_style} cost-effective strategy for: {{product}}"
        ))
    
    content_nodes = []
    
    # Add platform-specific writers based on requirements
    for platform in platforms:
        if platform == "twitter":
            writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "Twitter Writer", "Writes Twitter content", f"twitter_{campaign_style}",
                f"Create {campaign_style} Twitter content: {{strategy}}"
            ))
        elif platform == "linkedin":
            writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "LinkedIn Writer", "Writes LinkedIn content", f"linkedin_{campaign_style}",
                f"Create professional {campaign_style} LinkedIn content: {{strategy}}"
            ))
        elif platform == "instagram":
            writer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "Instagram Writer", "Writes Instagram content", f"instagram_{campaign_style}",
                f"Create visual {campaign_style} Instagram content: {{strategy}}"
            ))
        
        builder.connect(strategist, writer, graphbit.PyWorkflowEdge.data_flow())
        content_nodes.append(writer)
    
    # Add review step for premium campaigns
    if budget_level == "premium":
        reviewer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Campaign Reviewer", "Reviews campaign content", "premium_review",
            "Ensure premium quality across all content: {all_content}"
        ))
        
        for content_node in content_nodes:
            builder.connect(content_node, reviewer, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Dynamic usage
campaign = create_adaptive_social_campaign(
    platforms=["twitter", "linkedin"], 
    campaign_style="professional", 
    budget_level="premium"
)
```

**Examples**: [Content Creation Workflows](02-use-case-examples.md#content-creation-examples)

---

## 📊 Data Analysis & Processing

Transform raw data into insights with AI-powered analysis workflows.

### Static Customer Feedback Analysis
```python
def create_feedback_analyzer():
    builder = graphbit.PyWorkflowBuilder("Feedback Analyzer")
    
    # Parallel analysis: sentiment + categorization + priority
    sentiment = builder.add_node(create_sentiment_node())
    category = builder.add_node(create_category_node()) 
    priority = builder.add_node(create_priority_node())
    
    # Merge results for final report
    reporter = builder.add_node(create_report_node())
    
    # Fan-out then fan-in pattern
    for analyzer in [sentiment, category, priority]:
        builder.connect(analyzer, reporter)
    
    return builder.build()
```

### Dynamic Data Processing Pipeline
Adapt processing based on data characteristics:

```python
def create_dynamic_data_processor(data_source, data_size, processing_requirements):
    """Dynamic data processing workflow"""
    builder = graphbit.PyWorkflowBuilder(f"DataProcessor-{data_source}-{data_size}")
    
    # Data validation (always needed)
    validator = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Data Validator", "Validates data quality", "validator",
        f"Validate {data_source} data quality and structure: {{raw_data}}"
    ))
    
    previous_node = validator
    
    # Processing pipeline varies by data size
    if data_size == "large":
        # Add chunking for large datasets
        chunker = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Data Chunker", "Chunks large data", "chunker",
            "Split large dataset into manageable chunks: {validated_data}"
        ))
        builder.connect(previous_node, chunker, graphbit.PyWorkflowEdge.data_flow())
        previous_node = chunker
    
    # Add processing steps based on requirements
    for requirement in processing_requirements:
        if requirement == "sentiment_analysis":
            processor = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "Sentiment Processor", "Analyzes sentiment", "sentiment",
                "Perform sentiment analysis on: {data_chunk}"
            ))
        elif requirement == "entity_extraction":
            processor = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "Entity Extractor", "Extracts entities", "entities",
                "Extract named entities from: {data_chunk}"
            ))
        elif requirement == "summarization":
            processor = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                "Summarizer", "Summarizes content", "summary",
                "Summarize key insights from: {data_chunk}"
            ))
        
        builder.connect(previous_node, processor, graphbit.PyWorkflowEdge.data_flow())
        previous_node = processor
    
    # Final aggregation for large datasets
    if data_size == "large":
        aggregator = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Results Aggregator", "Aggregates results", "aggregator",
            "Combine and summarize all processing results: {processed_chunks}"
        ))
        builder.connect(previous_node, aggregator, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Dynamic usage
workflow = create_dynamic_data_processor(
    data_source="customer_feedback",
    data_size="large", 
    processing_requirements=["sentiment_analysis", "entity_extraction", "summarization"]
)
```

**Examples**: [Data Analysis Workflows](02-use-case-examples.md#data-analysis-examples)

---

## 🏢 Business Process Automation

Automate complex business workflows with conditional logic and intelligent routing.

### Static Customer Support Automation
```python
def create_support_workflow():
    builder = graphbit.PyWorkflowBuilder("Support Automation")
    
    # Classify → Route → Respond
    classifier = create_classifier_node()
    urgent_handler = create_urgent_handler_node()
    standard_handler = create_standard_handler_node()
    
    # Conditional routing based on urgency
    builder.connect(classifier, urgent_handler, condition="urgency == 'high'")
    builder.connect(classifier, standard_handler, condition="urgency != 'high'")
    
    return builder.build()
```

### Dynamic Business Process Engine
Generate workflows based on business rules:

```python
def create_dynamic_approval_workflow(request_type, amount, department, approval_policy):
    """Dynamic approval workflow based on business rules"""
    builder = graphbit.PyWorkflowBuilder(f"Approval-{request_type}-{department}")
    
    # Initial request processor
    processor = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Request Processor", "Processes requests", "processor",
        f"Process {request_type} request for {department}: {{request_details}}"
    ))
    
    previous_node = processor
    
    # Determine approval chain based on amount and policy
    approval_chain = approval_policy.get_approval_chain(amount, department, request_type)
    
    for i, approver_level in enumerate(approval_chain):
        approver = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            f"Approver Level {i+1}", f"Reviews {approver_level} approval", f"approve_{i}",
            f"Review {request_type} request as {approver_level}: {{request_summary}}"
        ))
        
        # Add conditional logic based on approval outcome
        if i > 0:  # Not the first approver
            condition_node = builder.add_node(graphbit.PyWorkflowNode.condition_node(
                f"Approval Gate {i}", "Checks previous approval", 
                f"approval_status_{i-1} == 'approved'"
            ))
            builder.connect(previous_node, condition_node, graphbit.PyWorkflowEdge.data_flow())
            builder.connect(condition_node, approver, graphbit.PyWorkflowEdge.conditional("passed"))
        else:
            builder.connect(previous_node, approver, graphbit.PyWorkflowEdge.data_flow())
        
        previous_node = approver
    
    # Final processor
    finalizer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Request Finalizer", "Finalizes approved requests", "finalizer",
        f"Finalize approved {request_type} request: {{final_approval}}"
    ))
    builder.connect(previous_node, finalizer, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Usage with business rules
class ApprovalPolicy:
    def get_approval_chain(self, amount, department, request_type):
        if amount > 10000:
            return ["manager", "director", "vp"]
        elif amount > 1000:
            return ["manager", "director"] 
        else:
            return ["manager"]

policy = ApprovalPolicy()
workflow = create_dynamic_approval_workflow("expense", 5000, "engineering", policy)
```

**Examples**: [Business Automation](02-use-case-examples.md#business-automation)

---

## 🧠 Multi-Agent Collaboration

Coordinate multiple AI agents to solve complex problems through collaboration.

### Static Research & Writing Team
```python
# Specialist agents working together
researcher = create_specialist_agent("research", "domain expert")
fact_checker = create_specialist_agent("fact_check", "accuracy expert") 
writer = create_specialist_agent("write", "content expert")
editor = create_specialist_agent("edit", "editorial expert")

# Sequential collaboration with feedback loops
builder.connect(researcher, writer)
builder.connect(writer, fact_checker)
builder.connect(fact_checker, editor)
```

### Dynamic Expert Assembly
Create specialized teams based on problem complexity:

```python
def create_dynamic_expert_team(problem_domain, complexity, available_experts):
    """Dynamically assemble expert team based on problem requirements"""
    builder = graphbit.PyWorkflowBuilder(f"Expert-Team-{problem_domain}")
    
    # Problem analyzer determines required expertise
    analyzer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
        "Problem Analyzer", "Analyzes problem requirements", "analyzer",
        f"Analyze {problem_domain} problem and determine required expertise: {{problem_description}}"
    ))
    
    # Select experts based on analysis
    expert_nodes = []
    expert_selector = ExpertSelector(available_experts)
    
    if complexity == "high":
        # Multi-stage expert collaboration
        primary_experts = expert_selector.get_primary_experts(problem_domain)
        secondary_experts = expert_selector.get_secondary_experts(problem_domain)
        
        # Primary expert phase
        for expert in primary_experts:
            expert_node = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                f"{expert.name} Expert", f"Provides {expert.specialty} expertise", 
                f"expert_{expert.id}",
                f"Provide {expert.specialty} analysis for: {{problem_analysis}}"
            ))
            builder.connect(analyzer, expert_node, graphbit.PyWorkflowEdge.data_flow())
            expert_nodes.append(expert_node)
        
        # Synthesis node
        synthesizer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
            "Expert Synthesizer", "Synthesizes expert opinions", "synthesizer",
            "Combine expert analyses into coherent solution: {expert_opinions}"
        ))
        
        for expert_node in expert_nodes:
            builder.connect(expert_node, synthesizer, graphbit.PyWorkflowEdge.data_flow())
        
        # Secondary expert review
        for expert in secondary_experts:
            reviewer = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                f"{expert.name} Reviewer", f"Reviews from {expert.specialty} perspective",
                f"reviewer_{expert.id}",
                f"Review solution from {expert.specialty} perspective: {{synthesized_solution}}"
            ))
            builder.connect(synthesizer, reviewer, graphbit.PyWorkflowEdge.data_flow())
    
    else:
        # Simple expert consultation
        experts = expert_selector.get_basic_experts(problem_domain)
        for expert in experts:
            expert_node = builder.add_node(graphbit.PyWorkflowNode.agent_node(
                f"{expert.name} Expert", f"Provides {expert.specialty} expertise",
                f"expert_{expert.id}",
                f"Provide {expert.specialty} solution for: {{problem_analysis}}"
            ))
            builder.connect(analyzer, expert_node, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Dynamic expert team creation
team_workflow = create_dynamic_expert_team(
    problem_domain="machine_learning",
    complexity="high",
    available_experts=expert_database.get_available_experts()
)
```

**Examples**: [Multi-Agent Workflows](02-use-case-examples.md#multi-agent-examples)

---

## ⚡ Performance & Scaling

Build high-performance workflows that scale with your needs.

### Concurrent Execution
```python
# Execute multiple workflows in parallel
executor = graphbit.PyWorkflowExecutor(config)

# Configure performance optimizations
pool_config = graphbit.PyPoolConfig(max_size=100, min_size=10)
retry_config = graphbit.PyRetryConfig(max_attempts=3, backoff_ms=1000)

executor.configure_pools(pool_config)
executor.configure_retries(retry_config)

# Monitor performance
stats = executor.get_performance_stats()
print(f"Success rate: {stats.success_rate()}")
```

### Memory Pool Optimization
Configure object pools and circuit breakers for production workloads.

**Learn more**: [Performance Guide](05-complete-api-reference.md#performance)

---

## 🔌 LLM Provider Integration

Connect with any LLM provider seamlessly through Python.

### Multiple Providers
```python
# OpenAI for production
openai_config = graphbit.PyLlmConfig.openai(api_key, "gpt-4")

# Anthropic for analysis
anthropic_config = graphbit.PyLlmConfig.anthropic(api_key, "claude-3-sonnet")

# HuggingFace for open-source models
huggingface_config = graphbit.PyLlmConfig.huggingface(hf_token, "microsoft/DialoGPT-medium")

# Local models with Ollama
ollama_config = graphbit.PyLlmConfig.ollama("localhost:11434", "llama2")
```

### Provider Fallback Strategies
Implement robust fallback chains for high availability.

```python
# Primary and fallback configurations
primary_config = graphbit.PyLlmConfig.openai(openai_key, "gpt-4")
fallback_config = graphbit.PyLlmConfig.huggingface(hf_token, "microsoft/DialoGPT-medium")

# Use in executor with automatic fallback
executor = graphbit.PyWorkflowExecutor(primary_config)
executor.add_fallback_config(fallback_config)
```

**Learn more**: [LLM Integration](05-complete-api-reference.md#pyllmconfig) | [HuggingFace Guide](08-huggingface-integration.md) | [Local LLMs](03-local-llm-integration.md)

---

## 🔧 Advanced Python Features

### Dynamic Workflow Generation
```python
# Generate workflows programmatically
def create_dynamic_workflow(task_type, complexity):
    builder = graphbit.PyWorkflowBuilder(f"Dynamic-{task_type}")
    
    if complexity == "simple":
        return create_single_agent_workflow(builder, task_type)
    elif complexity == "complex":
        return create_multi_stage_workflow(builder, task_type)
    else:
        return create_adaptive_workflow(builder, task_type)
```

### Workflow Factory Pattern
```python
class WorkflowFactory:
    """Factory for creating workflows based on runtime parameters"""
    
    @staticmethod
    def create_workflow(workflow_type: str, **kwargs) -> graphbit.PyWorkflow:
        if workflow_type == "content_creation":
            return WorkflowFactory._create_content_workflow(**kwargs)
        elif workflow_type == "data_analysis":
            return WorkflowFactory._create_analysis_workflow(**kwargs)
        elif workflow_type == "business_process":
            return WorkflowFactory._create_business_workflow(**kwargs)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    @staticmethod
    def _create_content_workflow(content_type, target_audience, complexity_level):
        builder = graphbit.PyWorkflowBuilder(f"Content-{content_type}-{complexity_level}")
        
        # Build workflow based on parameters
        if content_type == "blog" and complexity_level >= 3:
            # Complex blog workflow with research, outline, writing, editing
            pass
        elif content_type == "social" and target_audience == "professional":
            # Professional social media workflow
            pass
        
        return builder.build()

# Usage
workflow = WorkflowFactory.create_workflow(
    "content_creation",
    content_type="blog",
    target_audience="developers", 
    complexity_level=3
)
```

### Async/Await Support
```python
import asyncio

async def run_workflows_concurrently():
    executor = graphbit.PyWorkflowExecutor(config)
    
    # Run multiple workflows concurrently
    tasks = [
        executor.execute_async(workflow1),
        executor.execute_async(workflow2),
        executor.execute_async(workflow3)
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Context Managers
```python
# Use context managers for resource management
with graphbit.PyWorkflowExecutor(config) as executor:
    executor.set_variable("input", "data")
    context = executor.execute(workflow)
    
    # Automatic cleanup when exiting context
```

**Learn more**: [Advanced Features](05-complete-api-reference.md#advanced-features)

---

## 📚 Complete Documentation

### Core Python Guides
- **[Project Overview](index.md)** - What GraphBit is and why use it
- **[Installation Guide](00-installation-setup.md)** - Setup instructions for Python environments  
- **[Quick Start](01-getting-started-workflows.md)** - Build your first workflow in 5 minutes
- **[HuggingFace Integration](08-huggingface-integration.md)** - Open-source models and cost-effective AI
- **[Embeddings Guide](07-embeddings-guide.md)** - Semantic search, similarity analysis, and RAG pipelines
- **[Complete API Reference](05-complete-api-reference.md)** - Full Python library documentation

### Practical Python Examples
- **[Use Case Examples](02-use-case-examples.md)** - Real-world Python workflow examples
- **[Local LLM Integration](03-local-llm-integration.md)** - Using Ollama with Python
- **[Document Processing](04-document-processing-guide.md)** - Working with documents in Python

### Python Developer Resources  
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to GraphBit
- **[API Quick Reference](06-api-quick-reference.md)** - Python API cheat sheet
- **[Changelog](CHANGELOG.md)** - Version history and updates

---

## 🐍 Python-Specific Features

### Type Hints Support
```python
from typing import Optional, Dict, Any
import graphbit

def create_typed_workflow(config: graphbit.PyLlmConfig) -> graphbit.PyWorkflow:
    builder = graphbit.PyWorkflowBuilder("Typed Workflow")
    # Your workflow logic here
    return builder.build()

def execute_with_types(
    workflow: graphbit.PyWorkflow, 
    variables: Dict[str, Any]
) -> Optional[graphbit.PyWorkflowContext]:
    executor = graphbit.PyWorkflowExecutor(config)
    return executor.execute(workflow)
```

### Integration with Python Ecosystem
```python
# Works with popular Python libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Process DataFrame with GraphBit
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    workflow = create_data_processing_workflow()
    executor = graphbit.PyWorkflowExecutor(config)
    
    for index, row in df.iterrows():
        executor.set_variable("data", row.to_dict())
        context = executor.execute(workflow)
        # Process results back to DataFrame
    
    return df
```

---
