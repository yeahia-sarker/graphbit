# Examples

This page contains practical examples of using GraphBit for various use cases.

## Embeddings & Semantic Analysis Examples

### Intelligent Document Search

Build a semantic search system that understands the meaning of your documents:

```python
import asyncio
import graphbit
import os

async def create_document_search_system():
    # Setup embedding service
    embedding_config = graphbit.PyEmbeddingConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    
    # Document corpus
    documents = [
        "GraphBit is a declarative agentic workflow automation framework for Python",
        "Machine learning enables computers to learn patterns from data without explicit programming",
        "Natural language processing helps computers understand and generate human language",
        "Vector databases store and search high-dimensional embeddings efficiently",
        "RAG systems combine retrieval and generation for better AI responses"
    ]
    
    # Generate embeddings for all documents
    print("Generating embeddings for document corpus...")
    doc_embeddings = await embedding_service.embed_texts(documents)
    
    # Search function
    async def search_documents(query, top_k=3):
        query_embedding = await embedding_service.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                query_embedding, doc_embedding
            )
            similarities.append((documents[i], similarity))
        
        # Return top matches
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Test searches
    queries = [
        "workflow automation tools",
        "AI and machine learning",
        "database for vectors"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = await search_documents(query)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.3f} - {doc}")

# Run the example
asyncio.run(create_document_search_system())
```

### RAG-Powered Q&A System

Combine embeddings with workflows for intelligent question answering:

```python
import asyncio
import graphbit
import os

class RAGQuestionAnswering:
    def __init__(self, embedding_service, llm_config):
        self.embedding_service = embedding_service
        self.llm_config = llm_config
        self.knowledge_base = []
        self.embeddings = []
    
    async def add_knowledge(self, documents):
        """Add documents to the knowledge base"""
        self.knowledge_base.extend(documents)
        new_embeddings = await self.embedding_service.embed_texts(documents)
        self.embeddings.extend(new_embeddings)
        print(f"Added {len(documents)} documents to knowledge base")
    
    async def answer_question(self, question):
        """Answer a question using RAG approach"""
        # Find relevant context using embeddings
        question_embedding = await self.embedding_service.embed_text(question)
        
        # Calculate similarities and get top 3 documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                question_embedding, doc_embedding
            )
            similarities.append((i, similarity))
        
        # Get top 3 most relevant documents
        top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        context = "\n\n".join([self.knowledge_base[i] for i, _ in top_docs])
        
        # Create RAG workflow
        builder = graphbit.PyWorkflowBuilder("RAG Q&A")
        
        rag_agent = graphbit.PyWorkflowNode.agent_node(
            "RAG Agent",
            "Answers questions using provided context",
            "rag_agent",
            f"""Based on the following context, provide a comprehensive answer to the question.
            If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        builder.add_node(rag_agent)
        workflow = builder.build()
        
        # Execute workflow
        executor = graphbit.PyWorkflowExecutor(self.llm_config)
        result = executor.execute(workflow)
        
        return {
            "answer": "Generated answer based on context",  # Extract from result
            "context_docs": [self.knowledge_base[i] for i, _ in top_docs],
            "relevance_scores": [score for _, score in top_docs]
        }

async def rag_qa_example():
    # Setup services
    embedding_config = graphbit.PyEmbeddingConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    llm_config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")
    
    # Create RAG system
    rag_system = RAGQuestionAnswering(embedding_service, llm_config)
    
    # Add knowledge base
    knowledge_docs = [
        "GraphBit is a Python framework for building agentic workflows with LLMs",
        "GraphBit supports OpenAI, Anthropic, and local LLMs through unified APIs",
        "Embeddings in GraphBit enable semantic search and similarity analysis",
        "GraphBit workflows can be executed synchronously or asynchronously",
        "The framework includes performance optimizations like memory pools and circuit breakers",
        "GraphBit provides both static and dynamic workflow generation capabilities"
    ]
    
    await rag_system.add_knowledge(knowledge_docs)
    
    # Ask questions
    questions = [
        "What is GraphBit and what can it do?",
        "How does GraphBit handle different LLM providers?",
        "What performance optimizations does GraphBit offer?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = await rag_system.answer_question(question)
        print(f"Answer: {result['answer']}")
        print("Relevant sources:")
        for i, (doc, score) in enumerate(zip(result['context_docs'], result['relevance_scores']), 1):
            print(f"  {i}. ({score:.3f}) {doc[:100]}...")

asyncio.run(rag_qa_example())
```

### Content Similarity Analysis

Analyze and group similar content automatically:

```python
import asyncio
import graphbit
import os

async def content_similarity_analysis():
    # Setup
    config = graphbit.PyEmbeddingConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    service = graphbit.PyEmbeddingService(config)
    
    # Content to analyze
    articles = [
        "Introduction to machine learning algorithms and their applications",
        "Deep learning and neural networks for image recognition",
        "Best practices for cooking Italian pasta dishes",
        "Python programming tips for beginners",
        "Advanced machine learning techniques for data scientists",
        "Traditional Italian recipes passed down through generations",
        "JavaScript frameworks for modern web development",
        "Artificial intelligence in healthcare diagnostics"
    ]
    
    # Generate embeddings
    embeddings = await service.embed_texts(articles)
    print(f"Generated embeddings for {len(articles)} articles")
    
    # Find most similar pairs
    similarities = []
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                embeddings[i], embeddings[j]
            )
            similarities.append((i, j, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Display top 5 most similar pairs
    print("\nMost similar article pairs:")
    for i, j, sim in similarities[:5]:
        print(f"\nSimilarity: {sim:.3f}")
        print(f"  Article {i}: {articles[i]}")
        print(f"  Article {j}: {articles[j]}")
    
    # Simple clustering based on similarity threshold
    print("\n" + "="*50)
    print("CONTENT CLUSTERING")
    print("="*50)
    
    clusters = {}
    assigned = set()
    cluster_id = 0
    threshold = 0.4  # Similarity threshold for clustering
    
    for i, embedding_i in enumerate(embeddings):
        if i in assigned:
            continue
        
        # Start new cluster
        clusters[cluster_id] = [i]
        assigned.add(i)
        
        # Find similar articles
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
    for cluster_id, indices in clusters.items():
        print(f"\nCluster {cluster_id + 1} ({len(indices)} articles):")
        for idx in indices:
            print(f"  - {articles[idx]}")

asyncio.run(content_similarity_analysis())
```

### Smart Content Recommendation Workflow

Combine embeddings with workflow automation for intelligent content recommendations:

```python
import asyncio
import graphbit
import os

async def smart_content_recommendation():
    # Setup services
    embedding_config = graphbit.PyEmbeddingConfig.openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    embedding_service = graphbit.PyEmbeddingService(embedding_config)
    llm_config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")
    
    # Content database
    content_library = [
        "Getting started with Python programming fundamentals",
        "Advanced Python decorators and metaclasses",
        "Machine learning with scikit-learn tutorial",
        "Deep learning using PyTorch for beginners",
        "Web development with Django framework",
        "Data analysis using pandas and numpy",
        "Building REST APIs with FastAPI",
        "Docker containerization for Python applications"
    ]
    
    # Generate embeddings for content library
    content_embeddings = await embedding_service.embed_texts(content_library)
    
    async def recommend_content(user_interests, num_recommendations=3):
        # Generate embedding for user interests
        interests_embedding = await embedding_service.embed_text(user_interests)
        
        # Find most relevant content
        similarities = []
        for i, content_embedding in enumerate(content_embeddings):
            similarity = graphbit.PyEmbeddingService.cosine_similarity(
                interests_embedding, content_embedding
            )
            similarities.append((i, similarity))
        
        # Get top matches
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_recommendations]
        recommended_content = [content_library[i] for i, _ in top_matches]
        
        # Create workflow to generate personalized recommendations
        builder = graphbit.PyWorkflowBuilder("Content Recommendation")
        
        recommender_agent = graphbit.PyWorkflowNode.agent_node(
            "Content Recommender",
            "Creates personalized content recommendations",
            "recommender",
            f"""Based on the user's interests: "{user_interests}"
            
And these relevant content options:
{chr(10).join(f"- {content}" for content in recommended_content)}

Create a personalized recommendation explanation for each item, explaining why it matches the user's interests and what they'll learn. Format as a numbered list."""
        )
        
        builder.add_node(recommender_agent)
        workflow = builder.build()
        
        # Execute workflow
        executor = graphbit.PyWorkflowExecutor(llm_config)
        result = executor.execute(workflow)
        
        return {
            "recommendations": recommended_content,
            "explanations": "Generated personalized explanations",  # Extract from result
            "similarity_scores": [score for _, score in top_matches]
        }
    
    # Test recommendations for different user profiles
    user_profiles = [
        "I'm interested in learning Python for data science and machine learning",
        "I want to build web applications and APIs using modern frameworks",
        "I'm a beginner looking to start programming with Python basics"
    ]
    
    for user_interests in user_profiles:
        print(f"\nUser Profile: {user_interests}")
        print("-" * 60)
        
        recommendations = await recommend_content(user_interests)
        
        print("Recommended Content:")
        for i, (content, score) in enumerate(zip(
            recommendations["recommendations"], 
            recommendations["similarity_scores"]
        ), 1):
            print(f"{i}. Score: {score:.3f} - {content}")

asyncio.run(smart_content_recommendation())
```

## Basic Examples

### Hello World

```python
import graphbit
import os

# Initialize
graphbit.init()
config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-4")

# Create workflow
builder = graphbit.PyWorkflowBuilder("Hello World")
node = graphbit.PyWorkflowNode.agent_node(
    "Greeter", "Generates greetings", "greeter",
    "Say hello to {name}"
)

node_id = builder.add_node(node)
workflow = builder.build()

# Execute
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("name", "World")
context = executor.execute(workflow)

print(f"Completed: {context.is_completed()}")
```

### Simple Data Processing

```python
# Data transformation pipeline
builder = graphbit.PyWorkflowBuilder("Data Processor")

# Extract data
extractor = graphbit.PyWorkflowNode.agent_node(
    "Data Extractor", "Extracts structured data", "extractor",
    "Extract names and emails from: {raw_text}"
)

# Validate data
validator = graphbit.PyWorkflowNode.condition_node(
    "Data Validator", "Validates extracted data",
    "email_count > 0 && name_count > 0"
)

# Format output
formatter = graphbit.PyWorkflowNode.agent_node(
    "Data Formatter", "Formats data for output", "formatter",
    "Format this data as CSV: {extracted_data}"
)

# Build pipeline
ext_id = builder.add_node(extractor)
val_id = builder.add_node(validator)
fmt_id = builder.add_node(formatter)

builder.connect(ext_id, val_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(val_id, fmt_id, graphbit.PyWorkflowEdge.conditional("passed"))

workflow = builder.build()
```

## Content Creation Examples

### Blog Post Generation

```python
def create_blog_workflow():
    builder = graphbit.PyWorkflowBuilder("Blog Post Generator")
    
    # Research phase
    researcher = graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches topic", "researcher",
        "Research key points about: {topic}. Include statistics and recent developments."
    )
    
    # Outline creation
    outliner = graphbit.PyWorkflowNode.agent_node(
        "Outliner", "Creates content outline", "outliner", 
        "Create a detailed blog post outline for: {topic} using this research: {research}"
    )
    
    # Content writing
    writer = graphbit.PyWorkflowNode.agent_node(
        "Writer", "Writes blog content", "writer",
        "Write a comprehensive blog post using this outline: {outline} and research: {research}"
    )
    
    # SEO optimization
    seo_optimizer = graphbit.PyWorkflowNode.agent_node(
        "SEO Optimizer", "Optimizes for SEO", "seo",
        "Optimize this blog post for SEO, suggest meta description and keywords: {content}"
    )
    
    # Build workflow
    r_id = builder.add_node(researcher)
    o_id = builder.add_node(outliner)
    w_id = builder.add_node(writer)
    s_id = builder.add_node(seo_optimizer)
    
    # Connect pipeline
    builder.connect(r_id, o_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(o_id, w_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(w_id, s_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Usage
workflow = create_blog_workflow()
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("topic", "Machine Learning in Healthcare")
result = executor.execute(workflow)
```

### Social Media Campaign

```python
def create_social_campaign():
    builder = graphbit.PyWorkflowBuilder("Social Media Campaign")
    
    # Campaign strategy
    strategist = graphbit.PyWorkflowNode.agent_node(
        "Campaign Strategist", "Creates campaign strategy", "strategist",
        "Create a social media campaign strategy for: {product} targeting {audience}"
    )
    
    # Platform-specific content
    twitter_writer = graphbit.PyWorkflowNode.agent_node(
        "Twitter Writer", "Writes Twitter content", "twitter",
        "Create 5 Twitter posts for this campaign: {strategy}. Keep under 280 characters each."
    )
    
    linkedin_writer = graphbit.PyWorkflowNode.agent_node(
        "LinkedIn Writer", "Writes LinkedIn content", "linkedin", 
        "Create a professional LinkedIn post for this campaign: {strategy}. Include call-to-action."
    )
    
    instagram_writer = graphbit.PyWorkflowNode.agent_node(
        "Instagram Writer", "Writes Instagram content", "instagram",
        "Create Instagram captions and hashtags for this campaign: {strategy}. Visual-focused approach."
    )
    
    # Content review
    reviewer = graphbit.PyWorkflowNode.agent_node(
        "Content Reviewer", "Reviews all content", "reviewer",
        "Review and ensure consistency across all social media content: {all_content}"
    )
    
    # Build workflow
    strat_id = builder.add_node(strategist)
    tw_id = builder.add_node(twitter_writer)
    li_id = builder.add_node(linkedin_writer)
    ig_id = builder.add_node(instagram_writer)
    rev_id = builder.add_node(reviewer)
    
    # Connect in parallel pattern
    builder.connect(strat_id, tw_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(strat_id, li_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(strat_id, ig_id, graphbit.PyWorkflowEdge.data_flow())
    
    # Merge for review
    builder.connect(tw_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(li_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(ig_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()
```

## Data Analysis Examples

### Customer Feedback Analysis

```python
def create_feedback_analyzer():
    builder = graphbit.PyWorkflowBuilder("Feedback Analyzer")
    
    # Sentiment analysis
    sentiment_analyzer = graphbit.PyWorkflowNode.agent_node(
        "Sentiment Analyzer", "Analyzes sentiment", "sentiment",
        "Analyze the sentiment of this feedback: {feedback}. Rate as positive, negative, or neutral with confidence."
    )
    
    # Category classification
    categorizer = graphbit.PyWorkflowNode.agent_node(
        "Category Classifier", "Classifies feedback category", "categorizer",
        "Classify this feedback into categories (product, service, billing, support): {feedback}"
    )
    
    # Priority assessment
    priority_assessor = graphbit.PyWorkflowNode.agent_node(
        "Priority Assessor", "Assesses urgency", "priority",
        "Assess the priority level (low, medium, high, urgent) of this feedback: {feedback}"
    )
    
    # Action recommendation
    action_recommender = graphbit.PyWorkflowNode.agent_node(
        "Action Recommender", "Recommends actions", "recommender",
        "Based on sentiment: {sentiment}, category: {category}, and priority: {priority}, recommend specific actions."
    )
    
    # Build parallel analysis workflow
    sent_id = builder.add_node(sentiment_analyzer)
    cat_id = builder.add_node(categorizer)
    pri_id = builder.add_node(priority_assessor)
    act_id = builder.add_node(action_recommender)
    
    # Parallel processing
    builder.connect(sent_id, act_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(cat_id, act_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(pri_id, act_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()
```

### Sales Data Insights

```python
def create_sales_analyzer():
    builder = graphbit.PyWorkflowBuilder("Sales Data Analyzer")
    
    # Data validation
    validator = graphbit.PyWorkflowNode.condition_node(
        "Data Validator", "Validates sales data",
        "revenue > 0 && date_range_valid == true && customer_count > 0"
    )
    
    # Trend analysis
    trend_analyzer = graphbit.PyWorkflowNode.agent_node(
        "Trend Analyzer", "Analyzes sales trends", "trends",
        "Analyze these sales trends and identify patterns: {sales_data}"
    )
    
    # Performance comparison
    comparator = graphbit.PyWorkflowNode.agent_node(
        "Performance Comparator", "Compares performance periods", "comparator",
        "Compare current performance with previous period: {current_data} vs {previous_data}"
    )
    
    # Forecast generation
    forecaster = graphbit.PyWorkflowNode.agent_node(
        "Sales Forecaster", "Generates forecasts", "forecaster",
        "Based on trends: {trends} and comparisons: {comparison}, forecast next quarter's performance."
    )
    
    # Executive summary
    summarizer = graphbit.PyWorkflowNode.agent_node(
        "Executive Summarizer", "Creates executive summary", "summarizer",
        "Create an executive summary of sales analysis including trends, comparisons, and forecasts."
    )
    
    # Build analytical workflow
    val_id = builder.add_node(validator)
    trend_id = builder.add_node(trend_analyzer)
    comp_id = builder.add_node(comparator)
    fore_id = builder.add_node(forecaster)
    summ_id = builder.add_node(summarizer)
    
    builder.connect(val_id, trend_id, graphbit.PyWorkflowEdge.conditional("passed"))
    builder.connect(val_id, comp_id, graphbit.PyWorkflowEdge.conditional("passed"))
    builder.connect(trend_id, fore_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(comp_id, fore_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(fore_id, summ_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()
```

## Advanced Examples

### Multi-Stage Quality Control

```python
def create_quality_control_workflow():
    builder = graphbit.PyWorkflowBuilder("Quality Control Pipeline")
    
    # Initial content analysis
    initial_analyzer = graphbit.PyWorkflowNode.agent_node(
        "Initial Analyzer", "First-pass content analysis", "initial",
        "Analyze this content for basic quality metrics: {content}"
    )
    
    # Quality gate 1: Basic standards
    quality_gate_1 = graphbit.PyWorkflowNode.condition_node(
        "Quality Gate 1", "Basic quality check",
        "word_count >= 100 && readability_score >= 60"
    )
    
    # Enhancement for failed content
    enhancer = graphbit.PyWorkflowNode.agent_node(
        "Content Enhancer", "Improves content quality", "enhancer",
        "Improve this content to meet quality standards: {content}"
    )
    
    # Advanced analysis
    advanced_analyzer = graphbit.PyWorkflowNode.agent_node(
        "Advanced Analyzer", "Deep content analysis", "advanced",
        "Perform detailed analysis for accuracy, tone, and engagement: {content}"
    )
    
    # Quality gate 2: Advanced standards
    quality_gate_2 = graphbit.PyWorkflowNode.condition_node(
        "Quality Gate 2", "Advanced quality check",
        "accuracy_score >= 85 && engagement_score >= 75"
    )
    
    # Final polish
    polisher = graphbit.PyWorkflowNode.agent_node(
        "Content Polisher", "Final content polish", "polisher",
        "Apply final polish and optimization to this content: {content}"
    )
    
    # Approval workflow
    ia_id = builder.add_node(initial_analyzer)
    qg1_id = builder.add_node(quality_gate_1)
    enh_id = builder.add_node(enhancer)
    aa_id = builder.add_node(advanced_analyzer)
    qg2_id = builder.add_node(quality_gate_2)
    pol_id = builder.add_node(polisher)
    
    # Build quality control flow
    builder.connect(ia_id, qg1_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(qg1_id, enh_id, graphbit.PyWorkflowEdge.conditional("failed"))
    builder.connect(qg1_id, aa_id, graphbit.PyWorkflowEdge.conditional("passed"))
    builder.connect(enh_id, aa_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(aa_id, qg2_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(qg2_id, pol_id, graphbit.PyWorkflowEdge.conditional("passed"))
    
    return builder.build()
```

### Dynamic Workflow Generation

```python
async def create_dynamic_workflow(task_type: str, complexity: str):
    """Creates workflows dynamically based on parameters"""
    
    builder = graphbit.PyWorkflowBuilder(f"Dynamic {task_type} Workflow")
    
    if task_type == "research":
        # Research workflow
        nodes = [
            ("researcher", "Conducts primary research", f"Research {complexity} level information about: {{topic}}"),
            ("fact_checker", "Validates information", f"Verify the accuracy of this {complexity} research: {{research}}"),
            ("synthesizer", "Synthesizes findings", f"Create a {complexity} synthesis of: {{validated_research}}")
        ]
    elif task_type == "creative":
        # Creative workflow  
        nodes = [
            ("ideator", "Generates creative ideas", f"Generate {complexity} creative ideas for: {{brief}}"),
            ("developer", "Develops concepts", f"Develop these ideas into {complexity} concepts: {{ideas}}"),
            ("refiner", "Refines output", f"Refine these {complexity} concepts for final presentation: {{concepts}}")
        ]
    else:
        # Default analytical workflow
        nodes = [
            ("analyzer", "Analyzes data", f"Perform {complexity} analysis of: {{data}}"),
            ("interpreter", "Interprets results", f"Interpret these {complexity} analysis results: {{analysis}}"),
            ("recommender", "Makes recommendations", f"Make {complexity} recommendations based on: {{interpretation}}")
        ]
    
    # Add nodes and connect in sequence
    node_ids = []
    for agent_id, description, prompt in nodes:
        node = graphbit.PyWorkflowNode.agent_node(
            agent_id.title(), description, agent_id, prompt
        )
        node_id = builder.add_node(node)
        node_ids.append(node_id)
    
    # Connect sequentially
    for i in range(len(node_ids) - 1):
        builder.connect(node_ids[i], node_ids[i+1], graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Usage
research_workflow = await create_dynamic_workflow("research", "advanced")
creative_workflow = await create_dynamic_workflow("creative", "basic")
```

## Performance Optimization Examples

### High-Throughput Processing

```python
def setup_high_throughput_executor():
    config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo")
    
    # Configure for high throughput
    executor = (graphbit.PyWorkflowExecutor.new_high_throughput(config)
        .with_max_concurrent_workflows(20)
        .with_memory_pool_enabled(True)
        .with_uuid_pool_size(1000))
    
    # Add retry logic
    retry_config = (graphbit.PyRetryConfig.new(3)
        .with_exponential_backoff(500, 2.0, 5000)
        .with_jitter(0.1))
    
    # Add circuit breaker
    circuit_config = graphbit.PyCircuitBreakerConfig.new(5, 30000)
    
    return (executor
        .with_retry_config(retry_config)
        .with_circuit_breaker_config(circuit_config))

# Process batch of items
async def process_batch(items, workflow):
    executor = setup_high_throughput_executor()
    
    tasks = []
    for item in items:
        executor.set_variable("input", item)
        task = executor.execute_async(workflow)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    print(f"Processed {len(successful)}/{len(items)} successfully")
    return successful, failed
```

### Low-Latency Configuration

```python
def setup_low_latency_executor():
    config = graphbit.PyLlmConfig.openai(os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo")
    
    # Configure for low latency
    executor = (graphbit.PyWorkflowExecutor.new_low_latency(config)
        .with_max_node_execution_time(5000)  # 5 second timeout
        .with_fail_fast(True)
        .with_memory_pool_enabled(True))
    
    # Aggressive retry with short backoff
    retry_config = (graphbit.PyRetryConfig.new(2)
        .with_exponential_backoff(100, 1.5, 1000))
    
    return executor.with_retry_config(retry_config)
```

## Error Handling Examples

### Comprehensive Error Handling

```python
async def robust_workflow_execution(workflow, inputs):
    executor = graphbit.PyWorkflowExecutor(config)
    
    try:
        # Set variables
        for key, value in inputs.items():
            executor.set_variable(key, value)
        
        # Execute with timeout
        context = await asyncio.wait_for(
            executor.execute_async(workflow),
            timeout=30.0
        )
        
        if context.is_completed():
            return {"status": "success", "context": context}
        else:
            return {"status": "failed", "error": "Workflow not completed"}
            
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Workflow execution timed out"}
    except ValueError as e:
        return {"status": "validation_error", "error": str(e)}
    except ConnectionError as e:
        return {"status": "connection_error", "error": str(e)}
    except Exception as e:
        return {"status": "unknown_error", "error": str(e)}

# Usage with multiple workflows
async def process_with_fallback(primary_workflow, fallback_workflow, inputs):
    result = await robust_workflow_execution(primary_workflow, inputs)
    
    if result["status"] != "success":
        print(f"Primary workflow failed: {result['error']}")
        print("Trying fallback workflow...")
        result = await robust_workflow_execution(fallback_workflow, inputs)
    
    return result
```

## Additional Resources

- [Workflow Guide](quick-start.md) - Basic patterns and getting started
- [Python Library Documentation](python_library_documentation.md) - Complete API reference

## Community Examples

Join our [Discord community](https://discord.gg/graphbit) to share your own examples and learn from others!

---

*Have a great example to share? [Contribute to our documentation](CONTRIBUTING.md) or submit a pull request!* 
