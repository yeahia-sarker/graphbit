# 🚀 Getting Started with Workflows

Build your first AI workflows in minutes with this hands-on guide. We'll start simple and build up to more sophisticated patterns, covering both static and dynamic workflow creation.

## 🎯 What You'll Learn

- Create your first workflow in 5 minutes
- Build multi-agent pipelines
- Add conditional logic and branching
- Create dynamic workflows that adapt at runtime
- Handle errors and retries
- Optimize for performance

**Prerequisites**: [Installation complete](00-installation-setup.md) ✅

---

## 🏃‍♂️ 5-Minute Tutorial

Let's build your first workflow step-by-step:

### Step 1: Basic Setup
```python
import graphbit
import os

# Initialize GraphBit (always required)
graphbit.init()

# Configure your LLM provider
config = graphbit.PyLlmConfig.openai(
    os.getenv("OPENAI_API_KEY"), 
    "gpt-4"
)
```

### Step 2: Create Your First Workflow
```python
# Create a workflow builder
builder = graphbit.PyWorkflowBuilder("My First Workflow")
builder.description("A simple greeting workflow")

# Create an agent node
greeter = graphbit.PyWorkflowNode.agent_node(
    name="Greeter",
    description="Generates personalized greetings",
    agent_id="greeter",
    prompt="Create a friendly greeting for {name} who is a {profession}"
)

# Add node and build workflow
node_id = builder.add_node(greeter)
workflow = builder.build()
```

### Step 3: Execute and See Results
```python
# Create executor and set variables
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("name", "Alice")
executor.set_variable("profession", "software engineer")

# Execute the workflow
context = executor.execute(workflow)

# Check results
if context.is_completed():
    print("🎉 Workflow completed successfully!")
    print(f"⏱️  Execution time: {context.execution_time_ms()}ms")
else:
    print("❌ Workflow failed")
```

**Congratulations!** You've built your first AI workflow. 🎉

---

## 🏗️ Static vs Dynamic Workflows

Understanding the difference between static and dynamic workflows is crucial for building practical Python applications.

### Static Workflows
**Best for**: Predictable, repeatable processes with fixed structure

```python
def create_static_content_pipeline():
    """Static workflow - structure is fixed at design time"""
    builder = graphbit.PyWorkflowBuilder("Static Content Pipeline")
    
    # Fixed pipeline: Research → Write → Edit
    researcher = graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches topics", "research",
        "Research key points about: {topic}"
    )
    
    writer = graphbit.PyWorkflowNode.agent_node(
        "Writer", "Writes content", "writer",
        "Write a blog post using research: {research}"
    )
    
    editor = graphbit.PyWorkflowNode.agent_node(
        "Editor", "Edits content", "editor",
        "Edit and improve content: {content}"
    )
    
    # Fixed connections
    researcher_id = builder.add_node(researcher)
    writer_id = builder.add_node(writer)
    editor_id = builder.add_node(editor)
    
    builder.connect(researcher_id, writer_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(writer_id, editor_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Always produces the same workflow structure
static_workflow = create_static_content_pipeline()
```

### Dynamic Workflows  
**Best for**: Adaptive processes that change based on runtime conditions

```python
def create_dynamic_content_pipeline(content_type, complexity_level, has_deadline):
    """Dynamic workflow - structure varies based on parameters"""
    builder = graphbit.PyWorkflowBuilder(f"Dynamic-{content_type}-L{complexity_level}")
    
    # Start with research (always needed)
    researcher = graphbit.PyWorkflowNode.agent_node(
        "Researcher", "Researches content", "research",
        f"Research {content_type} topics: {{topic}}"
    )
    researcher_id = builder.add_node(researcher)
    previous_node_id = researcher_id
    
    # Add outline step for complex content
    if complexity_level >= 2:
        outliner = graphbit.PyWorkflowNode.agent_node(
            "Outliner", "Creates detailed outline", "outline",
            f"Create detailed {content_type} outline: {{research}}"
        )
        outliner_id = builder.add_node(outliner)
        builder.connect(previous_node_id, outliner_id, graphbit.PyWorkflowEdge.data_flow())
        previous_node_id = outliner_id
    
    # Different writers for different content types
    if content_type == "blog":
        writer = graphbit.PyWorkflowNode.agent_node(
            "Blog Writer", "Writes blog posts", "blog_writer",
            "Write comprehensive blog post: {input_data}"
        )
    elif content_type == "email":
        writer = graphbit.PyWorkflowNode.agent_node(
            "Email Writer", "Writes emails", "email_writer", 
            "Write professional email: {input_data}"
        )
    else:
        writer = graphbit.PyWorkflowNode.agent_node(
            "General Writer", "Writes content", "writer",
            "Write content: {input_data}"
        )
    
    writer_id = builder.add_node(writer)
    builder.connect(previous_node_id, writer_id, graphbit.PyWorkflowEdge.data_flow())
    previous_node_id = writer_id
    
    # Skip editing if there's a tight deadline
    if not has_deadline:
        editor = graphbit.PyWorkflowNode.agent_node(
            "Editor", "Edits and polishes", "editor",
            f"Edit and polish {content_type}: {{content}}"
        )
        editor_id = builder.add_node(editor)
        builder.connect(previous_node_id, editor_id, graphbit.PyWorkflowEdge.data_flow())
        previous_node_id = editor_id
    
    # Add quality check for high complexity
    if complexity_level >= 3:
        quality_checker = graphbit.PyWorkflowNode.agent_node(
            "Quality Checker", "Performs quality check", "quality",
            f"Perform final quality check on {content_type}: {{final_content}}"
        )
        quality_id = builder.add_node(quality_checker)
        builder.connect(previous_node_id, quality_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Creates different workflows based on parameters
blog_workflow = create_dynamic_content_pipeline("blog", 3, False)
email_workflow = create_dynamic_content_pipeline("email", 1, True)
```

---

## 🔗 Multi-Agent Workflows

Now let's build something more interesting - a content creation pipeline:

### Static Content Creation Pipeline
```python
def create_content_pipeline():
    builder = graphbit.PyWorkflowBuilder("Content Creation Pipeline")
    
    # Step 1: Generate ideas
    ideator = graphbit.PyWorkflowNode.agent_node(
        "Ideator",
        "Generates creative content ideas",
        "ideator", 
        "Generate 3 creative content ideas about: {topic}"
    )
    
    # Step 2: Write content
    writer = graphbit.PyWorkflowNode.agent_node(
        "Writer",
        "Writes content based on ideas",
        "writer",
        "Write a blog post about {topic} using these ideas: {ideas}"
    )
    
    # Step 3: Edit and improve
    editor = graphbit.PyWorkflowNode.agent_node(
        "Editor", 
        "Edits and improves content",
        "editor",
        "Edit and improve this content for clarity and engagement: {content}"
    )
    
    # Build the pipeline: Ideator → Writer → Editor
    ideator_id = builder.add_node(ideator)
    writer_id = builder.add_node(writer)
    editor_id = builder.add_node(editor)
    
    # Connect nodes in sequence
    builder.connect(ideator_id, writer_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(writer_id, editor_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Execute the pipeline
workflow = create_content_pipeline()
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("topic", "artificial intelligence in healthcare")

context = executor.execute(workflow)
print(f"Content pipeline completed: {context.is_completed()}")
```

### Dynamic Team Assembly
```python
def create_dynamic_content_team(content_requirements):
    """Assemble content team based on specific requirements"""
    builder = graphbit.PyWorkflowBuilder("Dynamic Content Team")
    
    # Always start with content strategist
    strategist = graphbit.PyWorkflowNode.agent_node(
        "Content Strategist", "Develops content strategy", "strategist",
        f"Develop strategy for {content_requirements['type']} targeting {content_requirements['audience']}: {{brief}}"
    )
    strategist_id = builder.add_node(strategist)
    previous_nodes = [strategist_id]
    
    # Add specialists based on requirements
    team_nodes = []
    
    if content_requirements.get("needs_research", False):
        researcher = graphbit.PyWorkflowNode.agent_node(
            "Research Specialist", "Conducts research", "researcher",
            "Research relevant information for content strategy: {strategy}"
        )
        researcher_id = builder.add_node(researcher)
        builder.connect(strategist_id, researcher_id, graphbit.PyWorkflowEdge.data_flow())
        team_nodes.append(researcher_id)
    
    if content_requirements.get("content_type") == "technical":
        tech_writer = graphbit.PyWorkflowNode.agent_node(
            "Technical Writer", "Writes technical content", "tech_writer",
            "Write technical content based on strategy: {strategy}"
        )
        tech_writer_id = builder.add_node(tech_writer)
        builder.connect(strategist_id, tech_writer_id, graphbit.PyWorkflowEdge.data_flow())
        team_nodes.append(tech_writer_id)
    
    if content_requirements.get("content_type") == "marketing":
        marketing_writer = graphbit.PyWorkflowNode.agent_node(
            "Marketing Writer", "Writes marketing content", "marketing_writer",
            "Write persuasive marketing content: {strategy}"
        )
        marketing_writer_id = builder.add_node(marketing_writer)
        builder.connect(strategist_id, marketing_writer_id, graphbit.PyWorkflowEdge.data_flow())
        team_nodes.append(marketing_writer_id)
    
    # Add quality assurance if high quality is required
    if content_requirements.get("quality_level", "standard") == "premium":
        qa_specialist = graphbit.PyWorkflowNode.agent_node(
            "QA Specialist", "Performs quality assurance", "qa",
            "Review all content for premium quality standards: {all_content}"
        )
        qa_id = builder.add_node(qa_specialist)
        
        # Connect all team members to QA
        for team_node in team_nodes:
            builder.connect(team_node, qa_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Example usage
requirements = {
    "type": "blog_post",
    "audience": "software_developers", 
    "content_type": "technical",
    "needs_research": True,
    "quality_level": "premium"
}

dynamic_team = create_dynamic_content_team(requirements)
```

### How Data Flows
```
Input: topic = "AI in healthcare"
   ↓
[Ideator] → ideas = "1. AI diagnostics, 2. Patient monitoring, 3. Drug discovery"
   ↓  
[Writer] → content = "Blog post about AI transforming healthcare..."
   ↓
[Editor] → final_content = "Polished blog post with improved clarity..."
```

---

## 🔀 Conditional Workflows

Add decision-making logic to your workflows:

### Quality Control Workflow
```python
def create_quality_workflow():
    builder = graphbit.PyWorkflowBuilder("Quality Control")
    
    # Analyze content quality
    analyzer = graphbit.PyWorkflowNode.agent_node(
        "Content Analyzer",
        "Analyzes content quality", 
        "analyzer",
        "Analyze this content and rate its quality (1-10): {content}"
    )
    
    # Quality gate - decides what happens next
    quality_gate = graphbit.PyWorkflowNode.condition_node(
        "Quality Gate",
        "Checks if content meets quality standards",
        "quality_score >= 7"  # Condition expression
    )
    
    # Improvement path (only if quality is low)
    improver = graphbit.PyWorkflowNode.agent_node(
        "Content Improver",
        "Improves low-quality content",
        "improver", 
        "Improve this content to make it higher quality: {content}"
    )
    
    # Approval path (for high-quality content)
    approver = graphbit.PyWorkflowNode.agent_node(
        "Content Approver",
        "Approves high-quality content",
        "approver",
        "Add final approval and publishing metadata: {content}"
    )
    
    # Build conditional workflow
    analyzer_id = builder.add_node(analyzer)
    gate_id = builder.add_node(quality_gate)
    improver_id = builder.add_node(improver)
    approver_id = builder.add_node(approver)
    
    # Connect with conditional logic
    builder.connect(analyzer_id, gate_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(gate_id, improver_id, graphbit.PyWorkflowEdge.conditional("failed"))
    builder.connect(gate_id, approver_id, graphbit.PyWorkflowEdge.conditional("passed"))
    
    return builder.build()

# Test with different content quality
workflow = create_quality_workflow()
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("content", "Your content to analyze here...")

context = executor.execute(workflow)
```

### Dynamic Conditional Logic
```python
def create_adaptive_approval_workflow(approval_rules):
    """Create approval workflow based on dynamic rules"""
    builder = graphbit.PyWorkflowBuilder("Adaptive Approval")
    
    # Content processor
    processor = graphbit.PyWorkflowNode.agent_node(
        "Content Processor", "Processes content for approval", "processor",
        "Process and categorize content for approval workflow: {content}"
    )
    processor_id = builder.add_node(processor)
    
    # Create approval chain based on rules
    previous_node_id = processor_id
    
    for i, rule in enumerate(approval_rules):
        # Create condition node
        condition = graphbit.PyWorkflowNode.condition_node(
            f"Approval Condition {i+1}", 
            f"Checks {rule['criteria']}", 
            rule['condition_expression']
        )
        condition_id = builder.add_node(condition)
        builder.connect(previous_node_id, condition_id, graphbit.PyWorkflowEdge.data_flow())
        
        # Create approver for this level
        approver = graphbit.PyWorkflowNode.agent_node(
            f"{rule['approver_type']} Approver",
            f"Handles {rule['approver_type']} approval", 
            f"approver_{i}",
            f"Review content as {rule['approver_type']}: {{processed_content}}"
        )
        approver_id = builder.add_node(approver)
        builder.connect(condition_id, approver_id, graphbit.PyWorkflowEdge.conditional("passed"))
        
        previous_node_id = approver_id
    
    return builder.build()

# Example usage with dynamic rules
approval_rules = [
    {
        "criteria": "content_sensitivity",
        "condition_expression": "sensitivity_level <= 'medium'",
        "approver_type": "editor"
    },
    {
        "criteria": "content_complexity", 
        "condition_expression": "complexity_score >= 7",
        "approver_type": "senior_editor"
    },
    {
        "criteria": "business_impact",
        "condition_expression": "business_impact == 'high'", 
        "approver_type": "manager"
    }
]

adaptive_workflow = create_adaptive_approval_workflow(approval_rules)
```

### Conditional Flow Diagram
```
[Content] → [Analyzer] → [Quality Gate]
                              ↓         ↓
                     (failed) ↓         ↓ (passed)
                              ↓         ↓
                         [Improver]  [Approver]
```

---

## 🌐 Parallel Processing

Process multiple things simultaneously for better performance:

### Social Media Campaign
```python
def create_social_campaign():
    builder = graphbit.PyWorkflowBuilder("Social Media Campaign")
    
    # Campaign strategy (runs first)
    strategist = graphbit.PyWorkflowNode.agent_node(
        "Campaign Strategist",
        "Creates campaign strategy",
        "strategist",
        "Create a social media campaign strategy for: {product} targeting {audience}"
    )
    
    # Platform-specific content (runs in parallel)
    twitter_writer = graphbit.PyWorkflowNode.agent_node(
        "Twitter Writer",
        "Writes Twitter content",
        "twitter",
        "Create 5 Twitter posts for this campaign: {strategy}. Keep under 280 characters each."
    )
    
    linkedin_writer = graphbit.PyWorkflowNode.agent_node(
        "LinkedIn Writer",
        "Writes LinkedIn content",
        "linkedin", 
        "Create a professional LinkedIn post for this campaign: {strategy}."
    )
    
    instagram_writer = graphbit.PyWorkflowNode.agent_node(
        "Instagram Writer",
        "Writes Instagram content",
        "instagram",
        "Create Instagram captions and hashtags for this campaign: {strategy}."
    )
    
    # Content review (waits for all platforms)
    reviewer = graphbit.PyWorkflowNode.agent_node(
        "Content Reviewer",
        "Reviews all content for consistency",
        "reviewer",
        "Review and ensure consistency across all social media content: {all_content}"
    )
    
    # Build fan-out/fan-in pattern
    strat_id = builder.add_node(strategist)
    tw_id = builder.add_node(twitter_writer)
    li_id = builder.add_node(linkedin_writer)
    ig_id = builder.add_node(instagram_writer)
    rev_id = builder.add_node(reviewer)
    
    # Fan-out: strategy feeds all platforms
    builder.connect(strat_id, tw_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(strat_id, li_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(strat_id, ig_id, graphbit.PyWorkflowEdge.data_flow())
    
    # Fan-in: all platforms feed reviewer
    builder.connect(tw_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(li_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(ig_id, rev_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Execute parallel workflow
workflow = create_social_campaign()
executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("product", "GraphBit workflow automation")
executor.set_variable("audience", "software developers")

context = executor.execute(workflow)
```

### Parallel Flow Diagram
```
[Strategy] 
     ↓
   ┌─────┬─────┬─────┐
   ↓     ↓     ↓     ↓
[Twitter][LinkedIn][Instagram]
   ↓     ↓     ↓     ↓  
   └─────┴─────┴─────┘
           ↓
      [Reviewer]
```

---

## ⚡ Async Execution

Handle multiple workflows concurrently:

### Batch Processing
```python
import asyncio

async def process_batch():
    # Configure executor
    executor = graphbit.PyWorkflowExecutor(config)
    
    # Prepare multiple workflows
    topics = [
        "AI in healthcare",
        "Sustainable technology", 
        "Future of remote work",
        "Cybersecurity trends",
        "Green energy solutions"
    ]
    
    # Execute all workflows concurrently
    tasks = []
    for i, topic in enumerate(topics):
        executor.set_variable("topic", topic)
        executor.set_variable("batch_id", str(i))
        
        task = executor.execute_async(workflow)
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # Report results
    completed = sum(1 for r in results if r.is_completed())
    print(f"✅ Completed {completed}/{len(results)} workflows")
    
    return results

# Run batch processing
workflow = create_content_pipeline()  # Use our earlier workflow
results = asyncio.run(process_batch())
```

---

## 🏭 Workflow Factory Pattern

For production applications, use factory patterns to create workflows programmatically:

```python
class WorkflowFactory:
    """Factory for creating workflows based on runtime parameters"""
    
    @staticmethod
    def create_content_workflow(content_type: str, complexity: int, deadline: bool) -> graphbit.PyWorkflow:
        if content_type == "blog":
            return WorkflowFactory._create_blog_workflow(complexity, deadline)
        elif content_type == "social":
            return WorkflowFactory._create_social_workflow(complexity, deadline)
        elif content_type == "email":
            return WorkflowFactory._create_email_workflow(complexity, deadline)
        else:
            raise ValueError(f"Unknown content type: {content_type}")
    
    @staticmethod
    def _create_blog_workflow(complexity: int, has_deadline: bool) -> graphbit.PyWorkflow:
        builder = graphbit.PyWorkflowBuilder(f"Blog-C{complexity}")
        
        # Base nodes
        researcher = graphbit.PyWorkflowNode.agent_node(
            "Researcher", "Researches blog topics", "research",
            "Research comprehensive information for blog: {topic}"
        )
        researcher_id = builder.add_node(researcher)
        previous_id = researcher_id
        
        # Add complexity-based nodes
        if complexity >= 2:
            outliner = graphbit.PyWorkflowNode.agent_node(
                "Outliner", "Creates blog outline", "outline",
                "Create detailed blog outline: {research}"
            )
            outliner_id = builder.add_node(outliner)
            builder.connect(previous_id, outliner_id, graphbit.PyWorkflowEdge.data_flow())
            previous_id = outliner_id
        
        writer = graphbit.PyWorkflowNode.agent_node(
            "Blog Writer", "Writes blog content", "writer",
            "Write engaging blog post: {input}"
        )
        writer_id = builder.add_node(writer)
        builder.connect(previous_id, writer_id, graphbit.PyWorkflowEdge.data_flow())
        previous_id = writer_id
        
        # Skip editing if deadline is tight
        if not has_deadline and complexity >= 2:
            editor = graphbit.PyWorkflowNode.agent_node(
                "Editor", "Edits blog content", "editor",
                "Edit and polish blog post: {content}"
            )
            editor_id = builder.add_node(editor)
            builder.connect(previous_id, editor_id, graphbit.PyWorkflowEdge.data_flow())
        
        return builder.build()

# Usage examples
urgent_blog = WorkflowFactory.create_content_workflow("blog", 1, True)
quality_blog = WorkflowFactory.create_content_workflow("blog", 3, False)
social_post = WorkflowFactory.create_content_workflow("social", 2, False)
```

---

## 🔧 Performance & Configuration

Optimize your workflows for production:

### High-Performance Setup
```python
# Configure executor for high performance
config = graphbit.PyLlmConfig.openai(api_key, "gpt-4")
executor = graphbit.PyWorkflowExecutor(config)

# Optimize memory pools
pool_config = graphbit.PyPoolConfig(
    max_size=50,   # Maximum pool size
    min_size=10    # Minimum pool size
)
executor.configure_pools(pool_config)

# Configure retries
retry_config = graphbit.PyRetryConfig(
    max_attempts=3,    # Retry failed operations
    backoff_ms=1000    # Wait between retries
)
executor.configure_retries(retry_config)

# Set timeouts
executor.set_max_execution_time(300000)  # 5 minutes max

# Monitor performance
context = executor.execute(workflow)
stats = executor.get_performance_stats()

print(f"Success rate: {stats.success_rate():.2%}")
print(f"Average execution time: {stats.avg_execution_time_ms()}ms")
```

### Error Handling
```python
def robust_workflow_execution(workflow, variables):
    try:
        # Set up executor with retries
        executor = graphbit.PyWorkflowExecutor(config)
        retry_config = graphbit.PyRetryConfig(max_attempts=3, backoff_ms=1000)
        executor.configure_retries(retry_config)
        
        # Set variables
        for key, value in variables.items():
            executor.set_variable(key, value)
        
        # Execute with timeout
        context = executor.execute(workflow)
        
        if context.is_completed():
            print("✅ Workflow completed successfully")
            return context
        else:
            print("❌ Workflow failed")
            # Handle failure cases
            return None
            
    except Exception as e:
        print(f"🚨 Error executing workflow: {e}")
        return None

# Use robust execution
result = robust_workflow_execution(
    workflow, 
    {"topic": "AI in healthcare"}
)
```

---

## 🎯 Common Patterns

### 1. Linear Pipeline
```python
# A → B → C → D
builder.connect(a_id, b_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(b_id, c_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(c_id, d_id, graphbit.PyWorkflowEdge.data_flow())
```

### 2. Fan-Out (Parallel Processing)
```python
# A → B, C, D (B, C, D run in parallel)
builder.connect(a_id, b_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(a_id, c_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(a_id, d_id, graphbit.PyWorkflowEdge.data_flow())
```

### 3. Fan-In (Merge Results)
```python
# B, C, D → A (A waits for B, C, D to complete)
builder.connect(b_id, a_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(c_id, a_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(d_id, a_id, graphbit.PyWorkflowEdge.data_flow())
```

### 4. Conditional Branching
```python
# A → B → (C if condition, D if not)
builder.connect(a_id, b_id, graphbit.PyWorkflowEdge.data_flow())
builder.connect(b_id, c_id, graphbit.PyWorkflowEdge.conditional("passed"))
builder.connect(b_id, d_id, graphbit.PyWorkflowEdge.conditional("failed"))
```

---

## 🚀 Next Steps

You now have the foundation to build powerful AI workflows! Here's what to explore next:

### Immediate Next Steps
1. **[Explore Real-World Examples](02-use-case-examples.md)** - See complete applications
2. **[Document Processing](04-document-processing-guide.md)** - Work with files and documents
3. **[Local LLMs](03-local-llm-integration.md)** - Use Ollama for privacy

### Advanced Topics
4. **[Complete API Reference](05-complete-api-reference.md)** - Full documentation
5. **[API Quick Reference](06-api-quick-reference.md)** - Python API cheat sheet

### Practice Ideas
- Build a news summarization pipeline
- Create a code review workflow
- Design a customer support automation
- Develop a content moderation system

---

## 💡 Tips for Success

1. **Start Simple**: Begin with single-agent workflows, then add complexity
2. **Use Static for Predictable Cases**: Choose static workflows for well-defined processes
3. **Use Dynamic for Adaptive Cases**: Choose dynamic workflows when structure needs to vary
4. **Test Iteratively**: Test each node before building complex pipelines
5. **Use Descriptive Names**: Clear node names make workflows easier to debug
6. **Handle Errors**: Always configure retries and error handling
7. **Monitor Performance**: Use performance stats to optimize workflows

**Ready to build something amazing?** The next guide will show you real-world applications! 🚀 
