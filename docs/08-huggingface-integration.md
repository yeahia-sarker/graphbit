# HuggingFace Models Integration Guide

This guide shows you how to integrate HuggingFace models into your GraphBit workflows, giving you access to thousands of open-source AI models.

## Overview

HuggingFace offers one of the largest collections of open-source AI models. GraphBit now supports HuggingFace models directly, allowing you to:

- **Access thousands of models** - From conversational AI to code generation
- **Cost-effective AI** - Many models are free or low-cost compared to proprietary APIs
- **Privacy-friendly** - Choose models that run on your infrastructure
- **Specialized models** - Find models fine-tuned for specific tasks

---

## Quick Start

### 1. Get Your API Token

1. Sign up at [HuggingFace](https://huggingface.co/)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with `Read` permissions

```bash
export HUGGINGFACE_API_KEY="hf_your_token_here"
```

### 2. Your First HuggingFace Workflow

```python
import graphbit
import os

# Initialize GraphBit
graphbit.init()

# Configure HuggingFace provider
config = graphbit.PyLlmConfig.huggingface(
    os.getenv("HUGGINGFACE_API_KEY"),
    "microsoft/DialoGPT-medium"  # Conversational AI model
)

# Create a simple chat workflow
builder = graphbit.PyWorkflowBuilder("HuggingFace Chat")
builder.description("Chatbot powered by HuggingFace")

chatbot = graphbit.PyWorkflowNode.agent_node(
    name="Chatbot",
    description="Friendly conversational agent",
    agent_id="hf_chatbot",
    prompt="You are a helpful assistant. Respond to: {user_message}"
)

# Build and execute
chat_id = builder.add_node(chatbot)
workflow = builder.build()

executor = graphbit.PyWorkflowExecutor(config)
executor.set_variable("user_message", "Hello! How are you?")

context = executor.execute(workflow)
print(f"Response: {context.get_variable('response')}")
```

---

## Popular HuggingFace Models

### Conversational AI
Best for: Chatbots, customer service, interactive applications

```python
# Microsoft DialoGPT - General conversation
config = graphbit.PyLlmConfig.huggingface(api_key, "microsoft/DialoGPT-medium")

# Facebook Blenderbot - Open-domain chatbot
config = graphbit.PyLlmConfig.huggingface(api_key, "facebook/blenderbot-400M-distill")

# Empathetic chatbot
config = graphbit.PyLlmConfig.huggingface(api_key, "facebook/blenderbot_small-90M")
```

### Text Generation
Best for: Content creation, creative writing, general text tasks

```python
# GPT-2 - Classic text generation
config = graphbit.PyLlmConfig.huggingface(api_key, "gpt2")

# GPT-2 Medium - Better quality
config = graphbit.PyLlmConfig.huggingface(api_key, "gpt2-medium")

# DistilGPT-2 - Faster, smaller model
config = graphbit.PyLlmConfig.huggingface(api_key, "distilgpt2")
```

### Code Generation
Best for: Programming assistance, code completion, technical workflows

```python
# Salesforce CodeGen - Code generation
config = graphbit.PyLlmConfig.huggingface(api_key, "Salesforce/codegen-350M-mono")

# Microsoft CodeBERT - Code understanding
config = graphbit.PyLlmConfig.huggingface(api_key, "microsoft/codebert-base")

# CodeT5 - Code-to-text and text-to-code
config = graphbit.PyLlmConfig.huggingface(api_key, "Salesforce/codet5-small")
```

### Specialized Models
Best for: Domain-specific tasks

```python
# BART for summarization
config = graphbit.PyLlmConfig.huggingface(api_key, "facebook/bart-large-cnn")

# RoBERTa for sentiment analysis
config = graphbit.PyLlmConfig.huggingface(api_key, "cardiffnlp/twitter-roberta-base-sentiment-latest")

# SciBERT for scientific text
config = graphbit.PyLlmConfig.huggingface(api_key, "allenai/scibert_scivocab_uncased")
```

---

## Complete Examples

### Multi-Model Content Pipeline

Use different HuggingFace models for different parts of your pipeline:

```python
import graphbit
import os

def create_content_analysis_pipeline():
    """Create a pipeline using multiple HuggingFace models"""
    
    # Main workflow builder
    builder = graphbit.PyWorkflowBuilder("Multi-Model Content Analysis")
    builder.description("Analyze content with specialized HuggingFace models")
    
    # Content generator using GPT-2
    content_generator = graphbit.PyWorkflowNode.agent_node(
        name="Content Generator",
        description="Generates content using GPT-2",
        agent_id="content_gen",
        prompt="Generate a blog post about: {topic}"
    )
    
    # Sentiment analyzer using RoBERTa
    sentiment_analyzer = graphbit.PyWorkflowNode.agent_node(
        name="Sentiment Analyzer", 
        description="Analyzes sentiment using RoBERTa",
        agent_id="sentiment_analyzer",
        prompt="Analyze the sentiment of this content: {content}"
    )
    
    # Summarizer using BART
    summarizer = graphbit.PyWorkflowNode.agent_node(
        name="Summarizer",
        description="Summarizes content using BART",
        agent_id="summarizer", 
        prompt="Summarize this content: {content}"
    )
    
    # Result aggregator
    aggregator = graphbit.PyWorkflowNode.transform_node(
        name="Result Aggregator",
        description="Combines all analysis results",
        transformation="aggregate_analysis_results"
    )
    
    # Build workflow graph
    gen_id = builder.add_node(content_generator)
    sent_id = builder.add_node(sentiment_analyzer)
    summ_id = builder.add_node(summarizer)
    agg_id = builder.add_node(aggregator)
    
    # Connect nodes - parallel analysis after generation
    builder.connect(gen_id, sent_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(gen_id, summ_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(sent_id, agg_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(summ_id, agg_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Execute with different models
def run_multi_model_analysis():
    # Different configs for different tasks
    gpt2_config = graphbit.PyLlmConfig.huggingface(
        os.getenv("HUGGINGFACE_API_KEY"), 
        "gpt2"
    )
    
    workflow = create_content_analysis_pipeline()
    executor = graphbit.PyWorkflowExecutor(gpt2_config)
    
    # Set input
    executor.set_variable("topic", "artificial intelligence trends")
    
    # Execute
    context = executor.execute(workflow)
    
    if context.is_completed():
        print("✅ Multi-model analysis completed!")
        print(f"Generated content: {context.get_variable('content')}")
        print(f"Sentiment: {context.get_variable('sentiment')}")
        print(f"Summary: {context.get_variable('summary')}")
    
    return context

# Run the example
if __name__ == "__main__":
    graphbit.init()
    result = run_multi_model_analysis()
```

### Conversational Agent Workflow

Create an intelligent chatbot using HuggingFace conversational models:

```python
def create_chatbot_workflow():
    """Advanced chatbot with context and personality"""
    
    builder = graphbit.PyWorkflowBuilder("HuggingFace Chatbot")
    builder.description("Intelligent chatbot with conversation memory")
    
    # Context analyzer - understands conversation context
    context_analyzer = graphbit.PyWorkflowNode.agent_node(
        name="Context Analyzer",
        description="Analyzes conversation context and intent",
        agent_id="context_analyzer",
        prompt="""
        Analyze this conversation for context and user intent:
        
        Conversation history: {conversation_history}
        Latest message: {user_message}
        
        Determine:
        1. User's intent
        2. Emotional tone
        3. Required response type
        """
    )
    
    # Response generator using DialoGPT
    response_generator = graphbit.PyWorkflowNode.agent_node(
        name="Response Generator",
        description="Generates contextual responses",
        agent_id="response_gen",
        prompt="""
        Generate a helpful and engaging response:
        
        Context: {context_analysis}
        User message: {user_message}
        Conversation history: {conversation_history}
        
        Response should be:
        - Contextually appropriate
        - Helpful and informative
        - Engaging and natural
        """
    )
    
    # Response refiner
    response_refiner = graphbit.PyWorkflowNode.agent_node(
        name="Response Refiner",
        description="Refines and polishes responses",
        agent_id="refiner",
        prompt="""
        Refine this chatbot response for clarity and engagement:
        
        Original response: {raw_response}
        Context: {context_analysis}
        
        Make it more natural and conversational.
        """
    )
    
    # Build workflow
    context_id = builder.add_node(context_analyzer)
    response_id = builder.add_node(response_generator)
    refiner_id = builder.add_node(response_refiner)
    
    # Sequential processing
    builder.connect(context_id, response_id, graphbit.PyWorkflowEdge.data_flow())
    builder.connect(response_id, refiner_id, graphbit.PyWorkflowEdge.data_flow())
    
    return builder.build()

# Chatbot execution class
class HuggingFaceChatbot:
    def __init__(self):
        graphbit.init()
        
        # Use DialoGPT for conversation
        self.config = graphbit.PyLlmConfig.huggingface(
            os.getenv("HUGGINGFACE_API_KEY"),
            "microsoft/DialoGPT-medium"
        )
        
        self.workflow = create_chatbot_workflow()
        self.executor = graphbit.PyWorkflowExecutor(self.config)
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response"""
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_message}")
        
        # Set workflow variables
        self.executor.set_variable("user_message", user_message)
        self.executor.set_variable("conversation_history", "\n".join(self.conversation_history[-10:]))  # Keep last 10 messages
        
        # Execute workflow
        context = self.executor.execute(self.workflow)
        
        if context.is_completed():
            response = context.get_variable("refined_response")
            self.conversation_history.append(f"Assistant: {response}")
            return response
        else:
            return "Sorry, I encountered an error. Please try again."
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

# Usage example
def run_chatbot_demo():
    chatbot = HuggingFaceChatbot()
    
    print("🤖 HuggingFace Chatbot Ready!")
    print("Type 'quit' to exit, 'reset' to clear history")
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            print("🔄 Conversation history cleared!")
            continue
        
        response = chatbot.chat(user_input)
        print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    run_chatbot_demo()
```

---

## Performance Optimization

### Model Selection Guidelines

#### For Speed (Low Latency)
```python
# Smaller, faster models
fast_configs = [
    graphbit.PyLlmConfig.huggingface(api_key, "distilgpt2"),           # 82M params
    graphbit.PyLlmConfig.huggingface(api_key, "microsoft/DialoGPT-small"),  # 117M params
    graphbit.PyLlmConfig.huggingface(api_key, "facebook/blenderbot_small-90M"),  # 90M params
]
```

#### For Quality (Better Responses)
```python
# Larger, higher-quality models  
quality_configs = [
    graphbit.PyLlmConfig.huggingface(api_key, "gpt2-medium"),         # 355M params
    graphbit.PyLlmConfig.huggingface(api_key, "microsoft/DialoGPT-medium"), # 345M params
    graphbit.PyLlmConfig.huggingface(api_key, "facebook/blenderbot-400M-distill"), # 400M params
]
```

### High-Performance Configuration

```python
# Optimize executor for HuggingFace models
def create_optimized_executor(config):
    executor = graphbit.PyWorkflowExecutor.new_high_throughput(config)
    
    # Increase timeout for larger models
    executor = executor.with_max_node_execution_time(60000)  # 60 seconds
    
    # Enable memory pools for better performance
    executor = executor.enable_memory_pools()
    
    # Configure retries for reliability
    retry_config = graphbit.PyRetryConfig.default()
    retry_config = retry_config.with_exponential_backoff(1000, 2.0, 10000)
    executor = executor.with_retry_config(retry_config)
    
    return executor

# Usage
config = graphbit.PyLlmConfig.huggingface(api_key, "microsoft/DialoGPT-medium")
executor = create_optimized_executor(config)
```

---

## Best Practices

### 1. Model Selection Strategy

```python
def select_model_for_task(task_type: str, performance_priority: str = "balanced"):
    """Smart model selection based on task and performance needs"""
    
    models = {
        "conversation": {
            "fast": "microsoft/DialoGPT-small",
            "balanced": "microsoft/DialoGPT-medium", 
            "quality": "facebook/blenderbot-400M-distill"
        },
        "text_generation": {
            "fast": "distilgpt2",
            "balanced": "gpt2",
            "quality": "gpt2-medium"
        },
        "code_generation": {
            "fast": "Salesforce/codegen-350M-mono",
            "balanced": "microsoft/codebert-base",
            "quality": "Salesforce/codet5-base"
        }
    }
    
    return models.get(task_type, {}).get(performance_priority, "gpt2")

# Usage
model = select_model_for_task("conversation", "quality")
config = graphbit.PyLlmConfig.huggingface(api_key, model)
```

### 2. Error Handling and Fallbacks

```python
def create_robust_hf_executor():
    """Create executor with fallback strategy"""
    
    # Primary: High-quality model
    primary_config = graphbit.PyLlmConfig.huggingface(
        os.getenv("HUGGINGFACE_API_KEY"),
        "microsoft/DialoGPT-medium"
    )
    
    # Fallback: Faster, more reliable model
    fallback_config = graphbit.PyLlmConfig.huggingface(
        os.getenv("HUGGINGFACE_API_KEY"), 
        "distilgpt2"
    )
    
    executor = graphbit.PyWorkflowExecutor(primary_config)
    
    # Configure circuit breaker for automatic fallback
    circuit_breaker = graphbit.PyCircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_ms=60000
    )
    executor = executor.with_circuit_breaker_config(circuit_breaker)
    
    return executor, fallback_config
```

### 3. Cost and Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_minute: int):
    """Decorator for rate limiting HuggingFace API calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Usage
class RateLimitedExecutor:
    def __init__(self, config):
        self.executor = graphbit.PyWorkflowExecutor(config)
    
    @rate_limit(calls_per_minute=60)  # Limit to 60 calls per minute
    def execute_workflow(self, workflow):
        return self.executor.execute(workflow)
```

---

## Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Check if model exists
def verify_model_availability(model_name: str) -> bool:
    """Verify a HuggingFace model is available"""
    try:
        import requests
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url)
        return response.status_code == 200
    except:
        return False

# Usage
if not verify_model_availability("microsoft/DialoGPT-medium"):
    print("Model not found, using fallback...")
    model = "gpt2"
else:
    model = "microsoft/DialoGPT-medium"
```

#### API Rate Limits
```python
def handle_rate_limits(executor, workflow, max_retries=3):
    """Handle rate limiting with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return executor.execute(workflow)
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) * 60  # Exponential backoff
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")
```

#### Memory Issues with Large Models
```python
# Use smaller models or optimize memory
def optimize_for_memory():
    config = graphbit.PyLlmConfig.huggingface(
        api_key,
        "distilgpt2"  # Smaller model
    )
    
    # Low-latency executor uses less memory
    executor = graphbit.PyWorkflowExecutor.new_low_latency(config)
    executor = executor.disable_memory_pools()  # Reduce memory usage
    
    return executor
```

---

## Integration with Other Providers

### Hybrid Workflows
Combine HuggingFace with other providers for optimal results:

```python
def create_hybrid_workflow():
    """Use different providers for different strengths"""
    
    # HuggingFace for specialized tasks
    hf_config = graphbit.PyLlmConfig.huggingface(
        os.getenv("HUGGINGFACE_API_KEY"),
        "microsoft/DialoGPT-medium"
    )
    
    # OpenAI for complex reasoning
    openai_config = graphbit.PyLlmConfig.openai(
        os.getenv("OPENAI_API_KEY"),
        "gpt-4"
    )
    
    # Create workflow with mixed providers
    builder = graphbit.PyWorkflowBuilder("Hybrid AI Pipeline")
    
    # HuggingFace for initial conversation
    chat_agent = graphbit.PyWorkflowNode.agent_node(
        "Chat Agent", "Initial conversation", "hf_chat",
        "Have a friendly conversation: {user_input}"
    )
    
    # OpenAI for complex analysis
    analyzer = graphbit.PyWorkflowNode.agent_node(
        "Analyzer", "Deep analysis", "gpt4_analyzer", 
        "Analyze conversation for insights: {conversation}"
    )
    
    # Build workflow...
    return builder.build()
```

---

## Next Steps

- **Explore the Model Hub**: Browse [HuggingFace Model Hub](https://huggingface.co/models) for specialized models
- **Fine-tune Models**: Learn how to fine-tune models for your specific use case
- **Deploy Custom Models**: Host your own models and use them with GraphBit
- **Monitor Performance**: Use GraphBit's built-in performance monitoring

**Learn More:**
- [Complete API Reference](05-complete-api-reference.md#pyllmconfig)
- [Performance Optimization Guide](05-complete-api-reference.md#performance)
- [Multi-Provider Workflows](02-use-case-examples.md)
- [Local LLM Integration](03-local-llm-integration.md)

---

*Ready to explore thousands of open-source AI models? Start building with HuggingFace and GraphBit today!* 🚀 
