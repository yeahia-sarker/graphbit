#!/usr/bin/env python3

"""GraphBit framework benchmark implementation.

Optimized for Rust concurrency and performance.
"""

import os
import sys
import uuid
from typing import Any, Dict

try:
    import graphbit
except ImportError:
    print("GraphBit Python bindings not installed. " "Run 'maturin develop' in graphbit/")
    sys.exit(1)

from .common import (
    COMPLEX_WORKFLOW_STEPS,
    CONCURRENT_TASK_PROMPTS,
    MEMORY_INTENSIVE_PROMPT,
    PARALLEL_TASKS,
    SEQUENTIAL_TASKS,
    SIMPLE_TASK_PROMPT,
    BaseBenchmark,
    BenchmarkMetrics,
    BenchmarkScenario,
    calculate_throughput,
    count_tokens_estimate,
    get_standard_llm_config,
)


class GraphBitBenchmark(BaseBenchmark):
    """GraphBit framework benchmark implementation optimized for Rust concurrency."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GraphBit benchmark with configuration."""
        super().__init__(config)
        self.executors: Dict[str, Any] = {}  # Cache different executor types
        self.llm_config: Any = None
        self.shared_agent_id: str = ""  # Reuse agent across tasks

        if graphbit is None:
            raise ImportError("GraphBit Python bindings not installed. " "Please run 'maturin develop' in the graphbit/ directory.")

    async def setup(self) -> None:
        """Set up GraphBit with optimized executors for different scenarios."""
        # Initialize GraphBit
        graphbit.init()

        # Get standardized configuration
        llm_config = get_standard_llm_config(self.config)
        openai_key = os.getenv("OPENAI_API_KEY") or llm_config["api_key"]
        if not openai_key:
            raise ValueError("OpenAI API key not found in environment or config")

        # Create LLM configuration using the Python binding API
        self.llm_config = graphbit.PyLlmConfig.openai(openai_key, llm_config["model"])

        # Configure optimized retry settings
        retry_config = (
            graphbit.PyRetryConfig(max_attempts=3)
            .with_exponential_backoff(
                initial_delay_ms=50,  # Faster initial retry
                multiplier=1.5,
                max_delay_ms=2000,  # Shorter max delay
            )
            .with_jitter(0.1)
        )

        # Configure circuit breaker with optimized settings
        circuit_breaker_config = graphbit.PyCircuitBreakerConfig(
            failure_threshold=3,  # Fail faster
            recovery_timeout_ms=15000,  # Recover faster
        )

        # Create specialized executors for different workloads

        # High-throughput executor for parallel and concurrent tasks
        self.executors["high_throughput"] = (
            graphbit.PyWorkflowExecutor.new_high_throughput(self.llm_config)
            .with_max_node_execution_time(20000)
            .with_fail_fast(False)
            .with_retry_config(retry_config)
            .with_circuit_breaker_config(circuit_breaker_config)
        )

        # Low-latency executor for simple tasks
        self.executors["low_latency"] = (
            graphbit.PyWorkflowExecutor.new_low_latency(self.llm_config).with_max_node_execution_time(10000).with_fail_fast(True).with_circuit_breaker_config(circuit_breaker_config)
        )

        # Memory-optimized executor for complex workflows
        self.executors["memory_optimized"] = (
            graphbit.PyWorkflowExecutor.new_memory_optimized(self.llm_config)
            .with_max_node_execution_time(60000)
            .with_fail_fast(False)
            .with_retry_config(retry_config)
            .with_circuit_breaker_config(circuit_breaker_config)
        )

        # Default executor for mixed workloads
        self.executors["default"] = (
            graphbit.PyWorkflowExecutor.new_high_throughput(self.llm_config)
            .with_max_node_execution_time(30000)
            .with_fail_fast(False)
            .with_retry_config(retry_config)
            .with_circuit_breaker_config(circuit_breaker_config)
        )

        # Create a shared agent ID to reuse across tasks for better performance
        self.shared_agent_id = str(uuid.uuid4())

    async def teardown(self) -> None:
        """Cleanup GraphBit resources."""
        self.executors.clear()
        self.llm_config = None
        self.shared_agent_id = ""

    async def run_simple_task(self) -> BenchmarkMetrics:
        """Run a simple single-task benchmark using optimized low-latency executor."""
        self.monitor.start_monitoring()

        try:
            # Use low-latency executor for single tasks
            executor = self.executors["low_latency"]

            # Create a simple workflow with one agent node
            builder = graphbit.PyWorkflowBuilder("Simple Task Workflow")
            builder.description("Single task execution benchmark")

            node = graphbit.PyWorkflowNode.agent_node(
                name="Task Executor",
                description="Executes the simple task",
                agent_id=self.shared_agent_id,
                prompt=SIMPLE_TASK_PROMPT,
            )

            builder.add_node(node)
            workflow = builder.build()

            # Validate workflow before execution
            workflow.validate()

            # Execute the workflow
            result = executor.execute(workflow)

            if result.is_failed():
                raise Exception(f"Workflow execution failed: {result.state()}")

            # Extract output
            output_content = self._extract_output_from_context(result, self.shared_agent_id, "Task Executor")

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.SIMPLE_TASK.value,
                task_name="Simple Task",
                output=output_content,
            )

            # Count tokens
            token_count = count_tokens_estimate(SIMPLE_TASK_PROMPT + str(output_content))

        except Exception as e:
            self.logger.error(f"Error in simple task benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = token_count
        metrics.throughput_tasks_per_sec = calculate_throughput(1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_sequential_pipeline(self) -> BenchmarkMetrics:
        """Run a sequential pipeline benchmark using GraphBit."""
        self.monitor.start_monitoring()

        try:
            # Use default executor for sequential workflows
            executor = self.executors["default"]

            # Create a workflow with sequential tasks
            builder = graphbit.PyWorkflowBuilder("Sequential Pipeline Workflow")
            builder.description("Sequential pipeline execution benchmark")

            # Add nodes for each task in sequence
            previous_node_id = None
            node_ids = []

            for i, task in enumerate(SEQUENTIAL_TASKS):
                task_name = f"Task {i+1}"
                node = graphbit.PyWorkflowNode.agent_node(
                    name=task_name,
                    description=f"Executes {task_name}",
                    agent_id=self.shared_agent_id,  # Reuse shared agent
                    prompt=task,
                )
                node_id = builder.add_node(node)
                node_ids.append(node_id)

                # Connect nodes sequentially using data flow edges
                if previous_node_id is not None:
                    edge = graphbit.PyWorkflowEdge.data_flow()
                    builder.connect(previous_node_id, node_id, edge)

                previous_node_id = node_id

            workflow = builder.build()

            # Validate workflow
            workflow.validate()

            # Execute the workflow
            result = executor.execute(workflow)

            if result.is_failed():
                raise Exception(f"Workflow execution failed: {result.state()}")

            # Extract outputs and count tokens
            total_tokens = 0
            variables = result.variables()

            if variables:
                for i, (_key, value) in enumerate(variables):
                    if i < len(SEQUENTIAL_TASKS):
                        total_tokens += count_tokens_estimate(SEQUENTIAL_TASKS[i] + str(value))

                        # Log LLM output to file
                        task_num = i + 1
                        self.log_output(
                            scenario_name=BenchmarkScenario.SEQUENTIAL_PIPELINE.value,
                            task_name=f"Task {task_num}",
                            output=str(value),
                        )
            else:
                # Fallback token counting
                total_tokens = sum(count_tokens_estimate(task) for task in SEQUENTIAL_TASKS)
                for i, _task in enumerate(SEQUENTIAL_TASKS):
                    self.log_output(
                        scenario_name=BenchmarkScenario.SEQUENTIAL_PIPELINE.value,
                        task_name=f"Task {i+1}",
                        output=f"Sequential task {i+1} completed successfully",
                    )

        except Exception as e:
            self.logger.error(f"Error in sequential pipeline benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.throughput_tasks_per_sec = calculate_throughput(len(SEQUENTIAL_TASKS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_parallel_pipeline(self) -> BenchmarkMetrics:
        """Run a parallel pipeline benchmark using high-throughput executor and native parallel execution."""
        self.monitor.start_monitoring()

        try:
            # Use high-throughput executor for parallel tasks
            executor = self.executors["high_throughput"]

            # Create individual workflows for each parallel task for true concurrency
            workflows = []

            for i, task in enumerate(PARALLEL_TASKS):
                builder = graphbit.PyWorkflowBuilder(f"Parallel Task {i+1}")
                builder.description(f"Parallel task {i+1} execution")

                node = graphbit.PyWorkflowNode.agent_node(
                    name=f"Parallel Task {i+1}",
                    description=f"Executes parallel task {i+1}",
                    agent_id=self.shared_agent_id,  # Reuse shared agent
                    prompt=task,
                )

                builder.add_node(node)
                workflow = builder.build()

                # Validate each workflow
                workflow.validate()
                workflows.append(workflow)

            # Execute all workflows concurrently using native Rust concurrency
            results = executor.execute_concurrent(workflows)

            # Process results and count tokens
            total_tokens = 0
            successful_count = 0

            for i, result in enumerate(results):
                if result and result.is_completed() and not result.is_failed():
                    successful_count += 1

                    # Extract output
                    output_content = self._extract_output_from_context(result, self.shared_agent_id, f"Task {i+1}")

                    # Count tokens
                    total_tokens += count_tokens_estimate(PARALLEL_TASKS[i] + str(output_content))

                    # Log LLM output to file
                    self.log_output(
                        scenario_name=BenchmarkScenario.PARALLEL_PIPELINE.value,
                        task_name=f"Task {i+1}",
                        output=output_content,
                    )
                else:
                    # Failed task - count input tokens only
                    total_tokens += count_tokens_estimate(PARALLEL_TASKS[i])
                    self.log_output(
                        scenario_name=BenchmarkScenario.PARALLEL_PIPELINE.value,
                        task_name=f"Task {i+1}",
                        output=f"Parallel task {i+1} failed: {result.state() if result else 'execution error'}",
                    )

            # Calculate error rate
            error_rate = (len(PARALLEL_TASKS) - successful_count) / len(PARALLEL_TASKS) if PARALLEL_TASKS else 0.0

        except Exception as e:
            self.logger.error(f"Error in parallel pipeline benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(PARALLEL_TASKS)
        metrics.error_rate = error_rate
        metrics.throughput_tasks_per_sec = calculate_throughput(len(PARALLEL_TASKS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_complex_workflow(self) -> BenchmarkMetrics:
        """Run a complex workflow benchmark using memory-optimized executor."""
        self.monitor.start_monitoring()

        try:
            # Use memory-optimized executor for complex workflows
            executor = self.executors["memory_optimized"]

            # Create a complex workflow with dependencies
            builder = graphbit.PyWorkflowBuilder("Complex Workflow")
            builder.description("Complex workflow with dependencies benchmark")

            node_mapping = {}

            # Create nodes
            for step in COMPLEX_WORKFLOW_STEPS:
                node = graphbit.PyWorkflowNode.agent_node(
                    name=step["task"],
                    description=f"Executes {step['task']}",
                    agent_id=self.shared_agent_id,  # Reuse shared agent
                    prompt=step["prompt"],
                )

                node_id = builder.add_node(node)
                node_mapping[step["task"]] = node_id

            # Create edges based on dependencies
            for step in COMPLEX_WORKFLOW_STEPS:
                if step["depends_on"]:
                    for dependency in step["depends_on"]:
                        edge = graphbit.PyWorkflowEdge.data_flow()
                        builder.connect(node_mapping[dependency], node_mapping[step["task"]], edge)

            workflow = builder.build()

            # Validate workflow
            workflow.validate()

            # Execute the workflow
            result = executor.execute(workflow)

            if result.is_failed():
                raise Exception(f"Workflow execution failed: {result.state()}")

            # Extract outputs and count tokens
            total_tokens = 0
            variables = result.variables()

            if variables:
                for i, (_key, value) in enumerate(variables):
                    step_name = COMPLEX_WORKFLOW_STEPS[i]["task"] if i < len(COMPLEX_WORKFLOW_STEPS) else f"Step {i+1}"

                    if i < len(COMPLEX_WORKFLOW_STEPS):
                        total_tokens += count_tokens_estimate(COMPLEX_WORKFLOW_STEPS[i]["prompt"] + str(value))

                    # Log LLM output to file
                    self.log_output(
                        scenario_name=BenchmarkScenario.COMPLEX_WORKFLOW.value,
                        task_name=step_name,
                        output=str(value),
                    )
            else:
                # Fallback token counting
                total_tokens = sum(count_tokens_estimate(step["prompt"]) for step in COMPLEX_WORKFLOW_STEPS)
                for step in COMPLEX_WORKFLOW_STEPS:
                    self.log_output(
                        scenario_name=BenchmarkScenario.COMPLEX_WORKFLOW.value,
                        task_name=step["task"],
                        output=f"{step['task']} completed successfully",
                    )

        except Exception as e:
            self.logger.error(f"Error in complex workflow benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.throughput_tasks_per_sec = calculate_throughput(len(COMPLEX_WORKFLOW_STEPS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_memory_intensive(self) -> BenchmarkMetrics:
        """Run a memory-intensive benchmark using memory-optimized executor."""
        self.monitor.start_monitoring()

        try:
            # Use memory-optimized executor specifically designed for this workload
            executor = self.executors["memory_optimized"]

            # Create a workflow with memory-intensive task
            builder = graphbit.PyWorkflowBuilder("Memory Intensive Workflow")
            builder.description("Memory intensive task benchmark")

            node = graphbit.PyWorkflowNode.agent_node(
                name="Memory Intensive Task",
                description="Processes large amounts of data",
                agent_id=self.shared_agent_id,  # Reuse shared agent
                prompt=MEMORY_INTENSIVE_PROMPT,
            )

            builder.add_node(node)
            workflow = builder.build()

            # Validate workflow
            workflow.validate()

            # Execute the workflow with the memory-optimized executor
            result = executor.execute(workflow)

            if result.is_failed():
                error_message = f"Workflow execution failed: {result.state()}"
                print(f"Memory intensive task failed: {error_message}")
                raise Exception(error_message)

            # Extract output
            output_content = self._extract_output_from_context(result, self.shared_agent_id, "Memory Intensive Task")

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.MEMORY_INTENSIVE.value,
                task_name="Memory Intensive Task",
                output=output_content,
            )

            # Count tokens
            token_count = count_tokens_estimate(MEMORY_INTENSIVE_PROMPT + str(output_content))

        except Exception as e:
            print(f"Memory intensive benchmark failed: {str(e)}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = token_count
        metrics.throughput_tasks_per_sec = calculate_throughput(1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_concurrent_tasks(self) -> BenchmarkMetrics:
        """Run concurrent tasks benchmark using native Rust batch execution for maximum performance."""
        self.monitor.start_monitoring()

        try:
            # Use high-throughput executor optimized for concurrent workloads
            executor = self.executors["high_throughput"]

            # Create individual workflows for each concurrent task
            workflows = []

            for i, prompt in enumerate(CONCURRENT_TASK_PROMPTS):
                builder = graphbit.PyWorkflowBuilder(f"Concurrent Task {i+1}")
                builder.description(f"Concurrent task {i+1}")

                node = graphbit.PyWorkflowNode.agent_node(
                    name=f"Task {i+1}",
                    description=f"Executes concurrent task {i+1}",
                    agent_id=self.shared_agent_id,  # Reuse shared agent for efficiency
                    prompt=prompt,
                )

                builder.add_node(node)
                workflow = builder.build()

                # Validate each workflow
                workflow.validate()
                workflows.append(workflow)

            # Execute all workflows concurrently using native Rust batch processing
            # This leverages Rust's tokio runtime and optimized concurrency management
            batch_results = executor.execute_batch(workflows)

            # Process results with improved error handling
            total_tokens = 0
            successful_count = 0

            for i, result in enumerate(batch_results):
                if result and result.is_completed() and not result.is_failed():
                    successful_count += 1

                    # Extract output using improved method
                    output_content = self._extract_output_from_context(result, self.shared_agent_id, f"Task {i+1}")

                    # Count tokens for both input and output
                    input_tokens = count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                    output_tokens = count_tokens_estimate(output_content)
                    total_tokens += input_tokens + output_tokens

                    # Log LLM output to file
                    self.log_output(
                        scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                        task_name=f"Task {i+1}",
                        output=output_content,
                    )
                else:
                    # Failed task - only count input tokens
                    total_tokens += count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                    self.log_output(
                        scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                        task_name=f"Task {i+1}",
                        output=f"Task {i+1} failed: {result.state() if result else 'batch execution error'}",
                    )

            # Calculate error rate
            error_rate = (len(batch_results) - successful_count) / len(batch_results) if batch_results else 1.0

        except Exception as e:
            print(f"Concurrent tasks benchmark failed: {str(e)}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(CONCURRENT_TASK_PROMPTS)
        metrics.error_rate = error_rate
        metrics.throughput_tasks_per_sec = calculate_throughput(len(CONCURRENT_TASK_PROMPTS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_high_performance_concurrent_tasks(self) -> BenchmarkMetrics:
        """Demonstrate ultimate concurrent performance using native Rust agent task execution.

        This method shows how to use the lowest-level concurrent task execution for maximum throughput.
        """
        self.monitor.start_monitoring()

        try:
            # Use high-throughput executor with maximum concurrency settings
            executor = self.executors["high_throughput"]

            # Option 1: Use native concurrent agent tasks (available in Python bindings)
            # This is the most efficient approach as it bypasses workflow overhead
            try:
                # Execute concurrent agent tasks directly - bypasses workflow overhead
                concurrent_results = executor.execute_concurrent_agent_tasks(CONCURRENT_TASK_PROMPTS, self.shared_agent_id)

                # Process native concurrent results
                total_tokens = 0
                successful_count = 0

                for i, result in enumerate(concurrent_results):
                    if not result.startswith("ERROR:"):  # Successful result
                        successful_count += 1
                        output_content = result

                        # Count tokens
                        input_tokens = count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                        output_tokens = count_tokens_estimate(output_content)
                        total_tokens += input_tokens + output_tokens

                        # Log output
                        self.log_output(
                            scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                            task_name=f"Native Task {i+1}",
                            output=output_content,
                        )
                    else:  # Failed task
                        # Failed task
                        total_tokens += count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                        self.log_output(
                            scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                            task_name=f"Native Task {i+1}",
                            output=result,
                        )

                print("✅ Used native Rust concurrent agent task execution for maximum performance")

            except Exception as e:
                # Fallback to workflow-based batch execution if native method unavailable
                print(f"⚠️  Native concurrent agent tasks not available ({e}), using workflow batch execution")

                # Create minimal workflows for batch execution
                workflows = []
                for i, prompt in enumerate(CONCURRENT_TASK_PROMPTS):
                    builder = graphbit.PyWorkflowBuilder(f"HiPerf Task {i+1}")

                    node = graphbit.PyWorkflowNode.agent_node(
                        name=f"HiPerf Task {i+1}",
                        description="High-performance concurrent task",
                        agent_id=self.shared_agent_id,
                        prompt=prompt,
                    )

                    builder.add_node(node)
                    workflow = builder.build()
                    workflow.validate()
                    workflows.append(workflow)

                # Execute with maximum concurrency
                batch_results = executor.execute_batch(workflows)

                # Process batch results
                total_tokens = 0
                successful_count = 0

                for i, result in enumerate(batch_results):
                    if result and result.is_completed() and not result.is_failed():
                        successful_count += 1
                        output_content = self._extract_output_from_context(result, self.shared_agent_id, f"HiPerf Task {i+1}")

                        # Count tokens
                        input_tokens = count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                        output_tokens = count_tokens_estimate(output_content)
                        total_tokens += input_tokens + output_tokens

                        # Log output
                        self.log_output(
                            scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                            task_name=f"HiPerf Task {i+1}",
                            output=output_content,
                        )
                    else:
                        # Failed task
                        total_tokens += count_tokens_estimate(CONCURRENT_TASK_PROMPTS[i])
                        self.log_output(
                            scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                            task_name=f"HiPerf Task {i+1}",
                            output=f"HiPerf task {i+1} failed: {result.state() if result else 'batch error'}",
                        )

            # Calculate error rate
            error_rate = (len(CONCURRENT_TASK_PROMPTS) - successful_count) / len(CONCURRENT_TASK_PROMPTS) if CONCURRENT_TASK_PROMPTS else 1.0

        except Exception as e:
            print(f"High-performance concurrent tasks benchmark failed: {str(e)}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(CONCURRENT_TASK_PROMPTS)
        metrics.error_rate = error_rate
        metrics.throughput_tasks_per_sec = calculate_throughput(len(CONCURRENT_TASK_PROMPTS), metrics.execution_time_ms / 1000)

        return metrics

    def _extract_output_from_context(self, context: Any, agent_id: str = "", fallback_name: str = "Task") -> str:
        """
        Improved output extraction from workflow context.

        Args:
            context: PyWorkflowContext object
            agent_id: Optional agent ID to try as a key
            fallback_name: Fallback name for output if extraction fails

        Returns:
            str: Extracted output content
        """
        import json

        variables = context.variables()
        output_content = ""

        if variables:
            # GraphBit uses node_result_X keys, not agent IDs
            for _key, value in variables:
                if value and str(value).strip():
                    value_str = str(value).strip()

                    # Skip explicit null values
                    if value_str in ["null", "None", '"null"', '"None"']:
                        continue

                    # Handle JSON-encoded strings (GraphBit returns "\"Hello World\"")
                    import contextlib

                    with contextlib.suppress(json.JSONDecodeError):
                        if value_str.startswith('"') and value_str.endswith('"'):
                            decoded_value = json.loads(value_str)
                            if decoded_value and str(decoded_value).strip() != "null":
                                output_content = str(decoded_value)
                                break

                    # Use raw value if JSON decoding fails but it's not null
                    if value_str != "null":
                        output_content = value_str
                        break

        # Final fallback for failed/null outputs
        if not output_content or output_content.strip() == "" or output_content in ["null", "None"]:
            output_content = f"{fallback_name} completed successfully, but no detailed output was captured."

        return output_content
