"""LlamaIndex framework benchmark implementation."""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI

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


class LlamaIndexBenchmark(BaseBenchmark):
    """LlamaIndex framework benchmark implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LlamaIndex benchmark with configuration."""
        super().__init__(config)
        self.llm: Optional[Any] = None
        self.index: Optional[Any] = None
        self.query_engine: Optional[Any] = None
        self.chat_engine: Optional[Any] = None

    async def setup(self) -> None:
        """Set up LlamaIndex for benchmarking."""
        # Get standardized configuration
        llm_config = get_standard_llm_config(self.config)
        api_key = os.getenv("OPENAI_API_KEY") or llm_config["api_key"]
        if not api_key:
            raise ValueError("OpenAI API key not found in environment or config")

        # Initialize LlamaIndex LLM
        self.llm = OpenAI(
            model=llm_config["model"],
            api_key=api_key,
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )

        # Set global settings
        Settings.llm = self.llm

        # Create a simple index with some dummy documents for query-based tasks
        documents = [
            Document(text="This is a sample document for benchmarking purposes."),
            Document(text="Another document with different content to test retrieval."),
            Document(text="Complex workflow tasks require sophisticated reasoning capabilities."),
        ]

        # Create vector store index
        self.index = VectorStoreIndex.from_documents(documents)

        # Create query and chat engines
        self.query_engine = self.index.as_query_engine()
        self.chat_engine = self.index.as_chat_engine()

    async def teardown(self) -> None:
        """Cleanup LlamaIndex resources."""
        self.llm = None
        self.index = None
        self.query_engine = None
        self.chat_engine = None

    async def run_simple_task(self) -> BenchmarkMetrics:
        """Run a simple single-task benchmark using LlamaIndex."""
        self.monitor.start_monitoring()

        try:
            # Execute simple task using LlamaIndex LLM directly
            if self.llm is None:
                raise ValueError("LLM not initialized")
            response = await self.llm.acomplete(SIMPLE_TASK_PROMPT)
            result = str(response)

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.SIMPLE_TASK.value,
                task_name="Simple Task",
                output=result,
            )

            # Count tokens
            token_count = count_tokens_estimate(SIMPLE_TASK_PROMPT + result)

        except Exception as e:
            logging.error(f"Error in simple task: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = token_count
        metrics.throughput_tasks_per_sec = calculate_throughput(1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_sequential_pipeline(self) -> BenchmarkMetrics:
        """Run a sequential pipeline benchmark using LlamaIndex."""
        self.monitor.start_monitoring()

        try:
            previous_result = ""
            total_tokens = 0

            # Execute tasks sequentially
            for i, task in enumerate(SEQUENTIAL_TASKS):
                if i == 0:
                    # First task doesn't have previous result
                    prompt = task
                else:
                    prompt = f"Previous result: {previous_result}\n\nNew task: {task}"

                if self.llm is None:
                    raise ValueError("LLM not initialized")
                response = await self.llm.acomplete(prompt)
                result = str(response)
                previous_result = result
                total_tokens += count_tokens_estimate(prompt + result)

                # Log LLM output to file
                self.log_output(
                    scenario_name=BenchmarkScenario.SEQUENTIAL_PIPELINE.value,
                    task_name=f"Task {i+1}",
                    output=result,
                )

        except Exception as e:
            logging.error(f"Error in sequential pipeline: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.throughput_tasks_per_sec = calculate_throughput(len(SEQUENTIAL_TASKS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_parallel_pipeline(self) -> BenchmarkMetrics:
        """Run a parallel pipeline benchmark using LlamaIndex."""
        self.monitor.start_monitoring()

        try:
            # Create coroutines for parallel execution
            if self.llm is None:
                raise ValueError("LLM not initialized")
            tasks = [self.llm.acomplete(task) for task in PARALLEL_TASKS]

            # Execute tasks in parallel
            responses = await asyncio.gather(*tasks)
            results = [str(response) for response in responses]

            # Count tokens
            total_tokens = sum(count_tokens_estimate(task + result) for task, result in zip(PARALLEL_TASKS, results))

            # Log outputs
            for i, result in enumerate(results):
                self.log_output(
                    scenario_name=BenchmarkScenario.PARALLEL_PIPELINE.value,
                    task_name=f"Parallel Task {i+1}",
                    output=result,
                )

        except Exception as e:
            logging.error(f"Error in parallel pipeline: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(PARALLEL_TASKS)
        metrics.throughput_tasks_per_sec = calculate_throughput(len(PARALLEL_TASKS), metrics.execution_time_ms / 1000)

        return metrics

    async def run_complex_workflow(self) -> BenchmarkMetrics:
        """Run a complex workflow benchmark using LlamaIndex query engine."""
        self.monitor.start_monitoring()

        try:
            total_tokens = 0
            results = []

            # Execute complex workflow steps using query engine
            for i, step in enumerate(COMPLEX_WORKFLOW_STEPS):
                # Use query engine for more complex reasoning
                if self.query_engine is None:
                    raise ValueError("Query engine not initialized")
                response = self.query_engine.query(step["prompt"])
                result = str(response)
                results.append(result)
                total_tokens += count_tokens_estimate(step["prompt"] + result)

                # Log LLM output to file
                self.log_output(
                    scenario_name=BenchmarkScenario.COMPLEX_WORKFLOW.value,
                    task_name=f"Workflow Step {i+1}",
                    output=result,
                )

            # Final synthesis step
            synthesis_prompt = "Synthesize the following results into a comprehensive conclusion:\n\n"
            synthesis_prompt += "\n\n".join([f"Step {i+1}: {result}" for i, result in enumerate(results)])

            if self.llm is None:
                raise ValueError("LLM not initialized")
            final_response = await self.llm.acomplete(synthesis_prompt)
            final_result = str(final_response)
            total_tokens += count_tokens_estimate(synthesis_prompt + final_result)

            # Log final synthesis
            self.log_output(
                scenario_name=BenchmarkScenario.COMPLEX_WORKFLOW.value,
                task_name="Final Synthesis",
                output=final_result,
            )

        except Exception as e:
            logging.error(f"Error in complex workflow: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.throughput_tasks_per_sec = calculate_throughput(len(COMPLEX_WORKFLOW_STEPS) + 1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_memory_intensive(self) -> BenchmarkMetrics:
        """Run a memory-intensive benchmark using LlamaIndex."""
        self.monitor.start_monitoring()

        try:
            # Create multiple documents for memory-intensive processing
            large_documents = []
            for i in range(50):
                large_content = MEMORY_INTENSIVE_PROMPT + f" Document {i+1}. " + "Lorem ipsum dolor sit amet. " * 100
                large_documents.append(Document(text=large_content))

            # Create a larger index
            memory_index = VectorStoreIndex.from_documents(large_documents)
            memory_query_engine = memory_index.as_query_engine()

            # Execute memory-intensive query
            query = "Analyze and summarize all the information across all documents, identifying key patterns and insights."
            response = memory_query_engine.query(query)
            result = str(response)

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.MEMORY_INTENSIVE.value,
                task_name="Memory Intensive Query",
                output=result,
            )

            # Count tokens
            token_count = count_tokens_estimate(query + result)

        except Exception as e:
            logging.error(f"Error in memory intensive task: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = token_count
        metrics.throughput_tasks_per_sec = calculate_throughput(1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_concurrent_tasks(self) -> BenchmarkMetrics:
        """Run concurrent tasks benchmark using LlamaIndex."""
        self.monitor.start_monitoring()

        try:
            # Create multiple concurrent tasks using both LLM and query engine
            if self.llm is None:
                raise ValueError("LLM not initialized")
            llm_tasks = []
            query_tasks = []

            # Half the tasks use direct LLM
            for i, prompt in enumerate(CONCURRENT_TASK_PROMPTS[: len(CONCURRENT_TASK_PROMPTS) // 2]):
                llm_tasks.append(self.llm.acomplete(f"Task {i+1}: {prompt}"))

            # Half the tasks use query engine
            for i, prompt in enumerate(
                CONCURRENT_TASK_PROMPTS[len(CONCURRENT_TASK_PROMPTS) // 2 :],
                start=len(CONCURRENT_TASK_PROMPTS) // 2,
            ):
                # Note: query_engine.query is not async, so we'll use LLM for all concurrent tasks
                query_tasks.append(self.llm.acomplete(f"Query Task {i+1}: {prompt}"))

            # Execute all tasks concurrently
            all_tasks = llm_tasks + query_tasks
            responses = await asyncio.gather(*all_tasks)
            results = [str(response) for response in responses]

            # Count tokens
            total_tokens = sum(count_tokens_estimate(prompt + result) for prompt, result in zip(CONCURRENT_TASK_PROMPTS, results))

            # Log outputs
            for i, (_prompt, result) in enumerate(zip(CONCURRENT_TASK_PROMPTS, results)):
                self.log_output(
                    scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                    task_name=f"Concurrent Task {i+1}",
                    output=result,
                )

        except Exception as e:
            logging.error(f"Error in concurrent tasks: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(CONCURRENT_TASK_PROMPTS)
        metrics.throughput_tasks_per_sec = calculate_throughput(len(CONCURRENT_TASK_PROMPTS), metrics.execution_time_ms / 1000)

        return metrics
