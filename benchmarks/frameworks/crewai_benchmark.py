"""CrewAI framework benchmark implementation."""

import asyncio
import os
from typing import Any, Dict, List, Optional

try:
    from crewai import LLM, Agent, Crew, Task
    from crewai.llm import LLM as CrewAILLM
except ImportError:
    Agent = None
    Task = None
    Crew = None
    LLM = None
    CrewAILLM = None

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


class CrewAIBenchmark(BaseBenchmark):
    """CrewAI framework benchmark implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize CrewAI benchmark with configuration."""
        super().__init__(config)
        self.llm: Optional[Any] = None
        self.agents: Dict[str, Any] = {}
        self.crews: Dict[str, Any] = {}

        if Agent is None or Task is None or Crew is None:
            raise ImportError("CrewAI not installed. Please install with: pip install crewai")

    async def setup(self) -> None:
        """Set up CrewAI for benchmarking."""
        # Get standardized configuration
        llm_config = get_standard_llm_config(self.config)
        api_key = os.getenv("OPENAI_API_KEY") or llm_config["api_key"]
        if not api_key:
            raise ValueError("OpenAI API key not found in environment or config")

        # Initialize CrewAI LLM - using the correct format
        try:
            self.llm = LLM(
                model=f"openai/{llm_config['model']}",  # Use provider/model format
                api_key=api_key,
                temperature=llm_config["temperature"],
                max_tokens=llm_config.get("max_tokens", 4000),
            )
        except Exception as e:
            # Don't fall back to None, raise the error instead
            raise ValueError(f"Failed to initialize CrewAI LLM: {e}")

        # Pre-create common agents
        self._setup_agents()

    def _setup_agents(self) -> None:
        """Set up common CrewAI agents."""
        # General purpose agent
        agent_config = {
            "role": "General Assistant",
            "goal": "Complete assigned tasks efficiently and accurately",
            "backstory": "You are an experienced assistant capable of handling various tasks.",
            "verbose": False,
            "allow_delegation": False,
        }
        if self.llm:
            agent_config["llm"] = self.llm

        self.agents["general"] = Agent(**agent_config)

        # Analysis agent
        analyst_config = {
            "role": "Data Analyst",
            "goal": "Analyze data and provide insights",
            "backstory": "You are a skilled data analyst with expertise in pattern recognition.",
            "verbose": False,
            "allow_delegation": False,
        }
        if self.llm:
            analyst_config["llm"] = self.llm

        self.agents["analyst"] = Agent(**analyst_config)

        # Content creator agent
        creator_config = {
            "role": "Content Creator",
            "goal": "Create engaging and informative content",
            "backstory": "You are a creative writer with expertise in marketing and communication.",
            "verbose": False,
            "allow_delegation": False,
        }
        if self.llm:
            creator_config["llm"] = self.llm

        self.agents["creator"] = Agent(**creator_config)

        # Technical expert agent
        technical_config = {
            "role": "Technical Expert",
            "goal": "Provide technical solutions and explanations",
            "backstory": "You are a technical expert with deep knowledge across various domains.",
            "verbose": False,
            "allow_delegation": False,
        }
        if self.llm:
            technical_config["llm"] = self.llm

        self.agents["technical"] = Agent(**technical_config)

    async def teardown(self) -> None:
        """Cleanup CrewAI resources."""
        self.llm = None
        self.agents = {}
        self.crews = {}

    async def run_simple_task(self) -> BenchmarkMetrics:
        """Run a simple single-task benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            # Create a simple task
            task = Task(
                description=SIMPLE_TASK_PROMPT,
                agent=self.agents["general"],
                expected_output="A comprehensive analysis and summary",
            )

            # Create crew with the task
            crew = Crew(agents=[self.agents["general"]], tasks=[task], verbose=False)

            # Execute the task (wrap synchronous call in async)
            result = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.SIMPLE_TASK.value,
                task_name="Simple Task",
                output=str(result),
            )

            # Count tokens
            token_count = count_tokens_estimate(SIMPLE_TASK_PROMPT + str(result))

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
        """Run a sequential pipeline benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            total_tokens = 0
            results: List[str] = []

            # Execute tasks sequentially like other frameworks, logging each individually
            for i, task_desc in enumerate(SEQUENTIAL_TASKS):
                if "product" in task_desc or "marketing" in task_desc:
                    agent = self.agents["creator"]
                else:
                    agent = self.agents["analyst"]

                # Build context from previous results
                if results:
                    context = f"Previous results: {' | '.join(results)}\n\n"
                    full_prompt = f"{context}Task {i+1}: {task_desc}"
                else:
                    full_prompt = f"Task {i+1}: {task_desc}"

                task = Task(
                    description=full_prompt,
                    agent=agent,
                    expected_output="Detailed response addressing the task requirements",
                )

                # Create crew for this individual task
                crew = Crew(agents=[agent], tasks=[task], verbose=False)

                # Execute task (wrap synchronous call in async)
                result = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)
                result_str = str(result)
                results.append(result_str)

                # Log each individual task output like other frameworks
                self.log_output(
                    scenario_name=BenchmarkScenario.SEQUENTIAL_PIPELINE.value,
                    task_name=f"Task {i+1}",
                    output=result_str,
                )

                # Count tokens for this task
                total_tokens += count_tokens_estimate(full_prompt + result_str)

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
        """Run a parallel pipeline benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            # Create parallel tasks with individual crews to simulate parallel execution
            async def execute_task(task_desc: str, agent_key: str) -> Any:
                agent = self.agents[agent_key]
                task = Task(
                    description=task_desc,
                    agent=agent,
                    expected_output="Clear and concise response",
                )

                crew = Crew(agents=[agent], tasks=[task], verbose=False)

                # Execute task (wrap synchronous call in async)
                return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

            # Create coroutines for parallel execution
            agent_keys = ["general", "technical", "creator", "analyst"]
            tasks = [execute_task(task_desc, agent_keys[i % len(agent_keys)]) for i, task_desc in enumerate(PARALLEL_TASKS)]

            # Execute tasks in parallel
            results = await asyncio.gather(*tasks)

            # Log individual parallel task outputs
            for i, (_task_desc, result) in enumerate(zip(PARALLEL_TASKS, results)):
                self.log_output(
                    scenario_name=BenchmarkScenario.PARALLEL_PIPELINE.value,
                    task_name=f"Parallel Task {i+1}",
                    output=str(result),
                )

            # Count tokens
            total_tokens = sum(count_tokens_estimate(task_desc + str(result)) for task_desc, result in zip(PARALLEL_TASKS, results))

        except Exception as e:
            self.logger.error(f"Error in parallel pipeline benchmark: {e}")
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
        """Run a complex workflow benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            results: Dict[str, str] = {}
            total_tokens = 0

            # Execute workflow steps in dependency order like other frameworks
            for step in COMPLEX_WORKFLOW_STEPS:
                # Choose appropriate agent based on task type
                if "analysis" in step["task"]:
                    agent = self.agents["analyst"]
                elif "solution" in step["task"] or "recommendation" in step["task"]:
                    agent = self.agents["technical"]
                elif "cost" in step["task"] or "risk" in step["task"]:
                    agent = self.agents["analyst"]
                else:
                    agent = self.agents["general"]

                # Build context from dependencies
                context_parts = []
                for dep in step["depends_on"]:
                    if dep in results:
                        context_parts.append(f"{dep}: {results[dep]}")

                context = " | ".join(context_parts) if context_parts else "None"
                full_prompt = f"Context: {context}\n\nTask: {step['prompt']}"

                task = Task(
                    description=full_prompt,
                    agent=agent,
                    expected_output="Comprehensive analysis and recommendations",
                )

                # Create crew for this step
                crew = Crew(agents=[agent], tasks=[task], verbose=False)

                # Execute step (wrap synchronous call in async)
                result = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)
                result_str = str(result)
                results[step["task"]] = result_str

                # Log individual workflow step output
                self.log_output(
                    scenario_name=BenchmarkScenario.COMPLEX_WORKFLOW.value,
                    task_name=step["task"],
                    output=result_str,
                )

                total_tokens += count_tokens_estimate(full_prompt + result_str)

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
        """Run a memory-intensive benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            # Create large data structures to test memory usage like other frameworks
            large_data = ["data" * 1000] * 1000  # ~4MB of string data

            # Create memory-intensive task
            task = Task(
                description=MEMORY_INTENSIVE_PROMPT,
                agent=self.agents["analyst"],
                expected_output="Detailed data analysis with insights and recommendations",
            )

            # Create crew with memory-intensive task
            crew = Crew(agents=[self.agents["analyst"]], tasks=[task], verbose=False)

            # Execute memory-intensive task (wrap synchronous call in async)
            result = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

            # Clean up like other frameworks
            del large_data

            # Log LLM output to file
            self.log_output(
                scenario_name=BenchmarkScenario.MEMORY_INTENSIVE.value,
                task_name="Memory Intensive",
                output=str(result),
            )

            # Count tokens
            token_count = count_tokens_estimate(MEMORY_INTENSIVE_PROMPT + str(result))

        except Exception as e:
            self.logger.error(f"Error in memory intensive benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = token_count
        metrics.throughput_tasks_per_sec = calculate_throughput(1, metrics.execution_time_ms / 1000)

        return metrics

    async def run_concurrent_tasks(self) -> BenchmarkMetrics:
        """Run concurrent tasks benchmark using CrewAI."""
        self.monitor.start_monitoring()

        try:
            # Create concurrent tasks with individual crews like parallel pipeline
            async def execute_concurrent_task(task_desc: str, agent_key: str) -> Any:
                agent = self.agents[agent_key]
                task = Task(
                    description=task_desc,
                    agent=agent,
                    expected_output="Clear and informative response",
                )

                crew = Crew(agents=[agent], tasks=[task], verbose=False)

                # Execute task (wrap synchronous call in async)
                return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

            # Create coroutines for concurrent execution
            agent_keys = ["general", "technical", "creator", "analyst"]
            tasks = [execute_concurrent_task(task_desc, agent_keys[i % len(agent_keys)]) for i, task_desc in enumerate(CONCURRENT_TASK_PROMPTS)]

            # Execute concurrent tasks
            results = await asyncio.gather(*tasks)

            # Log individual concurrent task outputs
            for i, (_task_desc, result) in enumerate(zip(CONCURRENT_TASK_PROMPTS, results)):
                self.log_output(
                    scenario_name=BenchmarkScenario.CONCURRENT_TASKS.value,
                    task_name=f"Concurrent Task {i+1}",
                    output=str(result),
                )

            # Count tokens
            total_tokens = sum(count_tokens_estimate(task_desc + str(result)) for task_desc, result in zip(CONCURRENT_TASK_PROMPTS, results))

        except Exception as e:
            self.logger.error(f"Error in concurrent tasks benchmark: {e}")
            metrics = self.monitor.stop_monitoring()
            metrics.error_rate = 1.0
            metrics.token_count = 0
            return metrics

        metrics = self.monitor.stop_monitoring()
        metrics.token_count = total_tokens
        metrics.concurrent_tasks = len(CONCURRENT_TASK_PROMPTS)
        metrics.throughput_tasks_per_sec = calculate_throughput(len(CONCURRENT_TASK_PROMPTS), metrics.execution_time_ms / 1000)

        return metrics
