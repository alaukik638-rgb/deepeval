from deepeval.tracing import observe, Trace
from DeepEval.Metrics.AgenticMetrics.agents.math_calculator_agent_practice_001 import math_agent
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()

def test_task_completion_metric_math_agent():
    # Wrap the agent in @observe so deepeval can trace it
    @observe()
    def run_math_agent(user_input: str) -> str:
        result = math_agent.invoke({
            "messages": [{"role": "user", 'content': user_input}]
        })
        # Last message in the list is the final agent response
        return result["messages"][-1].content

    # Dataset with meaningful math inputs
    dataset = EvaluationDataset(
        goldens = [
            Golden(input = "What is 12 multiplied by 7?", multimodal = False),
            Golden(input = "Divide 100 by 4 and then add 15.", multimodal = False),
            Golden(input = "What is 9 divided by 0", multimodal = False)
        ]
    )

    # Metric - Task Completion
    task_completion_metric = TaskCompletionMetric(
        threshold = 0.7,
        model = "gpt-4o",
        include_reason = True,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
    )

    # Build test_cases by running the agent
    test_cases = []
    for golden in dataset.goldens:
        output = run_math_agent(golden.input)
        test_cases.append(LLMTestCase(
            input = golden.input,
            actual_output = output
        ))

    evaluate(test_cases, metrics=[task_completion_metric])

