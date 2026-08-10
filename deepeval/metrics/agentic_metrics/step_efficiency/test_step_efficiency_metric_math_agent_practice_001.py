import pytest
from deepeval import evaluate, assert_test
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from dotenv import load_dotenv
from deepeval.metrics.agentic_metrics.agents.math_calculator_agent_practice_001 import math_agent

load_dotenv()

step_efficiency_metric = StepEfficiencyMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
        strict_mode = False,
        async_mode = True
)

dataset = EvaluationDataset(
    goldens = [
        Golden(input="What is 12 multiplied by 7?", multimodal=False),
        Golden(input="Divide 100 by 4 and then add 15.", multimodal=False),
        Golden(input="What is 9 divided by 0", multimodal=False),
    ]
)

# Attach metric to observe(), not to evaluate()
@observe(name = "math_agent_run", metrics = [step_efficiency_metric])
def run_math_agent(user_input: str) -> str:
    update_current_trace(
        input = user_input,
    )
    result = math_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )
    output = result["messages"][-1].content
    # Set output on the trace
    update_current_trace(output = output)
    return output

@pytest.mark.parametrize("golden", dataset.goldens)
def test_step_efficiency_metric_math_calculator(golden: Golden):
    run_math_agent(golden.input)
    assert_test(
        golden=golden,
        observed_callback = run_math_agent,
    )