from deepeval.tracing import observe
from DeepEval.Metrics.AgenticMetrics.agents.ecommerce_customer_support_agent_practice_001 import support_agent
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()

def test_task_completion_metric_ecommerce_support_agent():
    @observe()
    def run_ecommerce_support_agent(user_input: str) -> str:
        result = support_agent.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": user_input
                }]
            }
        )
        return result["messages"][-1].content

    dataset = EvaluationDataset(
        goldens = [
            Golden(input = "I want to refund order ORD-001 because the item was damaged.", multimodal = False),
            Golden(input = "Where is my order ORD-002?", multimodal = False),
            Golden(input = "Show me some shoes.", multimodal = False),
        ]
    )

    task_completion_metric = TaskCompletionMetric(
        threshold = 0.7,
        model = "gpt-4o",
        include_reason = True,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
        task = None
    )

    test_cases = []
    for golden in dataset.goldens:
        output = run_ecommerce_support_agent(golden.input)
        test_cases.append(LLMTestCase(
            input = golden.input,
            actual_output = output
        ))

    evaluate(test_cases, metrics = [task_completion_metric])

""" 
actual output: The refund for order ORD-001 has been initiated due to the item being damaged. 
You will receive a confirmation within 24 hours.

Task Completion (score: 0.95, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
reason: The actual outcome aligns closely with the desired task as the refund process for the specified order was initiated. 
However, the task could have been considered fully complete if the confirmation had already been sent, 
rather than being scheduled for the future., error: None)
"""

"""
actual output: Here are some shoes available:
1. **Blue Running Shoes** - $49
2. **Red Sneakers** - $39

Task Completion (score: 0.9, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
reason: The system successfully displayed available shoe options by providing a list with specific details such as color and price. 
However, it could have included more options or additional details like sizes or brands to fully meet the task's potential scope., 
error: None)
"""

"""
actual output: Your order ORD-002 is currently being processed and is expected to ship in 2 days.
 
Task Completion (score: 0.95, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
reason: The system successfully determined the status of order ORD-002 as 'being processed' and provided an 
expected shipping date of 2 days, which aligns closely with the task requirements. 
However, it could have included additional details such as the current stage of processing or potential delays 
to achieve a perfect score., error: None)
"""