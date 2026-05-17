from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, trace_manager
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()


def test_within_component_answer_relevancy_metric():
    metric = AnswerRelevancyMetric(
        threshold=0.5,
    )

    @observe(metrics=[metric])
    def inner_component(user_input: str):
        # Simulate some processing (e.g., LLM call)
        response = "Hello! How can I help you today?"

        # Set test case at runtime for THIS component
        test_case = LLMTestCase(
            input=user_input,
            actual_output=response
        )
        update_current_span(test_case=test_case)
        return response

    @observe
    def llm_app(input: str):
        output = inner_component(input)
        return output

    # Execute with tracing - metrics evaluate at component level
    goldens = [
        Golden(input="Hi")
    ]

    for golden in goldens:
        llm_app(golden.input)

    # Results are captured in the trace