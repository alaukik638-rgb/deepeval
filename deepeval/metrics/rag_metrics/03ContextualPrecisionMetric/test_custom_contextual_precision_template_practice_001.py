from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from typing import List

load_dotenv()

class CustomContextualPrecisionTemplate(ContextualPrecisionTemplate):
    @staticmethod
    def generate_verdicts(
            input: str,
            expected_output: str,
            retrieval_context: List[str],
            multimodal: bool = False
    ):
        return f"""
        Given the input, expected output, and retrieval context, please generate a list of JSON objects,
        to determine whether each node in the retrieval context was remotely useful in arriving at the expected output.
        
        Example JSON:
        {{
            "verdicts": [
                {{
                    "verdict": "yes",
                    "reason": "..."
                }}
            ]
        }}
        
        The number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
        **
        
        Input:
        {input}
        
        Expected Output:
        {expected_output}
        
        Retrieval Context:
        {retrieval_context}
        
        JSON:
        """

def test_custom_contextual_precision_metric():
    metric = ContextualPrecisionMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        evaluation_template = CustomContextualPrecisionTemplate,
        verbose_mode = True,
    )

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=[
            # "We do not care about our customers.",
            "All customers are eligible for a 30 day full refund at no extra cost.",
            "But we always try not to refund the money and blame customers for their faults.",
            "Full refund policy is just a gimmick to attract customers.",
            "Usually we provide partial store credit which amounts to no more than 30% of actual spend and can be used in that store only",
        ],
    )

    evaluate(test_cases = [test_case], metrics = [metric])