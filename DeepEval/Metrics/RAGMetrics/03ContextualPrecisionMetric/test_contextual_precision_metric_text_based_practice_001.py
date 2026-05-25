from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

def test_contextual_precision_metric_text():
    metric = ContextualPrecisionMetric(
        model = "gpt-4.1",
        threshold = 0.7,
    )

    # Irrelevant node is appearing at the end.
    test_case_1 = LLMTestCase(
        input = "What if these shoes don't fit?",
        actual_output = "We offer a 30-day full refund at no extra cost.",
        expected_output = "You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context = [
            "All customers are eligible for a 30 day full refund at no extra cost.",
            "We don't care about our customers.",
        ]
    )

    # Irrelevant node is appearing at the beginning.
    test_case_2 = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=[
            "We don't care about our customers.",
            "All customers are eligible for a 30 day full refund at no extra cost.",
        ]
    )

    evaluate(test_cases = [test_case_1, test_case_2], metrics = [metric])