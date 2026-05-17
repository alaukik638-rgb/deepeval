from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

load_dotenv()

def test_faithfulness():
    metric = FaithfulnessMetric(
        threshold = 0.5,
        include_reason = True,
    )

    test_case = LLMTestCase(
        input = "What is the capital of India",
        actual_output = "New Delhi",
        retrieval_context = ["Saharsa is the capital of India"]
    )

    evaluate(test_cases=[test_case], metrics=[metric])