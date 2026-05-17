from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

load_dotenv()

def test_faithfulness():
    metric = FaithfulnessMetric(
        threshold = 0.5,
        include_reason = True,
        #model = "o1",
    )

    test_case_1 = LLMTestCase(
        input = "What is the capital of India",
        actual_output = "Delhi",
        retrieval_context = ["Saharsa is the capital of India", "Delhi is the capital of India"],
    )

    test_case_2 = LLMTestCase(
        input="What is the capital of India",
        actual_output="New Delhi",
        retrieval_context = ["Either Saharsa or Delhi is the capital of India"]
    )

    evaluate(test_cases=[test_case_1, test_case_2], metrics=[metric])