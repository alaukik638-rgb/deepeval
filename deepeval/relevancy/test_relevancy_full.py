from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

def test_relevancy():
    metric = AnswerRelevancyMetric(
        threshold = 0.5,
        include_reason = True
    )
    # relevancy = Number of relevant statements / Total number of statements

    test_case = LLMTestCase(
        input = "What is the capital of India",
        actual_output = 'New Delhi'
    )

    evaluate(test_cases=[test_case], metrics=[metric])
