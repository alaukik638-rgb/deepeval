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

    test_case_1 = LLMTestCase(
        input  = "What is the capital of India?",
        actual_output = "This is a very good question. The capital of India is New Delhi. It is one of the best cities in Asia. You should visit it."
    )

    test_case_2 = LLMTestCase(
        input = "Who is the Prime minister of India?",
        actual_output = "Jawahar Lal Nehru was the first Prime minister of India after independence."
    )

    evaluate(test_cases=[test_case_1, test_case_2], metrics=[metric])