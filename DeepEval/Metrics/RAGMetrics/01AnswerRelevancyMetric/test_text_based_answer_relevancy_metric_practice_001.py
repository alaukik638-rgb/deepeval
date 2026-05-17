from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

def test_text_based_answer_relevancy_metric():
    metric = AnswerRelevancyMetric(
        threshold = 0.7,
        model = "gpt-4.1",
        include_reason = True,
    )

    test_case = LLMTestCase(
        input = "What if these shoes don't fit?",
        actual_output = "We offer a 30-day full refund at no extra cost.",
    )

    # metric.measure(test_case)
    # print(metric.score, metric.reason)
    # evaluate(metrics = [metric], test_cases = [test_case])
    evaluate([test_case], [metric])