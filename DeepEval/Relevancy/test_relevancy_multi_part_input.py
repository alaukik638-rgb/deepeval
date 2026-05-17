from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

def test_relevancy():
    metric = AnswerRelevancyMetric(
        threshold = 0.5,
        include_reason = True,
    )

    test_case_1 = LLMTestCase(
        input = "Where is Taj Mahal and when it was built?",
        actual_output = "Taj Mahal is in Agra, India. It was built in 1776."
    )

    test_case_2 = LLMTestCase(
        input = "Where is Taj Mahal and when it was built?",
        actual_output = "Taj Mahal is located in Agra, India and it was built in 1776. It is one of the seven wonders of the world.",
    )

    test_case_3 = LLMTestCase(
        input="Where is Taj Mahal and when it was built?",
        actual_output = "Taj Mahal is located in Agra, India and it was built in 1776 by Shah Jahan."
    )

    evaluate(test_cases=[test_case_1, test_case_2, test_case_3], metrics=[metric])