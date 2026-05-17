from models.google_gemini_ai_model import get_gemini_model
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

def test_answer_relevancy_with_gemini():
    gemini_model = get_gemini_model()
    metric = AnswerRelevancyMetric(
        threshold = 0.5,
        model = gemini_model,
    )

    test_case = LLMTestCase(
        input = "Where is the capital of Belgium?",
        actual_output = "Brussels is the capital of Belgium."
    )

    evaluate(test_cases = [test_case], metrics = [metric])