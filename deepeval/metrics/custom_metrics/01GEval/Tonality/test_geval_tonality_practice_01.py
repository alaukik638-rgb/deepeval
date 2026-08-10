from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

def test_tonality():
    tonality_metrics = GEval(
        name = "Professionalism",
        evaluation_steps = [
            "Determine whether the actual output maintains a professional tone throughout.",
            "Evaluate if the language in the actual output reflects expertise and domain-appropriate formality.",
            "Ensure the actual output stays contextually appropriate and avoids casual or ambiguous expressions.",
            "Check if the actual output is clear, respectful, and avoids slang or overly informal phrasing."
        ],
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    test_case_1 = LLMTestCase(
        input = "What is the capital of India?",
        actual_output = "What's wrong with you man! Are you even an Indian? Why don't you know the capital of your own country? You sound pretty dumb to me."
    )

    test_case_2 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="The capital of India can be anywhere. It can be wherever you want. Do you want it to be New Delhi? Or may be Mumbai? How about Kolkata?"
    )

    test_case_3 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="This is a very nice question. I want to thank you for asking such an important question. India is a great country and its capital is Delhi."
    )

    evaluate(test_cases=[test_case_1, test_case_2, test_case_3], metrics=[tonality_metrics])