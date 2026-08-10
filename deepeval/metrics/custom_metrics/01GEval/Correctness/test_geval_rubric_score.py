from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

load_dotenv()

def test_correctness():
    correctness_metric = GEval(
        name = "correctness_metric",
        criteria = "Determine whether the actual output is factually correct based on the expected output",
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        rubric = [
            Rubric(score_range=(0, 2), expected_outcome="Factually incorrect"),
            Rubric(score_range=(3, 6), expected_outcome="Mostly correct"),
            Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details"),
            # Note: Do not use overlapping ranges
            Rubric(score_range=(10, 10), expected_outcome="100% correct")
        ],
        # strict_mode = True
    )

    test_case_1 = LLMTestCase(
        input = "The dog chased the cat up the tree, who ran up the tree?",
        actual_output = "It depends, some might consider the cat, while others might argue the dog.",
        expected_output = "The cat."
    )

    test_case_2 = LLMTestCase(
        input = "What is the capital of India?",
        actual_output = "Delhi",
        expected_output = "New Delhi"
    )

    correctness_metric.measure(test_case_1)
    print("First Test Case rubric score: ", correctness_metric.score, correctness_metric.reason)

    correctness_metric.measure(test_case_2)
    print("Second Test Case rubric score: ", correctness_metric.score, correctness_metric.reason)

    evaluate(test_cases=[test_case_1, test_case_2], metrics=[correctness_metric])
