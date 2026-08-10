from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
load_dotenv()

def test_correctness():
    correctness_metric = GEval(
        name = "correctness_metric",
        criteria = "Determine whether the actual output is factually correct based on the expected output.",
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )

    test_case = LLMTestCase(
        input = "The dog chased the cat up the tree, who ran up the tree?",
        actual_output = "It depends, some might consider the cat, while others might argue the dog.",
        expected_output="The cat."
    )

    correctness_metric.measure(test_case)
    print(correctness_metric.score, correctness_metric.reason)

test_correctness()