from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

load_dotenv()

def test_correctness():
    correctness_metric = GEval(
        name = 'Correctness',
        criteria = "Check if the actual output is same as the expected output. Don't focus too much on formatting. If the meaning is the same that's fine. It just has to be a true answer",
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold = 0.5
    )

    test_case = LLMTestCase(
        input = "What is 5 divided by 2?",
        expected_output = '2.5',
        actual_output = "The result is 2.5"
    )

    assert_test(test_case, [correctness_metric])