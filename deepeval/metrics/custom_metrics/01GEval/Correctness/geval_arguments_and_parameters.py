from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()

correctness_metric = GEval(
    name = "correctness_metric",
    criteria = "Determine whether the actual output is factually correct based on the expected output",
    # Note: You can use either criteria or evaluation_steps and not both
    # evaluation_steps = [
    #     "Check whether the facts in 'actual output' contradicts any fact in 'expected output'",
    #     "You should also heavily penalize omission of details",
    #     "Vague language or contradicting OPINIONS are OK"
    # ],
    evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="It depends, some might consider the cat, while others might argue the dog.",
    expected_output="The cat."
)

evaluate(test_cases=[test_case], metrics=[correctness_metric])