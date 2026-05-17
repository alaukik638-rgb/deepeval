from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset, Golden

load_dotenv()

def test_correctness():
    correctness_metric = GEval(
        name = "Correctness",
        criteria = "Check if the actual output is same as the expected output. Don't focus too much on formatting. If the meaning is the same that's fine. It just has to be a true answer",
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold = 0.5
    )

    goldens = [
        Golden(input = "What is the capital of India?", expected_output = "New Delhi"),
        Golden(input = "What is 12 * 3?", expected_output = "36"),
    ]

    dataset = EvaluationDataset(goldens = goldens)

    def simulate_LLM_answers(prompt):
        return{
            "What is the capital of India?": "The capital of India is New Delhi",
            "What is 12 * 3?": "thirty-six"
        }[prompt]

    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
                input = golden.input,
                expected_output = golden.expected_output,
                actual_output = simulate_LLM_answers(golden.input)
            )
        )
    evaluate(test_cases=dataset.test_cases, metrics=[correctness_metric])