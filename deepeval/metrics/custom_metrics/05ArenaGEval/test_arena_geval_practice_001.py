from deepeval import compare
from deepeval.metrics import ArenaGEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, Contestant, ArenaTestCase
from dotenv import load_dotenv

load_dotenv()

def test_arena_geval():
    arena_test_case = ArenaTestCase(
        contestants = [
            Contestant(
                name = "GPT-4",
                hyperparameters = {'model': 'gpt-4'},
                test_case = LLMTestCase(
                    input = "What is the capital of India?",
                    actual_output = "New Delhi",
                ),
            ),
            Contestant(
                name = "GPT-4.1",
                hyperparameters = {'model': 'gpt-4.1'},
                test_case = LLMTestCase(
                    input = "What is the capital of India?",
                    actual_output = "New Delhi is the capital of India."
                ),
            ),
        ],
    )

    metric = ArenaGEval(
        name = 'Friendly',
        criteria = "Choose the winner of the more friendly contestant based on the input and actual output",
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    compare(test_cases = [arena_test_case], metric = metric)
