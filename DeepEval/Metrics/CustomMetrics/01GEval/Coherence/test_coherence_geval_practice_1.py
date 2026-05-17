from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

load_dotenv()

def test_coherence():
    coherence_metric = GEval(
        name = "Coherence/Clarity",
        evaluation_steps = [
            "Evaluate whether the response uses clear and direct language.",
            "Check if the explanation avoids jargon or explains it when used.",
            "Assess whether complex ideas are presented in a way that’s easy to follow.",
            "Identify any vague or confusing parts that reduce understanding."
        ],
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    test_case_1 = LLMTestCase(
        input = "What is the capital of India?",
        actual_output = "Capital of India is New Delhi."
    )

    test_case_2 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="New Delhi."
    )

    test_case_3 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="Capital of India is Delhi. Isn't it? Or is it Mumbai?"
    )

    test_case_4 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="You should not ask these kind of questions. You must know the answer of such simple questions. Are you dumb or mentally challenged?"
    )

    test_case_5 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="Capital of India is New Delhi. Isn't it? Or is it Calcutta? It used to be Calcutta once upon a time."
    )

    test_case_6 = LLMTestCase(
        input="What is the capital of India?",
        actual_output="Capital of India is New Delhi. Delhi is one of the oldest cities in India. Delhi is located on the banks of Yamuna. Yamuna is one of the holiest rivers in India. Pandavas of Mahabharata have their capital named as Indraprastha which a lot of historians believe was in and around modern day Delhi."
    )

    # evaluate(test_cases=[test_case_1, test_case_4], metrics=[coherence_metric])
    evaluate(test_cases=[test_case_6], metrics=[coherence_metric])