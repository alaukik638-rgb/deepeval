from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

def test_custom():
    custom_faithfulness_metric = GEval(
        name = "Medical Diagnosis faithfulness_metric",
        criteria="Evaluate the factual alignment of the actual output with the retrieved contextual information in a medical context.",
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold = 0.5,
    )

    test_cases = [

        # Case 1: Faithful (should PASS)
        LLMTestCase(
            input="What are the symptoms of diabetes?",
            actual_output="Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
            retrieval_context=[
                "Diabetes symptoms include excessive thirst, frequent urination, fatigue, and blurred vision."
            ]
        ),

        # Case 2: Hallucination (should FAIL)
        LLMTestCase(
            input="What are the symptoms of diabetes?",
            actual_output="Diabetes causes hair loss, skin peeling, and joint swelling.",
            retrieval_context=[
                "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision."
            ]
        ),

        # Case 3: Contradiction (should FAIL)
        LLMTestCase(
            input="Can hypertension cause headaches?",
            actual_output="Hypertension never causes headaches.",
            retrieval_context=[
                "Hypertension may sometimes cause headaches, especially in severe cases."
            ]
        ),

        # Case 4: Partial faithfulness_metric (borderline)
        LLMTestCase(
            input="What are treatments for asthma?",
            actual_output="Asthma is treated using inhalers.",
            retrieval_context=[
                "Asthma treatment includes inhalers, corticosteroids, and avoiding triggers."
            ]
        ),

        # Case 5: Extra Unsupported Info (should FAIL)
        LLMTestCase(
            input="What is anemia?",
            actual_output="Anemia is a condition where there is a lack of red blood cells and it always leads to cancer.",
            retrieval_context=[
                "Anemia is a condition characterized by a deficiency of red blood cells or hemoglobin."
            ]
        ),

        # Case 6: Multi-context Faithful (should PASS)
        LLMTestCase(
            input="What causes flu?",
            actual_output="Flu is caused by influenza viruses and spreads through respiratory droplets.",
            retrieval_context=[
                "Influenza (flu) is caused by influenza viruses.",
                "It spreads mainly through droplets when infected people cough or sneeze."
            ]
        ),

        # Case 7: Mixing Correct + Incorrect (should FAIL)
        LLMTestCase(
            input="What are symptoms of COVID-19?",
            actual_output="COVID-19 symptoms include fever, cough, and broken bones.",
            retrieval_context=[
                "COVID-19 symptoms include fever, cough, fatigue, and loss of taste or smell."
            ]
        ),
    ]

    evaluate(test_cases=test_cases, metrics=[custom_faithfulness_metric])
