from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from ollama_model import OllamaLLM

# load_dotenv()

# ollama_llm = OllamaLLM(model = "llama3:8b")
ollama_llm = OllamaLLM(model = "mistral")

def test_all_metric():
    faithfulness_metric = FaithfulnessMetric(
        penalize_ambiguous_claims = True,
        threshold = 0.5,
        model = ollama_llm
        # evaluation_template = []
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold = 0.5,
    )

    contextual_precision_metric = ContextualPrecisionMetric(
        threshold = 0.5,
    )

    contextual_recall_metric = ContextualRecallMetric(
        threshold = 0.5,
    )

    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold = 0.5,
    )

    test_cases = [

        # Case 1: Faithful (should pass)
        LLMTestCase(
            input="What are the symptoms of diabetes?",
            actual_output="Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
            expected_output="Symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
            retrieval_context=[
                "Diabetes symptoms include excessive thirst, frequent urination, fatigue, and blurred vision."
            ]
        ),

        # Case 2: Hallucination (should fail)
        LLMTestCase(
            input="What are the symptoms of diabetes?",
            actual_output="Diabetes causes hair loss, skin peeling, and joint swelling.",
            expected_output="Symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
            retrieval_context=[
                "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision."
            ]
        ),

        # Case 3: Contradiction (should fail)
        LLMTestCase(
            input="Can hypertension cause headaches?",
            actual_output="Hypertension never causes headaches.",
            expected_output="Hypertension can sometimes cause headaches, especially in severe cases.",
            retrieval_context=[
                "Hypertension may sometimes cause headaches, especially in severe cases."
            ]
        ),

        # Case 4: Partial Faithfulness (borderline)
        LLMTestCase(
            input="What are treatments for asthma?",
            actual_output="Asthma is treated using inhalers.",
            expected_output="Asthma treatment includes inhalers, corticosteroids, and avoiding triggers.",
            retrieval_context=[
                "Asthma treatment includes inhalers, corticosteroids, and avoiding triggers."
            ]
        ),

        # Case 5: Extra Unsupported Info (should fail)
        LLMTestCase(
            input="What is anemia?",
            actual_output="Anemia is a condition where there is a lack of red blood cells and it always leads to cancer.",
            expected_output="Anemia is a condition characterized by a deficiency of red blood cells or hemoglobin.",
            retrieval_context=[
                "Anemia is a condition characterized by a deficiency of red blood cells or hemoglobin."
            ]
        ),

        # Case 6: Multi-context Faithful (should pass)
        LLMTestCase(
            input="What causes flu?",
            actual_output="Flu is caused by influenza viruses and spreads through respiratory droplets.",
            expected_output="Flu is caused by influenza viruses and spreads via respiratory droplets.",
            retrieval_context=[
                "Influenza (flu) is caused by influenza viruses.",
                "It spreads mainly through droplets when infected people cough or sneeze."
            ]
        ),

        # Case 7: Mixed Correct + Incorrect (should fail)
        LLMTestCase(
            input="What are symptoms of COVID-19?",
            actual_output="COVID-19 symptoms include fever, cough, and broken bones.",
            expected_output="COVID-19 symptoms include fever, cough, fatigue, and loss of taste or smell.",
            retrieval_context=[
                "COVID-19 symptoms include fever, cough, fatigue, and loss of taste or smell."
            ]
        ),
    ]

    test_case_1 = LLMTestCase(
        input="What are the symptoms of diabetes?",
        actual_output="Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
        expected_output="Symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
        retrieval_context=[
            "Diabetes symptoms include excessive thirst, frequent urination, fatigue, and blurred vision."
        ]
    )

    test_case_2 = LLMTestCase(
        input="What are the symptoms of diabetes?",
        actual_output="Diabetes causes hair loss, skin peeling, and joint swelling.",
        expected_output="Symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
        retrieval_context=[
            "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision."
        ]
    )

    # evaluate(test_cases=[test_case_2], task_completion=[faithfulness_metric, answer_relevancy_metric, contextual_precision_metric,
    #                                          contextual_recall_metric, contextual_relevancy_metric])
    evaluate(test_cases=[test_case_2], metrics=[faithfulness_metric])