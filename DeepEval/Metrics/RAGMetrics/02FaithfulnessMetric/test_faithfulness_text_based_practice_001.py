import os
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

load_dotenv()

# class GoogleGeminiAI()

def test_faithfulness_text_based():

    metric = FaithfulnessMetric(
        threshold=0.6,
        model="openai/gpt-4.1",
        # model = "gemini-2.5-flash",
        include_reason=True,
    )

    user_input = "What if these shoes don't fit?"

    actual_output = "We offer a 30-day full refund at no extra cost."

    retrieval_context = [
        "All customers are eligible for a 30 day full refund at no extra cost."
    ]

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    evaluate(
        test_cases=[test_case],
        metrics=[metric]
    )