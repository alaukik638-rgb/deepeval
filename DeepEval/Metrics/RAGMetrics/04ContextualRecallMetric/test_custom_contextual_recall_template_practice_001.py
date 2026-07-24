from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics.contextual_recall import ContextualRecallTemplate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from typing import List

load_dotenv()

class CustomContextualRecallTemplate(ContextualRecallTemplate):
    @staticmethod
    def generate_verdicts(
            expected_output: str,
            retrieval_context: List[str],
            multimodal: bool = True
    ):
        return f"""
        For EACH sentence in the expected output, determine whether it can be attributed SOLELY to the retrieval context provided below.
    
        STRICT RULES:
        - You MUST NOT use any external knowledge about the topic.
        - If a claim is NOT explicitly stated in the retrieval context, verdict is "no".
        - If a claim is hedged (e.g., "might", "possibly") in the context, verdict is "no".
        - Partial support counts as "no".
        - Only unambiguous, direct support counts as "yes".
        
        Example JSON:
        {{
            "verdicts": [
                {{
                    "verdict": "yes",
                    "reason": "...",
                }}
            ]
        }}
        
        Expected Output:
        {expected_output}
        
        Retrieval Context:
        {retrieval_context}
        
        JSON:
        """

def test_custom_contextual_recall_template():
    metric = ContextualRecallMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        async_mode = True,
        verbose_mode = False,
        strict_mode = False,
        evaluation_template = CustomContextualRecallTemplate,
    )

    test_case = LLMTestCase(
        input="What are the main benefits of Playwright?",
        actual_output="""
            Playwright supports cross-browser testing and provides automatic waiting features.
            """,
        expected_output="""
            Playwright supports cross-browser testing, automatic waiting, network interception,
            and parallel execution for faster test runs.
            """,
        retrieval_context=[
            """
            Playwright is a modern automation framework that supports:
            - Cross-browser testing (Chromium, Firefox, WebKit)
            - Auto-waiting for stable execution
            - Network interception and API mocking
            """
        ]
    )

    evaluate(test_cases = [test_case], metrics = [metric])