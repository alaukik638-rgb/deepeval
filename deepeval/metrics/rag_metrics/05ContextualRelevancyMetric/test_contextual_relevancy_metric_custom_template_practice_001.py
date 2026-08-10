from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

class CustomContextualRelevancyTemplate(ContextualRelevancyTemplate):
    @staticmethod
    def generate_verdicts(input: str, context: str, multimodal: bool = False):
        return f"""Based on the input and context, please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input.
        Example JSON:
        {{
            "verdicts": [
                {{
                    "verdict": "yes",
                    "statement": "..."
                }}
            ]
        }}
        **
        
        Input:
        {input}
        
        Context:
        {context}
        
        JSON:
        """


def test_contextual_relevancy_metric_custom_template():
    metric = ContextualRelevancyMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
        evaluation_template = CustomContextualRelevancyTemplate
    )

    test_case = LLMTestCase(
        input = f"How can I reset my account password?",
        actual_output = f"Use the Forgot Password option on the login page.",
        retrieval_context = [
            f"Users can reset their passwords by clicking the 'Forgot Password' link on the login page and following the email instructions.",
        ]
    )

    evaluate([test_case], [metric])


