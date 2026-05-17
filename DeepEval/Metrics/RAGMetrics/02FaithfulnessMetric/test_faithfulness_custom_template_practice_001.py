from models.google_gemini_ai_model import get_gemini_model
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate
from deepeval.test_case import LLMTestCase

class CustomFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str, multimodal: bool = False):
        return f"""Based on the given text, please extract a comprehensive list of facts that can inferred from the provided text.

        Example:
        Example Text:
        "CNN claims that the sun is 3 times smaller than earth."
        
        Example JSON:
        {{
            "claims": []
        }}
        ===== END OF EXAMPLE ======
        
        Text:
        {actual_output}
        
        JSON:
        """

def test_faithfulness_custom_template():
    gemini_model = get_gemini_model()
    metric = FaithfulnessMetric(
        threshold = 0.6,
        include_reason = True,
        evaluation_template = CustomFaithfulnessTemplate,
        model = gemini_model
    )
    user_input = "What if these shoes don't fit?"

    actual_output = "We offer a 30-day full refund at no extra cost."

    retrieval_context = [
        "All customers are eligible for a 30 day full refund at no extra cost."
    ]

    test_case = LLMTestCase(
        input = user_input,
        actual_output = actual_output,
        retrieval_context = retrieval_context
    )

    metric.measure(test_case)
    print(metric.score, metric.reason)