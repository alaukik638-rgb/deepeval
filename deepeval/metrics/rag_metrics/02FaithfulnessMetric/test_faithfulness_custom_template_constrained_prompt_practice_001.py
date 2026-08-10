from models.google_gemini_ai_model import get_gemini_model
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate
from deepeval.test_case import LLMTestCase
from typing import List

class FaithfulnessCustomTemplateConstrainedPrompt(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str, multimodal: bool = False):
        return f"""Extract factual claims from the text.

        Rules:
        - One claim per statement
        - Keep claims concise and atomic
        - No opinions or interpretations
        - Max 10 claims
        - Return ONLY JSON
        
        Text:
        {actual_output}
        
        JSON:

        """

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str, multimodal: bool = False):
        return """Verify if each claim is supported by the context.

        Rules:
        - Answer only "yes" or "no" for each claim
        - "yes" = claim is directly supported by context
        - "no" = claim is NOT supported or contradicted
        - Be strict: if unsure, answer "no"
        - Return ONLY JSON
        
        Claims:
        {claims}
        
        Context:
        {retrieval_context}
        
        Verdicts:
        {{
            "verdicts": [
                {{}}
            ]
        }}
        
        JSON:
        
        """

    @staticmethod
    def generate_reason(
            score: float,
            contradictions: List[str],
            multimodal: bool = False
    ):
        return """Explain the faithfulness score in 1-2 sentences.

        Rules:
        - Be concise
        - Focus on unsupported claims if score is low
        - Max 50 words
        
        Score: {score}
        Contradictions: {contradictions}
        
        Reason:
        
        """

def test_faithfulness_custom_template_constrained_prompt():
    gemini_model = get_gemini_model()

    metric = FaithfulnessMetric(
        threshold = 0.7,
        model = gemini_model,
        include_reason = True,
        async_mode = False,
        evaluation_template = FaithfulnessCustomTemplateConstrainedPrompt,
    )

    # metric.evaluation_model.using_custom_template = True
    # metric.evaluation_model.custom_template = FaithfulnessCustomTemplateConstrainedPrompt

    test_case = LLMTestCase(
        input = "What if these shoes don't fit?",
        actual_output = "We offer a 30-day full refund at no extra cost.",
        retrieval_context = [
            "All customers are eligible for a 30 day full refund at no extra cost."
        ]
    )

    metric.measure(test_case)
    print(f"Metric Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")

    if __name__ == "__main__":
        test_faithfulness_custom_template_constrained_prompt()