from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

# “Custom prompts that are not constrained lead to latency failures, not just accuracy issues.”

class CustomAnswerRelevancyTemplateNotConstrainedPrompt(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str, multimodal: bool = False):
        return f"""
        Given the test, breakdown and generate a list of statements presented.
        
        Example:
        Our new laptop model features a high-resolution Retina display for crystal-clear visuals.
        
        {{
            "statements": [
            "The new laptop model has a high-resolution Retina display."
            ]
        }}
        
        ==== END OF EXAMPLE ====
        
        Text:
        {actual_output}
        
        JSON:
        """

class CustomAnswerRelevancyTemplateConstrainedPrompt(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str, multimodal: bool = False):
        return f"""
        Extract concise factual statements from the text.
        
        Rules:
        - Keep statements short
        - Do not explain
        - Max 5 statements
        - Return ONLY JSON
        
        Text:
        {actual_output}
        
        JSON:
        """

def test_custom_answer_relevancy_template():
    metric = AnswerRelevancyMetric(
        threshold = 0.7,
        include_reason = True,
        # verbose_mode = True,
        evaluation_template = CustomAnswerRelevancyTemplateConstrainedPrompt,
        # evaluation_template = CustomAnswerRelevancyTemplateNotConstrainedPrompt,
    )

    test_case = LLMTestCase(
        # input = "What is five divided by two?",
        # actual_output = "Five divided by two is Two point five. 5/2 is same as 20/8. Five divided by two is same as Ten divided by four",
        input = "Where can I get the best Hyderabadi Biryani in Bangalore?",
        actual_output = "Hyderabadi Chicken Biryani of Meghana Foods is pretty popular in Bangalore. But, most of the places in Bangalore do not do justice to Hyderabadi Biryani. If you want to enjoy the real Hyderabadi Biryani, you can visit Hyderabad which is around 600 kms from Bangalore."
    )

    metric.measure(test_case)
    print(f"Score: {metric.score}, Reason: {metric.reason}, Log: {metric.verbose_logs}, Statements: {metric.statements}, Verdicts: {metric.verdicts}")