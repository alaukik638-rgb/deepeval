from deepeval.metrics import ConversationalGEval
from deepeval.metrics.conversational_g_eval import ConversationalGEvalTemplate
import textwrap
from deepeval.test_case import Turn, TurnParams, ConversationalTestCase
from dotenv import load_dotenv
from ollama_model import OllamaLLM

load_dotenv()
llm_model = OllamaLLM(model="llama3:8b")
# llm_model = OllamaLLM(model="mistral")

class CustomConvoGEvalTemplate(ConversationalGEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(
            f"""
            You are given criteria for evaluating a conversation based on the following parameters: {parameters}.
            Write 3-4 clear and concise evaluation steps that describe how to judge the quality of each turn and the conversation overall.

            Criteria:
            {criteria}

            Return JSON only in the format:
            {{
                "steps": [
                    "Step 1",
                    "Step 2",
                    "Step 3"
                ]
            }}

            JSON:
            """
        )

def test_custom_conversational_gevel():
    metric = ConversationalGEval(
        name = "Professionalism",
        criteria = "Determine whether the assistant has acted professionally based on the content.",
        evaluation_template = CustomConvoGEvalTemplate,
        model = llm_model,
    )

    convo_test_case = ConversationalTestCase(
        turns = [
            Turn(role = "user", content = "What is the capital of India?"),
            Turn(role = "assistant", content = "Dum! Dum!! Dum!!! You don't even know the capital of your own country!!! The country in which you were born! The country in which you live. Wow! Just wow!"),
            Turn(role = "user", content = "Is it Calcutta?"),
            Turn(role = "assistant", content = "Is it Calcutta? Are you being serious? Is it 1900s? Is India still under British rule? Is there still a unified Bengal? Man! You don't know anything.")
        ]
    )

    metric.measure(convo_test_case)
    print(metric.score, metric.reason)


