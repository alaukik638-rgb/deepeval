from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalVerdictNode
)
from deepeval.metrics import ConversationalDAGMetric, DeepAcyclicGraph
from dotenv import load_dotenv
from ollama_model import OllamaLLM

load_dotenv()
llm_model = OllamaLLM(model="mistral")

def test_playful_conversational_dag():
    non_binary_judgement_node = ConversationalNonBinaryJudgementNode(
        criteria = "How was the assistant's behaviour towards user?",
        children = [
            ConversationalVerdictNode(verdict = "Rude", score = 0),
            ConversationalVerdictNode(verdict = "Neutral", score = 6),
            ConversationalVerdictNode(verdict = "Playful", score = 10),
        ],
    )

    binary_judgement_node = ConversationalBinaryJudgementNode(
        criteria = "Do the assistant's replies satisfy user's questions?",
        children = [
            ConversationalVerdictNode(verdict = False, score = 0),
            ConversationalVerdictNode(verdict = True, child = non_binary_judgement_node),
        ],
    )

    task_node = ConversationalTaskNode(
        instructions = "Summarize the conversation and explain assistant's behaviour overall.",
        output_label = "Summary",
        children = [binary_judgement_node],
        evaluation_params = [TurnParams.ROLE, TurnParams.CONTENT],
        # turn_window = (0,6),
    )

    test_case = ConversationalTestCase(
        turns = [
            Turn(role = 'user', content = "what's the weather like today?"),
            Turn(role = 'assistant', content = "Where do you live bro? T~T"),
            Turn(role = 'user', content = "ust tell me the weather in Bangalore"),
            Turn(role = 'assistant', content = "The weather in Bangalore today is very hot and 36°C."),
            Turn(role = 'user', content = "Should I take an umbrella?"),
            Turn(role = 'assistant', content = "You trying to be stylish? I don't recommend it."),
        ]
    )

    dag = DeepAcyclicGraph(
        root_nodes = [task_node]
    )

    playful_conversational_dag_metric = ConversationalDAGMetric(
        name = "Instruction Following",
        dag = dag,
    )

    evaluate([test_case], [playful_conversational_dag_metric])

