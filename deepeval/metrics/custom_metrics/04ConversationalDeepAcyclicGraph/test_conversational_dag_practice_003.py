from dotenv import load_dotenv
from ollama_model import OllamaLLM
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams
from deepeval.metrics import DeepAcyclicGraph, ConversationalDAGMetric
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalVerdictNode
)

load_dotenv()
llm_model = OllamaLLM(model='mistral')

def test_conversational_dag():
    test_case = ConversationalTestCase(
        turns = [
            Turn(role='user', content="Hey there!"),
            Turn(role='assistant', content="Get lost."),
            Turn(role='user', content="That's rude"),
            Turn(role='assistant', content="I apologize, I was having a bad moment."),
            Turn(role='user', content="It's okay, can you help me now?"),
            Turn(role='assistant', content="Of course! I'd be happy to help! What do you need? 😊"),
        ]
    )

    non_binary_node = ConversationalNonBinaryJudgementNode(
        criteria = "",
        children = [
            ConversationalVerdictNode(verdict = 'Rude', score = 10),
            ConversationalVerdictNode(verdict = 'Neutral', score = 6),
            ConversationalVerdictNode(verdict = 'Playful', score = 10),
        ],
    )

    binary_node = ConversationalBinaryJudgementNode(
        criteria = "",
        children = [
            ConversationalVerdictNode(verdict = False, score = 0),
            ConversationalVerdictNode(verdict = True, child = non_binary_node),
        ],
    )

    task_node = ConversationalTaskNode(
        instructions = "",
        output_label = 'Summary',
        evaluation_params = [TurnParams.ROLE, TurnParams.CONTENT],
        children = [binary_node],
        turn_window = (0,2), # Rude
        # turn_window = (2,5), # Apologized
    )

    dag = DeepAcyclicGraph(
        root_nodes = [task_node]
    )

    metric = ConversationalDAGMetric(
        name = "Instructions Following",
        dag = dag
    )

    evaluate(metrics = [metric], test_cases = [test_case])