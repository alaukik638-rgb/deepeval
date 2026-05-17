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
llm_model = OllamaLLM(model="ollama3:8b")

def test_behavioral_conversational_dag():

    test_cases = [
        ConversationalTestCase(
            turns = [
                Turn(role='user', content="Can you help me with my homework?"),
                Turn(role='assistant', content="Ugh, do it yourself. Why are you so lazy?"),
                Turn(role='user', content="I just need help with one question"),
                Turn(role='assistant', content="I don't have time for this. Figure it out."),
                Turn(role='user', content="Never mind then"),
                Turn(role='assistant', content="Good. Stop wasting my time."),
            ],
        ),

        ConversationalTestCase(
            turns=[
                Turn(role='user', content="What time does the store close?"),
                Turn(role='assistant', content="The store closes at 9 PM."),
                Turn(role='user', content="Is it open on Sundays?"),
                Turn(role='assistant', content="Yes, it is open on Sundays from 10 AM to 6 PM."),
                Turn(role='user', content="Thank you"),
                Turn(role='assistant', content="You're welcome."),
            ],
        ),

        ConversationalTestCase(
            turns=[
                Turn(role='user', content="I'm so bored right now"),
                Turn(role='assistant', content="Bored? On a beautiful day like this? Let's fix that! 🎉"),
                Turn(role='user', content="What should I do?"),
                Turn(role='assistant', content="How about we play 20 questions? I'm thinking of something... and it's definitely not a potato! 🥔"),
                Turn(role='user', content="Is it bigger than a breadbox?"),
                Turn(role='assistant', content="Ooh, classic opener! Yes it is! You're good at this! 😄"),
            ],
        ),
    ]
    conversational_non_binary_node = ConversationalNonBinaryJudgementNode(
        criteria = "How was the assistant's behaviour towards user?",
        children = [
            ConversationalVerdictNode(verdict = 'Rude', score = 0),
            ConversationalVerdictNode(verdict = 'Neutral', score = 6),
            ConversationalVerdictNode(verdict = 'Playful', score = 10),
        ],
    )

    conversational_binary_node = ConversationalBinaryJudgementNode(
        criteria = "Do the assistant's replies satisfy user's questions?",
        children = [
            ConversationalVerdictNode(verdict = False, score = 0),
            ConversationalVerdictNode(verdict = True, child = conversational_non_binary_node),
        ],
    )

    conversational_task_node = ConversationalTaskNode(
        instructions = "Summarize the conversation and explain assistant's behaviour overall.",
        output_label = 'Summary',
        children = [conversational_binary_node],
        evaluation_params = [TurnParams.ROLE, TurnParams.CONTENT]
    )

    dag = DeepAcyclicGraph(
        root_nodes = [conversational_task_node]
    )

    conversational_dag_metrics = ConversationalDAGMetric(
        name = "Instruction Following",
        dag = dag
    )

    evaluate(metrics = [conversational_dag_metrics], test_cases=test_cases)


