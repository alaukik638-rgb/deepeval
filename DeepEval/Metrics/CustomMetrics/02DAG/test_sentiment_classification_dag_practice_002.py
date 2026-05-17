from ollama_model import OllamaLLM
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import DAGMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    NonBinaryJudgementNode,
    BinaryJudgementNode,
    VerdictNode
)

llm_model = OllamaLLM(model = "llama3:8b")
load_dotenv()

def test_sentiment_classification_dag():
    check_explanation_node = NonBinaryJudgementNode(
        criteria = """
        Check if the extracted sentiment contains a clear explanation
        supporting the sentiment. If detailed → Good, else → Weak.
        """,
        children = [
            VerdictNode(verdict = "Good", score = 10),
            VerdictNode(verdict = "Weak", score = 5),
        ],
    )

    check_sentiment_node = BinaryJudgementNode(
        criteria = """
        Based on the input, the correct sentiment is Positive.
        Does the extracted sentiment match this?
        """,
        children = [
            VerdictNode(verdict = True, child = check_explanation_node),
            VerdictNode(verdict = False, score = 0),
        ],
    )

    extract_sentiment_node = TaskNode(
        instructions = """
        From the actual_output, extract ONLY the sentiment (Positive or Negative).
        Return exactly one word: Positive or Negative.
        """,
        output_label = "Extracted sentiment",
        children = [check_sentiment_node],
        evaluation_params = [LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    dag = DeepAcyclicGraph(root_nodes = [extract_sentiment_node])

    metric = DAGMetric(
        name = "Check Sentiments",
        # model = llm_model,
        dag = dag
    )

    test_cases = [
        LLMTestCase(
            input="I love this product. It works perfectly!",
            actual_output = "Sentiment: Positive. The user expresses satisfaction and happiness."
        ),

        LLMTestCase(
            input="I love this product. It works perfectly!",
            actual_output="Sentiment: Positive."
        ),

        LLMTestCase(
            input="I love this product. It works perfectly!",
            actual_output="Sentiment: Negative. The user is unhappy."
        )
    ]

    test_case_1 = LLMTestCase(
        input="I love this product. It works perfectly!",
        actual_output = "Sentiment: Positive. The user expresses satisfaction and happiness."
    )

    evaluate(test_cases=test_cases, metrics = [metric])
    # metric.measure(test_case_1)
    # print(metric.score, metric.reason)
