from ollama_model import OllamaLLM
# Find a solution so that you do not have to copy ollama_model.py in each folder
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode
)
from deepeval.metrics import DAGMetric
from dotenv import load_dotenv

load_dotenv()

# ollama_llm = OllamaLLM(model = "llama3:8b")
ollama_llm = OllamaLLM(model = "mistral")

def test_deep_acyclic_graph():
    correct_order_node = NonBinaryJudgementNode(
        criteria = "Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
        children = [
            VerdictNode(verdict =  "Yes", score = 10),
            VerdictNode(verdict = "Two are out of order", score = 4),
            VerdictNode(verdict = "All out of order", score = 0),
        ],
    )

    correct_headings_node = BinaryJudgementNode(
        criteria = "Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
        children = [
            VerdictNode(verdict = True, child = correct_order_node),
            VerdictNode(verdict = False, score = 0),
        ],
    )

    extract_headings_node = TaskNode(
        instructions = "Extract all headings in `actual_output`",
        output_label = "Summary headings",
        children = [correct_headings_node, correct_order_node],
        evaluation_params  = [LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    # Create DAG
    dag = DeepAcyclicGraph(root_nodes = [extract_headings_node])

    metric = DAGMetric(
        name = "Instruction Following",
        dag = dag,
        model = ollama_llm,
        threshold = 0.5,
    )

    test_case = LLMTestCase(
        input="""
        Alice: "Today's agenda: product update, blockers, and marketing timeline. Bob, updates?"
        Bob: "Core features are done, but we're optimizing performance for large datasets. Fixes by Friday, testing next week."
        Alice: "Charlie, does this timeline work for marketing?"
        Charlie: "We need finalized messaging by Monday."
        Alice: "Bob, can we provide a stable version by then?"
        Bob: "Yes, we'll share an early build."
        Charlie: "Great, we'll start preparing assets."
        Alice: "Plan: fixes by Friday, marketing prep Monday, sync next Wednesday. Thanks, everyone!"
        """,
        actual_output="""
        Intro:
        Alice outlined the agenda: product updates, blockers, and marketing alignment.

        Body:
        Bob reported performance issues being optimized, with fixes expected by Friday. Charlie requested finalized messaging by Monday for marketing preparation. Bob confirmed an early stable build would be ready.

        Conclusion:
        The team aligned on next steps: engineering finalizing fixes, marketing preparing content, and a follow-up sync scheduled for Wednesday.
        """
    )

    evaluate(test_cases=[test_case], metrics = [metric])
    # format_correctness = DAGMetric(name = "Correctness DAG", dag = dag)
    # format_correctness.measure(test_case)
    # print(format_correctness.score)
