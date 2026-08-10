from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

def test_contextual_recall_metric_text():
    metric = ContextualRecallMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
    )

    # Relevant Information Fully Retrieved
    test_case_full_retrieval = LLMTestCase(
        input = "What are the side effects of Drug X",
        expected_output = "Drug X may cause nausea, dizziness, headache, and fatigue.",
        retrieval_context = [
            "Common side effects of Drug X include headache, nausea, dizziness, and fatigue.",
        ]
    )

    # Partial Context Retrieval
    test_case_partial_retrieval = LLMTestCase(
        input = "What are the eligibility criteria for Policy A?",
        expected_output = "Applicants must be above 18 years old, have valid ID proof, and maintain a minimum balance of ₹10,000.",
        retrieval_context = [
            # "Applicants must be above 18 years old and provide valid ID proof.",
            # "Applicants must be above 18 years old, have valid ID proof, and maintain a minimum balance of ₹10,000."
        ]
    )

    # Missing Critical Information
    test_case_missing_critical_info = LLMTestCase(
        input= "What documents are required for claim reimbursement?",
        expected_output= "Hospital bills, discharge summary, ID proof, and prescription copies are mandatory.",
        retrieval_context= [
            "Claim reimbursement requires hospital bills and ID proof.",
        ]
    )

    # Irrelevant Context with Small Relevant Portion
    test_case_irrelevant_context = LLMTestCase(
        input= "What is the cancellation policy for premium subscriptions?",
        expected_output= "Users can cancel within 14 days for a full refund.",
        retrieval_context= [
            "Premium subscriptions include ad-free access and offline downloads. Users may cancel within 14 days for a full refund. Renewal is automatic unless disabled.",
        ]
    )

    # Contradictory / Incorrect Retrieval
    test_case_contradictory_retrieval = LLMTestCase(
        input="When is the assignment submission deadline?",
        expected_output="The assignment must be submitted by September 15.",
        retrieval_context=[
            "Assignments are due on September 20.",
        ]
    )

    # metric.measure(test_case_full_retrieval)
    # print(f"Full Relevant Score: {metric.score}")
    # print(f"Full Relevant Reason: {metric.reason}")
    #
    # metric.measure(test_case_partial_retrieval)
    # print(f"Partial Relevant Score: {metric.score}")
    # print(f"Partial Relevant Reason: {metric.reason}")
    #
    # metric.measure(test_case_missing_critical_info)
    # print(f"Missing Critical Info Score: {metric.score}")
    # print(f"Missing Critical Info Reason: {metric.reason}")
    #
    # metric.measure(test_case_irrelevant_context)
    # print(f"Irrelevant with small relevance Score: {metric.score}")
    # print(f"Irrelevant with small relevance Reason: {metric.reason}")
    #
    # metric.measure(test_case_contradictory_retrieval)
    # print(f"Contradictory relevance Score: {metric.score}")
    # print(f"Contradictory relevance Reason: {metric.reason}")

    # Single test case score evaluation
    evaluate(test_cases = [test_case_partial_retrieval], metrics = [metric])

    # All test case score evaluation
    # evaluate(test_cases=[test_case_full_retrieval, test_case_partial_retrieval, test_case_missing_critical_info, test_case_irrelevant_context, test_case_contradictory_retrieval], task_completion=[metric])
