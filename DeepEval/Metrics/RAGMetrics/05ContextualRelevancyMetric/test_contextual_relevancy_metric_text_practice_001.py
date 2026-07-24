from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

def test_contextual_relevancy_metric_text():
    metric = ContextualRelevancyMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
        strict_mode = False,
        async_mode = True,
        evaluation_template = ContextualRelevancyTemplate,
    )

    # High Relevancy - Projected Score: 1.0, Actual Score: 1.0
    test_case_HR_1 = LLMTestCase(
        input="How can I reset my account password?",
        retrieval_context = [
            "Users can reset their passwords by clicking the 'Forgot Password' link on the login page and following the email instructions."
        ],
        actual_output = "Use the Forgot Password option on the login page."
    )

    # High Relevancy - Projected Score: 0.95, Actual Score: 1.0
    test_case_HR_2 = LLMTestCase(
        input="How can I reset my account password?",
        retrieval_context = [
            "Users can reset their passwords by clicking the 'Forgot Password' link on the login page and following the email instructions."
        ],
        actual_output = "Use the Forgot Password option on the login page."
    )

    # Good Relevancy - Projected Score: 0.85, Actual Score: 1.0
    test_case_GR_1 = LLMTestCase(
        input="Can I carry extra baggage on international flights?",
        retrieval_context = [
            "Passengers may purchase additional baggage allowance before departure.",
            "Extra baggage fees vary depending on the destination."
        ],
        actual_output = "Yes, extra baggage can be purchased for an additional fee."
    )

    # Moderately High Relevancy - Projected Score: 0.75, Actual Score: 0.67
    test_case_MHR_1 = LLMTestCase(
        input="How do I apply for maternity leave?",
        retrieval_context = [
            "Employees are entitled to maternity leave benefits as outlined in the company handbook.",
            "Leave applications should be submitted through the HR portal.",
            "Managers receive automated notifications for leave requests."
        ],
        actual_output = "Submit your maternity leave request through the HR portal."
    )

    # Moderate Relevancy - Projected Score: 0.65, Actual Score: 0.5
    test_case_MR_1 = LLMTestCase(
        input="Can I update my registered mobile number online?",
        retrieval_context = [
            "Customers can update their personal details through internet banking.",
            "Address changes require additional verification."
        ],
        actual_output = "You may be able to update your mobile number through online banking."
    )

    # Moderate Relevancy - Projected Score: 0.55, Actual Score: 0.67
    test_case_MR_2 = LLMTestCase(
        input="What happens if I miss my EMI payment?",
        retrieval_context = [
            "Loan repayments are due on the specified monthly schedule.",
            "Late payment penalties may apply depending on the loan agreement.",
            "Interest rates are reviewed annually."
        ],
        actual_output = "Missing an EMI payment could result in late fees."
    )

    # Partial Relevancy - Projected Score: 0.45, Actual Score: 0.33
    test_case_PR_1 = LLMTestCase(
        input="Do health insurance plans cover dental procedures?",
        retrieval_context = [
            "Health insurance plans typically include hospitalization benefits.",
            "Preventive health checkups are covered under selected plans.",
            "Policy exclusions vary by provider."
        ],
        actual_output = "Dental coverage depends on your specific insurance plan."
    )

    # Weak Relevancy - Projected Score: 0.35, Actual Score: 0.33
    test_case_WR_1 = LLMTestCase(
        input="How can I change my seat after booking a flight?",
        retrieval_context = [
            "Passengers receive a booking confirmation email after payment.",
            "Online check-in opens 24 hours before departure.",
            "Airport lounges are available at selected locations."
        ],
        actual_output = "Seat changes can usually be made through the Manage Booking section."
    )

    # Low Relevancy - Projected Score: 0.25, Actual Score: 0.0
    test_case_LR_1 = LLMTestCase(
        input="Can students access recorded lectures?",
        retrieval_context = [
            "The university library is open from 8 AM to 10 PM.",
            "Student ID cards are required for campus entry.",
            "Academic calendars are published annually."
        ],
        actual_output = "Recorded lectures are available through the learning portal."
    )

    # Very Low Relevancy - Projected Score: 0.15, Actual Score: 0.0
    test_case_VLR_1 = LLMTestCase(
        input="How do I deactivate my credit card?",
        retrieval_context = [
            "The bank launched a new rewards program this year.",
            "Savings account interest rates have been revised.",
            "Customers can locate nearby ATMs using the mobile app."
        ],
        actual_output = "Contact customer support to deactivate your card."
    )

    # Irrelevant - Projected Score: 0.0, Actual Score: 0.0
    test_case_IR_1 = LLMTestCase(
        input="What are the side effects of this medication?",
        retrieval_context = [
            "The annual company picnic will be held in July.",
            "Employees must complete cybersecurity training.",
            "Parking permits are issued by administration."
        ],
        actual_output = "Common side effects include nausea and dizziness."
    )

    # Relevant But Noisy - Projected Score: 0.7, Actual Score: 0.5
    test_case_RBN_1 = LLMTestCase(
        input="How can I track my shipment?",
        retrieval_context = [
            "Shipment tracking numbers are emailed once orders are dispatched.",
            "Our customer support team is available 24/7.",
            "Holiday sales begin next week.",
            "Tracking information can also be found in the Orders section."
        ],
        actual_output = "Use your tracking number from the email or Orders page."
    )

    # Contradictory Retrieval - Projected Score: 0.4, Actual Score: 0.67
    test_case_CR_1 = LLMTestCase(
        input="How many paid vacation days do employees receive?",
        retrieval_context = [
            "Employees are entitled to 15 days of annual leave.",
            "Full-time employees receive 20 days of paid vacation each year.",
            "Unused leave may be carried forward with approval."
        ],
        actual_output = "Employees receive 20 days of paid vacation."
    )

    # Ambiguous Context - Projected Score: 0.5, Actual Score: 0.33
    test_case_AC_1 = LLMTestCase(
        input="Can I reschedule my appointment?",
        retrieval_context = [
            "Appointments should be confirmed 24 hours in advance.",
            "Cancellation policies vary by department.",
            "Patients can access their records online."
        ],
        actual_output = "Please contact the clinic to reschedule your appointment."
    )

    # Multiple Highly Relevant Chunks - Projected Score: 1.0, Actual Score: None
    test_case_MHRC_1 = LLMTestCase(
        input="How do I request a transcript from the university?",
        retrieval_context = [
            "Transcript requests can be submitted through the student portal.",
            "Students must clear outstanding dues before transcripts are issued.",
            "Digital transcripts are delivered within five business days."
        ],
        actual_output = "Submit your request through the student portal after clearing any dues."
    )

    test_cases = [
        value for name, value in locals().items() if name.startswith("test_case_")
    ]

    evaluate(test_cases, [metric])
    # evaluate(test_cases = [test_case_HR_1], task_completion = [metric])