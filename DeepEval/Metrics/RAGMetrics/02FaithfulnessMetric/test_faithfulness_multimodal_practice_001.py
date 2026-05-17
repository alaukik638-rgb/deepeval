from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase, MLLMImage

load_dotenv()

def test_faithfulness_multimodal():
    metric = FaithfulnessMetric(
        threshold = 0.7,
        # model = "gpt-4.1",
        include_reason = True,
        # verbose_mode = True,
    )

    input_image = MLLMImage(
        url = "https://lh3.googleusercontent.com/gps-cs-s/APNQkAHgUJsV-WCXSujz4-VtQjEWvjuuxmEZQKl3hfRDAW0MhUGdxmDLmTOyq3ViztSsrHmQ0TSI5vKONznwGPbRAw2PNHt8hpjjYEubFb09Vdp5lBppGFF2TcdiCwrR4CJo4KxlFRzx-q73AhQ=s1360-w1360-h1020"
    )

    retrieval_image = MLLMImage(
        url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/16:9/w_2240,c_limit/GettyImages-468366251.jpg"
    )

    user_input = f"Tell me about this landmark in France: {input_image}"
    actual_output = f"This appears to be Eiffel Tower, which is a famous landmark in France"
    retrieval_context = [
        f"The Eiffel Tower {retrieval_image} is a wrought-iron lattice tower built in the late 19th century.",
        f"It is located in Champ de Mars, Paris, France.",
        f"It was designed by Gustav Eiffel in 1889."
    ]

    test_case = LLMTestCase(
        input = user_input,
        actual_output = actual_output,
        retrieval_context = retrieval_context,
    )

    # metric.measure(test_case)
    # print(metric.score, metric.reason)
    evaluate(test_cases = [test_case], metrics = [metric])
