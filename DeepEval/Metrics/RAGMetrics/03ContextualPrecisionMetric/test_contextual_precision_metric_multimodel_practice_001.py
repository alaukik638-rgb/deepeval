from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase, MLLMImage
from dotenv import load_dotenv

load_dotenv()

def test_contextual_precision_metric_multimodel():
    input_image = MLLMImage(
        url = "https://lh3.googleusercontent.com/gps-cs-s/APNQkAHgUJsV-WCXSujz4-VtQjEWvjuuxmEZQKl3hfRDAW0MhUGdxmDLmTOyq3ViztSsrHmQ0TSI5vKONznwGPbRAw2PNHt8hpjjYEubFb09Vdp5lBppGFF2TcdiCwrR4CJo4KxlFRzx-q73AhQ=s1360-w1360-h1020",
    )

    expected_image = MLLMImage(
        url = "https://media.architecturaldigest.com/photos/66a951edce728792a48166e6/master/pass/GettyImages-955441104.jpg",
    )

    retrieval_image = MLLMImage(
        url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/1:1/w_3633,h_3633,c_limit/GettyImages-468366251.jpg",
    )

    metric = ContextualPrecisionMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
    )

    test_case = LLMTestCase(
        input = f"Tell me about this landmark in France: {input_image}",
        actual_output = f"This appears to be Eiffel Tower, which is a famous landmark in France",
        expected_output = f"The Eiffel Tower is located in Paris, France. {expected_image}",
        retrieval_context = [
            f"The Eiffel Tower {retrieval_image} is a wrought-iron lattice tower built in the late 19th century.",
        ]
    )

    evaluate(test_cases = [test_case], metrics = [metric])