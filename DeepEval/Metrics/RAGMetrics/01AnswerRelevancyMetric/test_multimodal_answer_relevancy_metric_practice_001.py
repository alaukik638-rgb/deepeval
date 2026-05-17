from deepeval import evaluate
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, MLLMImage

load_dotenv()

def test_multimodal_answer_relevancy_metric():
    metric = AnswerRelevancyMetric(
        threshold = 0.7,
        model = "gpt-4.1",
        include_reason = True,
        verbose_mode = True,
    )

    image = MLLMImage(
        # url = "https://media.cntraveler.com/photos/58de89946c3567139f9b6cca/16:9/w_2240,c_limit/GettyImages-468366251.jpg",
        # url = "https://media.architecturaldigest.com/photos/67acb9b0339bcbaaadeb91b5/16:9/w_2240,c_limit/GettyImages-873536102.jpg",
        url = "https://lh3.googleusercontent.com/gps-cs-s/APNQkAHgUJsV-WCXSujz4-VtQjEWvjuuxmEZQKl3hfRDAW0MhUGdxmDLmTOyq3ViztSsrHmQ0TSI5vKONznwGPbRAw2PNHt8hpjjYEubFb09Vdp5lBppGFF2TcdiCwrR4CJo4KxlFRzx-q73AhQ=s1360-w1360-h1020",
    )

    test_case = LLMTestCase(
        input = f"Tell me about this landmark in France: {image}",
        actual_output = f"This appears to be Eiffel Tower, which is a famous landmark in France"
    )

    evaluate([test_case], [metric])