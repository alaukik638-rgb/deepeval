from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics.contextual_recall import ContextualRecallTemplate
from deepeval.test_case import LLMTestCase, MLLMImage
from dotenv import load_dotenv
load_dotenv()

def test_contextual_recall_multimodel():
    input_image_car = MLLMImage(
        url = "https://cdn.pixabay.com/photo/2018/10/09/08/20/auto-3734396_1280.jpg",
    )

    expected_url = MLLMImage(
        url = "https://www.shutterstock.com/image-photo/damaged-silver-car-scratches-on-260nw-210881749.jpg",
    )

    retrieval_image = MLLMImage(
        url = "https://cdn.pixabay.com/photo/2018/10/09/08/20/auto-3734396_1280.jpg",
    )

    input_image_lungs = MLLMImage(
        url = "https://images.unsplash.com/photo-1581595219315-a187dd40c322",
    )


    metric = ContextualRecallMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
        evaluation_template = ContextualRecallTemplate,
    )

    test_case_car_damage = LLMTestCase(
        input = f"Based on the this image: {input_image_car}, what damages are visible on the vehicle?",
        expected_output = f"The vehicle has front bumper damage, a broken headlight, and visible dents on the front side.",
        retrieval_context = [
            f"""
            Inspection notes:
            - Front bumper is cracked
            - Left headlight is broken
            - Dent observed near the front fender
            - Rear side appears undamaged
            """
        ]
    )

    test_case_lungs_damage = LLMTestCase(
        input= f"Analyze the uploaded chest X-ray and summarize the findings: {input_image_lungs}",
        expected_output= "The X-ray indicates opacity in the lower left lung, suggesting pneumonia or infection.",
        retrieval_context=[
            """
            Radiology report:
            - No pleural effusion observed
            """,
            """
            Radiology report:
            - Mild opacity detected in left lower lung region
            - Findings may indicate pneumonia
            """
        ],
    )

    # evaluate(test_cases = [test_case_car_damage], task_completion = [metric])
    evaluate(test_cases=[test_case_lungs_damage], metrics=[metric])