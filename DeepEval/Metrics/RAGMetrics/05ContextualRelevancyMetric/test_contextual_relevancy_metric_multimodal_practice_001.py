from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.test_case import LLMTestCase, MLLMImage
from dotenv import load_dotenv

load_dotenv()

def test_contextual_relevancy_multimodal_metric():
    metric = ContextualRelevancyMetric(
        threshold = 0.7,
        model = 'gpt-4.1',
        include_reason = True,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
        evaluation_template = ContextualRelevancyTemplate,
    )

    input_image_1 = MLLMImage(
        url = "https://png.pngtree.com/png-clipart/20231110/original/pngtree-polar-bear-png-image_13524684.png"
    )

    retrieval_image_1 = MLLMImage(
        url = "https://png.pngtree.com/png-clipart/20231110/original/pngtree-polar-bear-png-image_13524684.png"
    )

    input_image_2 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20231110/original/pngtree-polar-bear-png-image_13524684.png"
    )

    retrieval_image_2 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20231110/original/pngtree-polar-bear-png-image_13524684.png"
    )

    input_image_3 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20240406/original/pngtree-statue-of-liberty-on-the-transparent-background-generative-ai-png-image_14768487.png"
    )

    retrieval_image_3 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20240406/original/pngtree-statue-of-liberty-on-the-transparent-background-generative-ai-png-image_14768487.png"
    )

    input_image_4 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20250426/original/pngtree-abstract-mona-lisa-portrait-png-image_20831491.png"
    )

    retrieval_image_4 = MLLMImage(
        url="https://png.pngtree.com/png-clipart/20250426/original/pngtree-abstract-mona-lisa-portrait-png-image_20831491.png"
    )

    input_image_5 = MLLMImage(
        url="https://readyfortakeoffbook.com/cdn/shop/articles/image-hero-Boeing-747-200-B.webp"
    )

    retrieval_image_5 = MLLMImage(
        url="https://readyfortakeoffbook.com/cdn/shop/articles/image-hero-Boeing-747-200-B.webp"
    )

    input_image_6 = MLLMImage(
        url="https://png.pngtree.com/png-vector/20240810/ourmid/pngtree-how-to-choose-fresh-ingredients-for-sushi-png-image_13425238.png"
    )

    retrieval_image_6 = MLLMImage(
        url="https://png.pngtree.com/png-vector/20240810/ourmid/pngtree-how-to-choose-fresh-ingredients-for-sushi-png-image_13425238.png"
    )

    # Highly Relevant Context (Expected: High Score)
    # Expected Score: 0.9 - 1.0, Actual Score: 1.0
    test_case_1 = LLMTestCase(
        input=f"Identify this animal and describe its habitat. {input_image_1}",
        actual_output="This appears to be a polar bear living in Arctic regions.",
        retrieval_context=[
            f"Polar bears {retrieval_image_1} are large carnivorous mammals.",
            "They primarily inhabit the Arctic region.",
            "Sea ice: This is their most important habitat. Polar bears use sea ice as a platform to hunt seals, their main source of food.",
            "Arctic Ocean and coastal areas: They are found in countries bordering the Arctic, including Canada, Russia, Norway, Greenland, and the United States.",
            "Snow-covered regions: Female polar bears often create dens in snowbanks where they give birth and care for their cubs.",
            "Cold climate ecosystems: Temperatures in their habitat are typically below freezing for much of the year."
        ]
    )

    # Partially Relevant Context (Expected: Medium Score)
    # Expected Score: 0.7 - 0.8, Actual Score: 0.5
    test_case_2 = LLMTestCase(
        input=f"Identify this animal and describe its habitat. {input_image_2}",
        actual_output="This appears to be a polar bear living in Arctic regions.",
        retrieval_context=[
            f"Polar bears {retrieval_image_2} are large carnivorous mammals.",
            "They primarily inhabit the Arctic region.",
            "Bears have an excellent sense of smell.",
            "Some bear species live in forests and mountains."
        ]
    )

    # Mostly Irrelevant Context (Expected: Low Score)
    # Expected Score: 0.1 - 0.3, Actual Score: 0.25
    test_case_3 = LLMTestCase(
        input=f"What monument is shown in this image? {input_image_3}",
        actual_output="The image shows the Statue of Liberty.",
        retrieval_context=[
            f"The Great Wall of China {retrieval_image_3} stretches thousands of miles across northern China.",
            "It was built over several dynasties for defense purposes.",
            "China has many UNESCO World Heritage Sites."
        ]
    )

    # Mixed Relevant and Irrelevant Context (Expected: Medium Score)
    # Expected Score: 0.5 - 0.7, Actual Score: 0.5
    test_case_4 = LLMTestCase(
        input=f"Tell me about this famous painting. {input_image_4}",
        actual_output="This is the Mona Lisa painted by Leonardo da Vinci.",
        retrieval_context=[
            f"The Mona Lisa {retrieval_image_4} was painted by Leonardo da Vinci during the Renaissance.",
            "It is displayed in the Louvre Museum in Paris.",
            "Vincent van Gogh painted The Starry Night.",
            "Oil painting techniques evolved significantly during the 19th century."
        ]
    )

    # Contradictory Context (Expected: Low-Medium Score)
    # Expected Score: 0.4 - 0.6, Actual Score: 0.67
    test_case_5 = LLMTestCase(
        input=f"What vehicle is shown in this image? {input_image_5}",
        actual_output="The image depicts a commercial passenger airplane.",
        retrieval_context=[
            f"The image {retrieval_image_5} shows a Boeing 747, a long-range commercial aircraft.",
            "This vehicle is commonly used for transporting passengers internationally.",
            "The object in the image is a military tank designed for combat operations."
        ]
    )

    # Relevant Context Hidden Within Noise (Expected: Medium Score)
    # Expected Score: 0.5 - 0.7, Actual Score: 0.17
    test_case_6 = LLMTestCase(
        input=f"Describe the food dish shown in this image. {input_image_6}",
        actual_output="The image appears to show sushi.",
        retrieval_context=[
            "Japanese cuisine includes ramen, tempura, and udon.",
            f"Sushi {retrieval_image_6} is a Japanese dish consisting of vinegared rice combined with seafood or vegetables.",
            "Rice cultivation has been important in East Asia for centuries.",
            "Many countries have unique culinary traditions.",
            "The Michelin Guide reviews restaurants worldwide.",
            "Food presentation is valued in Japanese culture."
        ]
    )

    test_cases = [
        value for name, value in locals().items() if name.startswith('test_case_')
    ]
    # print(test_cases)

    # evaluate(test_cases, [metric])
    evaluate(test_cases = [test_case_6], metrics = [metric])