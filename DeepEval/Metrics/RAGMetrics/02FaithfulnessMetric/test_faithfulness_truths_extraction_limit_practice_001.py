from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

# Test Case 1: Low limit with simple, faithful output
def test_faithfulness_truths_extraction_limit_low():
    metric = FaithfulnessMetric(
        threshold = 0.6,
        truths_extraction_limit = 2,
    )

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris. Paris is located in the north-central part of the country.",
        retrieval_context=[
            "Paris is the capital and largest city of France. It is located on the Seine River in the north-central part of France.",
            "France is a country in Western Europe with a population of about 67 million people."
        ],
    )

    evaluate([test_case], [metric])

def test_faithfulness_truths_extraction_limit_low_hallucination():
    metric = FaithfulnessMetric(
        threshold = 0.7,
        truths_extraction_limit = 2,
    )

    test_case = LLMTestCase(
        input="What is the population of Tokyo?",
        actual_output="Tokyo has a population of 14 million people. Tokyo is the smallest city in Japan. The city was founded in 3000 BC.",
        retrieval_context=[
            "Tokyo is the capital of Japan with a metropolitan population of approximately 37 million people.",
            "The Greater Tokyo Area is the most populous metropolitan area in the world.",
            "Tokyo is the largest city in Japan by population and area."
        ],
    )

    evaluate(test_cases = [test_case], metrics = [metric])
    # Should fail due to incorrect population figure

# Test Case 3: Higher limit with complex, multi-fact output
def test_faithfulness_truths_extraction_limit_high_complex():
    """ Test extraction limit with output containing many claims"""
    metric = FaithfulnessMetric(
        threshold = 0.6,
        truths_extraction_limit = 5,
    )

    test_case = LLMTestCase(
        input="Tell me about the Python programming language",
        actual_output="""Python is a high-level programming language created by Guido van Rossum. 
                It was first released in 1991. Python emphasizes code readability with significant whitespace. 
                It supports multiple programming paradigms including procedural, object-oriented, and functional programming. 
                Python is widely used in data science, web development, and automation.""",
        retrieval_context=[
            "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991.",
            "Python's design philosophy emphasizes code readability with its use of significant indentation.",
            "Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.",
            "Python is commonly used for web development, data analysis, artificial intelligence, and scientific computing.",
            "The language features dynamic typing and automatic memory management."
        ],
    )
    metric.measure(test_case)
    print(f"Test 3 - Score: {metric.score}, Reason: {metric.reason}")

# Test Case 4: Limit=1 vs Limit=10 comparison
def test_faithfulness_truths_extraction_limit_comparison():
    """Compare scores with different extraction limits"""

    shared_test_case_data = {
        "input": "What are the main features of electric cars?",
        "actual_output": """Electric cars run on electric motors powered by rechargeable batteries. 
        They produce zero direct emissions, making them environmentally friendly. 
        Electric vehicles have lower operating costs compared to gasoline cars. 
        Most modern electric cars have a range of 200-300 miles per charge. 
        They can be charged at home or at public charging stations.""",
        "retrieval_context": [
            "Electric vehicles (EVs) use electric motors instead of internal combustion engines.",
            "EVs are powered by rechargeable lithium-ion battery packs.",
            "Electric cars produce zero tailpipe emissions, contributing to cleaner air in urban areas.",
            "The average range of modern electric vehicles is between 250-350 miles on a single charge.",
            "Electric vehicles have lower fuel and maintenance costs compared to traditional gasoline vehicles.",
            "EVs can be charged at home using a standard outlet or dedicated charging station.",
            "Public charging infrastructure is expanding rapidly across many countries.",
            "Electric motors provide instant torque, resulting in quick acceleration."
        ]
    }

    test_case = LLMTestCase(**shared_test_case_data)

    # Test with limit = 2
    metric_low = FaithfulnessMetric(threshold = 0.6, truths_extraction_limit = 2)
    metric_low.measure(test_case)

    # Test with limit  = 8
    metric_high = FaithfulnessMetric(threshold = 0.6, truths_extraction_limit = 8)
    metric_high.measure(test_case)

    print(f"\nLimit = 2 - Score: {metric_low.score}")
    print(f"Limit = 8 - Score: {metric_high.score}")
    print(f"Difference: {abs(metric_high.score - metric_low.score)}")

# Test Case 5: Edge case with more claims than extraction limit
def test_faithfulness_more_claims_than_limit():
    """Tests behavior when actual output has more claims than extraction limit"""
    metric = FaithfulnessMetric(
        threshold = 0.5,
        truths_extraction_limit = 3,
    )

    test_case = LLMTestCase(
        input="Describe the solar system",
        actual_output="""The solar system consists of the Sun and eight planets. 
                Mercury is the closest planet to the Sun. Venus is the hottest planet. 
                Earth is the only planet known to support life. Mars is called the Red Planet. 
                Jupiter is the largest planet. Saturn has prominent rings. 
                Uranus rotates on its side. Neptune is the farthest planet from the Sun.""",
        retrieval_context=[
            "The solar system contains the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "Mercury is the innermost planet, orbiting closest to the Sun at an average distance of 58 million kilometers.",
            "Venus has the hottest surface temperature of any planet due to its thick atmosphere and greenhouse effect.",
            "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "Mars, the fourth planet, appears reddish due to iron oxide on its surface, earning it the nickname 'Red Planet'.",
            "Jupiter is the largest planet in the solar system with a mass more than twice that of all other planets combined.",
            "Saturn is known for its extensive ring system made of ice particles and rocky debris.",
            "Uranus has a unique axial tilt of 98 degrees, causing it to rotate nearly on its side.",
            "Neptune is the eighth and farthest known planet from the Sun in our solar system."
        ],
    )

    metric.measure(test_case)
    print(f"Test 5 - Score: {metric.score}, Reason: {metric.reason}")
