from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from functools import reduce
from difflib import SequenceMatcher
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# Length based
def generate_hypothetical_score_length(test_case) -> float | None:
    output_length = len(test_case.actual_output)
    print(f"Output length is: {output_length}")
    if output_length <= 10:
        return 0.3
    elif output_length <= 50:
        return 0.7
    else:
        return 0.9

def generate_hypothetical_score_keyword(test_case) -> float | None:
    keywords = ['result', 'answer', 'solution']
    count = sum(1 for kw in keywords if kw in test_case.actual_output.lower())
    print(f"Count of keywords is: {count}")
    return min(count / len(keywords), 1)

def generate_hypothetical_score_simple_match(test_case) -> float | None:
    return 1 if test_case.actual_output in test_case.expected_output or test_case.expected_output in test_case.actual_output else 0

def generate_hypothetical_score_similarity_ratio(test_case) -> float | None:
    similarity = SequenceMatcher(
        None, test_case.actual_output.lower(), test_case.expected_output.lower()
    ).ratio()
    print(f"Similarity Ratio: {similarity}")
    return similarity

def generate_hypothetical_reason_similarity_ratio(test_case) -> str | None:
    score = generate_hypothetical_score_similarity_ratio(test_case)
    if score > 0.8:
        return "High similarity between expected and actual output"
    elif score > 0.5:
        return "Moderate similarity, some differences found"
    else:
        return "Low similarity, outputs differ significantly"


# 5. Numeric answer extraction and comparison
def generate_hypothetical_score_numerical_comparison(test_case) -> float | None:
    expected_nums = re.findall(pattern = r'-?\d+\.?\d*', string = test_case.expected_output, flags = 0)
    actual_nums = re.findall(r'-?\d+\.?\d*', test_case.actual_output)

    if not expected_nums or not actual_nums:
        return 0.5

    expected_val = float(expected_nums[0])
    actual_val = float(actual_nums[0])
    print(f"Numerical Comparison - Expected value is: {expected_val} and Actual value is: {actual_val}")
    # Check if within 5% tolerance
    tolerance = abs(expected_val * 0.05)
    if abs(expected_val - actual_val) <= tolerance:
        return 1
    else:
        return max(0.0, 1.0 - abs(expected_val - actual_val)/expected_val)

# 6. Sentiment-based scoring (requires: pip install textblob)
def generate_hypothetical_score_sentiment_based(test_case) -> float | None:
    # Sentiment polarity requires emotional words in output
    # Try with actual_output as "This is amazing/terrible"
    blob = TextBlob(test_case.actual_output)
    print(f"Sentiment - Blob value is: {blob}")
    polarity = blob.sentiment.polarity # -1 to 1
    # Normalize between 0 - 1
    print(f"Sentiment - Polarity is: {polarity}")
    return (polarity + 1)/2

# 7. Embedding similarity (requires: pip install sentence-transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_hypothetical_score_embedding_similarity(test_case) -> float | None:
    exp_emb = model.encode(test_case.expected_output, convert_to_tensor = True)
    act_emb = model.encode(test_case.actual_output, convert_to_tensor = True)
    # print(f"Embedding similarity - Expected Embedding: {exp_emb} and Actual Embedding: {act_emb}")
    similarity = util.cos_sim(exp_emb, act_emb).item()
    print(f"Embedding similarity - Similarity: {similarity}")
    return max(0, min(1, similarity))

# 8. Multi-criteria weighted scoring
def generate_hypothetical_score_multi_criteria_weighted(test_case) -> float | None:
    # Criterion 1: Length appropriateness (30%)
    exp_len = len(test_case.expected_output.split())
    act_len = len(test_case.actual_output.split())
    len_score = 1.0 - min(abs(exp_len - act_len)/max(exp_len, 1), 1.0)
    print(f"Length score is: {len_score}")

    # Criterion 2: Keyword coverage (30%)
    exp_words = set(test_case.expected_output.lower().split())
    act_words = set(test_case.actual_output.lower().split())
    keyword_score = len(exp_words & act_words) / len(exp_words) if exp_words else 0
    print(f"Keyword Score is: {keyword_score}")

    # Criterion 3: Textual similarity (40%)
    similarity = SequenceMatcher(None, test_case.actual_output, test_case.expected_output).ratio()
    print(f"Similarity is: {similarity}")

    final_score = (len_score * 0.3) + (keyword_score * 0.3) + (similarity * 0.4)
    return final_score



def generate_hypothetical_reason(test_case) -> str | None:
    output_length = len(test_case.actual_output)
    if output_length <= 10:
        return "Output length is less than 10"
    elif output_length <= 50:
        return "Output length is less than 50"
    else:
        return "Output length is more than 50"


async def async_generate_hypothetical_score(test_case) -> float | None:
    # return generate_hypothetical_score(test_case)
    pass


async def async_generate_hypothetical_reason(test_case) -> str | None:
    # return generate_hypothetical_reason(test_case)
    pass

SCORING_FUNCTIONS = [
    generate_hypothetical_score_length,
    generate_hypothetical_score_keyword,
    generate_hypothetical_score_simple_match,
    generate_hypothetical_score_multi_criteria_weighted,
    generate_hypothetical_score_similarity_ratio,
    generate_hypothetical_score_numerical_comparison,
    generate_hypothetical_score_sentiment_based,
    generate_hypothetical_score_embedding_similarity,
    # Add other score functions
]

def generate_final_hypothetical_score(test_case) -> float:
    scores = []
    for func in SCORING_FUNCTIONS:
        try:
            score = func(test_case)
            if score is not None:
                scores.append(score)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")

    if not scores:
        return 0.0

    final_score = sum(scores) / len(scores)
    print(f"Individual scores: {scores}")
    print(f"Final score: {final_score}")

    return final_score

def test_custom_metric_single_turn():
    test_case = LLMTestCase(
        input="What is 5 divided by 2?",
        expected_output='2.5',
        actual_output='The result is 2.5'
    )

    test_metric = CustomMetric()

    assert_test(test_case, [test_metric])

class CustomMetric(BaseMetric):
    def __init__(
            self,
            threshold: float = 0.4,
            # evaluation_model: str = 'gpt-4.1',
            include_reason: bool = True,
            strict_mode: bool = False,
            async_mode: bool = False
    ):
        self.threshold = threshold
        # self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.score = None
        self.reason = None
        self.success = None
        self.error = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            self.score = generate_final_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            self.score = generate_final_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except (TypeError, AttributeError):
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Alaukik's Custom Metric 01"