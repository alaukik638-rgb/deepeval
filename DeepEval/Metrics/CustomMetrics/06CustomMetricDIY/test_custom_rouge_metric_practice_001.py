from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        self.score = self.scorer.rouge_score(
            prediction = test_case.actual_output,
            target = test_case.expected_output,
            score_type = 'rouge1',
            # score_type = 'rouge2',
            # score_type = 'rougeL',
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

def test_rouge_metric():
    metric = RougeMetric()
    test_case = LLMTestCase(
        input = "What is 5 divided by 2?",
        actual_output = '2.5',
        expected_output = "The result is 2.5"
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.is_successful())