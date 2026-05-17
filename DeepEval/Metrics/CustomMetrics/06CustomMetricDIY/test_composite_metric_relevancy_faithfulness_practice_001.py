from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class FaithfulRelevancyMetric(BaseMetric):
    def __init__(
            self,
            threshold: float = 0.5,
            evaluation_model: Optional[str] = "gpt-4.1",
            include_reason: bool = True,
            async_mode: bool = True,
            strict_mode: bool = False
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        try:
            relevancy_metric, faithfulness_metric = self.initialize_metrics()
            # Remember, deepeval's default metrics follow the same pattern as your custom metric!
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)

            # Custom logic to set score, reason, and success
            self.set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        try:
            relevancy_metric, faithfulness_metric = self.initialize_metrics()
            await relevancy_metric.a_measure(test_case)
            await faithfulness_metric.a_measure(test_case)
            self.set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        else:
            return self.success

    @property
    def __name__(self):
        return "Composite Relevancy Faithfulness Metric"

    ## Helper Methods ##
    def initialize_metrics(self):
        relevancy_metric = AnswerRelevancyMetric(
            threshold = self.threshold,
            model = self.evaluation_model,
            include_reason = self.include_reason,
            async_mode = self.async_mode,
            strict_mode = self.strict_mode,
        )

        faithfulness_metric = FaithfulnessMetric(
            threshold = self.threshold,
            model = self.evaluation_model,
            include_reason = self.include_reason,
            async_mode = self.async_mode,
            strict_mode = self.strict_mode,
        )

        return relevancy_metric, faithfulness_metric

    def set_score_reason_success(
            self,
            relevancy_metric: BaseMetric,
            faithfulness_metric: BaseMetric
    ):
        relevancy_score = relevancy_metric.score
        relevancy_reason = relevancy_metric.reason
        faithfulness_score = faithfulness_metric.score
        faithfulness_reason = faithfulness_metric.reason

        # Custom logic to set score
        composite_score = min(relevancy_score, faithfulness_score)
        self.score = 0 if self.strict_mode and composite_score < self.threshold else composite_score

        # Custom logic to set reason
        if self.include_reason:
            self.reason = relevancy_reason + "\n" + faithfulness_reason

        # Custom logic to set success
        self.success = self.score >= self.threshold

def test_composite_metric():
    metric = FaithfulRelevancyMetric()
    test_case = LLMTestCase(
        input = "What is the capital of India?",
        actual_output = 'New Delhi',
        expected_output = "The capital of India is New Delhi",
        retrieval_context = ["New Delhi is the capital of India"]
    )

    assert_test(test_case, [metric])