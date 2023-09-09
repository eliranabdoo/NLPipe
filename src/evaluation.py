from pydantic import BaseModel
import evaluate
from typing import List


class MetricsConfig(BaseModel):
    metrics: List[str]

def get_compute_metrics_func(metric_config: MetricsConfig):
    metric = evaluate.combine([
        evaluate.load(metric) for metric in metric_config.metrics
    ])
    def compute_metrics(eval_pred):
        hypotheses, references = eval_pred
        return metric.compute(predictions=hypotheses, references=references)
    return compute_metrics