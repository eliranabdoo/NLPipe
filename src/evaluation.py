from pydantic import BaseModel
import evaluate
from typing import List

from src.preprocess import ProcessingPipeline


class MetricsConfig(BaseModel):
    metrics: List[str]

def get_compute_metrics_func(metric_config: MetricsConfig, postprocess_pipeline: ProcessingPipeline):
    metric = evaluate.combine([
        evaluate.load(metric) for metric in metric_config.metrics
    ])
    def compute_metrics(eval_pred):
        hypotheses, references = eval_pred
        hypotheses = postprocess_pipeline.process(hypotheses)
        return metric.compute(predictions=hypotheses, references=references)
    return compute_metrics