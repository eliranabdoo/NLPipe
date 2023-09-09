import json
import os
import random
from typing import Dict, Optional
from uuid import uuid4
import numpy as np

from pydantic import BaseModel, Field
import torch
from tqdm import tqdm
from data_loading import DUMMY_DATA_PREP_CONFIG, load_data
from tests.main import ExperimentConfig
from model_config import ModelConfig, ModelLoadingType, load_model_and_tokenizer
from preprocess import ProcessingMode
from evaluation import MetricsConfig, get_compute_metrics_func
from utils import HydraConfigEncoder


CONFIG_PATH = ""
CKPT = 15508


class PredictionConfig(BaseModel):
    model_loading_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model_path="NousResearch/Llama-2-7b-hf",
            loading_type=ModelLoadingType.PEFT,
            loading_kwargs=dict(
                low_cpu_mem_usage=True,
                torch_dtype="float16",
                load_in_4bit=True,
            ),
        )
    )

    generation_kwargs: Dict = Field(
        default_factory=lambda: dict(
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
    )

    experiment_config: ExperimentConfig = Field(
        default_factory=lambda: ExperimentConfig.model_validate(
            json.load(
                open(
                    CONFIG_PATH,
                    "r",
                )
            )
        )
    )

    sample_size: Optional[int] = 200
    bzs: int = 1

    pred_dir: str = f"../../predictions/"
    checkpoint: str = f"checkpoint-{CKPT}"
    uid: str = Field(default_factory=lambda: str(uuid4()))


DUMMY_METRICS_CONFIG = MetricsConfig(metrics=["rouge"])
DUMMY_PREDICTION_CONFIG = PredictionConfig()


def predict(prediction_cfg: PredictionConfig, metrics_cfg: Optional[MetricsConfig]):
    os.makedirs(prediction_cfg.pred_dir, exist_ok=True)

    dataset_dict = load_data(
        prediction_cfg.experiment_config.data_loading_config.dataset_config
    )
    dataset = dataset_dict["test"]

    if prediction_cfg.sample_size is not None:
        dataset = dataset.select(
            random.sample(range(len(dataset)), k=prediction_cfg.sample_size)
        )
    data = [
        ([example[k] for k in ["html", "html_element"]], None) for example in dataset
    ]

    prediction_cfg.model_loading_config.model_path = os.path.join(
        prediction_cfg.experiment_config.artifacts_dir, prediction_cfg.checkpoint
    )
    preds = []
    pipeline = (
        prediction_cfg.experiment_config.inference_config.preprocess_config.to_pipeline()
    )
    data = pipeline.process(data, processing_mode=ProcessingMode.INFERENCE)
    model, tokenizer = load_model_and_tokenizer(prediction_cfg.model_loading_config)

    prompts = [datum[0] for datum in data]

    if prediction_cfg.bzs > 1:
        for i in range(0, len(prompts), prediction_cfg.bzs):
            prompt_batch = prompts[i : i + prediction_cfg.bzs]
            input_ids = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=prediction_cfg.experiment_config.max_seq_length,
            ).input_ids.cuda()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids, **prediction_cfg.generation_kwargs
                )

            preds_batch = [
                pred[len(prompt_batch[idx]) :]
                for idx, pred in enumerate(
                    tokenizer.batch_decode(
                        outputs.detach().cpu().numpy(), skip_special_tokens=True
                    )
                )
            ]
            preds += preds_batch
    else:
        for prompt in tqdm(prompts):
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=prediction_cfg.experiment_config.max_seq_length,
            ).input_ids.cuda()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids, **prediction_cfg.generation_kwargs
                )
            pred = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0][len(prompt) :]
            preds.append(pred)

    df = dataset.to_pandas()
    df["preds"] = preds

    preds_dir = os.path.join(prediction_cfg.pred_dir, prediction_cfg.uid)
    os.makedirs(preds_dir)
    df.to_csv(os.path.join(preds_dir, "preds.csv"))
    json.dump(
        prediction_cfg.model_dump(), open(os.path.join(preds_dir,"config.json"), "w"), cls=HydraConfigEncoder
    )

    if metrics_cfg is not None:
        compute_metrics = get_compute_metrics_func(metrics_cfg)
        metrics_result = compute_metrics(
            [df["preds"].to_list(), [[ref] for ref in df["description"].to_list()]]
        )
        json.dump(metrics_result, open(os.path.join(preds_dir,"metrics.json"), "w"))

    return preds


if __name__ == "__main__":
    preds = predict(
        prediction_cfg=DUMMY_PREDICTION_CONFIG, metrics_cfg=DUMMY_METRICS_CONFIG
    )
