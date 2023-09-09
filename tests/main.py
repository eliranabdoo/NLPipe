from datetime import datetime
import json
import os
from typing import Dict, Optional
from uuid import uuid4
import uuid
import hydra
from pydantic import BaseModel, Field, FieldValidationInfo, field_validator, model_validator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from preprocess import PromptingComponent
from data_loading import (
    DUMMY_DATA_PREP_CONFIG,
    DataPreparationConfig,
    load_and_preprocess_data,
)
from evaluation import MetricsConfig, get_compute_metrics_func
from model_config import DUMMY_MODEL_CONFIG, ModelConfig, load_model_and_tokenizer
from preprocess import DUMMY_PREPROCESS_PIPELINE, ProcessingPipelineConfig
from omegaconf import DictConfig

from utils import HydraConfigEncoder


class InferenceConfig(BaseModel):
    model_loading_config: ModelConfig
    preprocess_config: ProcessingPipelineConfig

DUMMY_INFERENCE_CONFIG = InferenceConfig(
    preprocess_config=ProcessingPipelineConfig.from_pipeline(DUMMY_PREPROCESS_PIPELINE),
    model_loading_config=DUMMY_MODEL_CONFIG
)
class ExperimentConfig(BaseModel):
    inference_config: InferenceConfig = Field(default_factory=lambda : DUMMY_INFERENCE_CONFIG)
    data_loading_config: DataPreparationConfig = Field(default_factory= lambda: DUMMY_DATA_PREP_CONFIG)
    metrics_config: Optional[MetricsConfig] = None # Field(default_factory= lambda: MetricsConfig(metrics=["rouge"]))
    uid: str = Field(
        default_factory=lambda: f"{datetime.now().isoformat()}-{str(uuid4())}"
    )
    output_dirs: Dict = Field(default_factory=lambda :{
       "artifacts_dir": "../../outputs/artifacts",
       "logs_dir": "../../outputs/logs" 
    })
    training_kwargs: Dict = Field(
        default_factory=lambda: dict(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            # per_device_eval_batch_size=1,
            # evaluation_strategy="steps",
            # eval_steps=1,
            logging_strategy="steps",
            logging_first_step=True,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=True,  # disable tqdm since with packing values are in correct,
        )
    )
    artifacts_dir: str
    logs_dir: str
    lora_config_kwargs: Optional[Dict] = Field(
        default_factory=lambda : dict(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
                                            )
    bnb_config_kwargs: Optional[Dict] = Field(
        default_factory=lambda: dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    )
    max_seq_length: int = 2048
    train_split: str = "train"
    test_split: Optional[str] = None
    train_on_completion_only: bool = True
    
    # @model_validator(mode="before")
    # @classmethod
    # def set_paths(cls, data):
    #     if isinstance(data, dict):
    #         for k in ["artifacts_dir", "logs_dir"]:
    #             if k not in data:
    #                 data[k] = f"outputs/{k.split('_')[0]}/{data['uid']}"
    #     return data
    
    @field_validator('artifacts_dir','logs_dir', mode='before')
    @classmethod
    def set_paths(cls, v, info: FieldValidationInfo) -> str:
        data = info.data
        outputs_dir = info.data['output_dirs'][info.field_name]
        if v is None:
            v = os.path.join(outputs_dir, data['uid'])
        return v
    

import json
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf

@hydra.main(config_path=".", config_name="config")
def run_training(cfg: DictConfig):
    cfg: ExperimentConfig = ExperimentConfig(**cfg)
    json.dump(cfg.model_dump(), open('config.json', 'w'), cls=HydraConfigEncoder)
    args = TrainingArguments(
        output_dir=cfg.artifacts_dir, logging_dir=cfg.logs_dir, **cfg.training_kwargs
    )

    pipeline = cfg.inference_config.preprocess_config.to_pipeline()
    dataset = load_and_preprocess_data(cfg.data_loading_config, pipeline)

    # TODO move bnb and lora to model config
    bnb_config = (
        BitsAndBytesConfig(**cfg.bnb_config_kwargs)
        if cfg.bnb_config_kwargs is not None
        else None
    )

    lora_config = (
        LoraConfig(**cfg.lora_config_kwargs)
        if cfg.lora_config_kwargs is not None
        else None
    )

    model_config = cfg.inference_config.model_loading_config
    if bnb_config is not None:
        model_config.loading_kwargs["quantization_config"] = bnb_config

    model, tokenizer = load_model_and_tokenizer(model_config)

    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # prepare model for training
    if lora_config is not None:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)


    trainer_kwargs = {}
    if cfg.metrics_config is not None:
        compute_metrics = get_compute_metrics_func(cfg.metrics_config)
        trainer_kwargs["compute_metrics"] = compute_metrics


    train_dataset = dataset[cfg.train_split]
    eval_dataset = dataset[cfg.test_split] if cfg.test_split is not None else None

    collator = None
    if cfg.train_on_completion_only:
        response_template = [step.label_field for step in pipeline.steps if isinstance(step, PromptingComponent)][0]
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        args=args,
        dataset_text_field=cfg.data_loading_config.dataset_config.data_adapter_config.data_adapter_result_key,
        data_collator=collator,
        **trainer_kwargs
    )
    trainer.train()

if __name__ == "__main__":
    # cfg= ExperimentConfig(artifacts_dir=None, logs_dir=None)
    # os.makedirs(cfg.logs_dir)
    # json.dump(cfg.model_dump(), open(os.path.join(cfg.logs_dir, "config.json"), "w"))
     run_training()

    
