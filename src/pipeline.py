from dataclasses import dataclass
from enum import Enum
import importlib
from typing import Any, Dict, Generic, Literal, Optional, Tuple, TypeVar

from pydantic import BaseModel
import torch
from .preprocess import ProcessingPipeline

from abc import abstractclassmethod, ABCMeta, abstractmethod


class NLConfigMeta(type):
    def __init__(cls, name, bases, attrs):
        required_field = getattr(cls, "kind", None)
        if required_field is not None and required_field not in attrs:
            raise ValueError(f"{required_field} is a required field in {cls.__name__}")
        super().__init__(name, bases, attrs)


class NLConfig(BaseModel):
    kind: str


class NLComponent(metaclass=ABCMeta):
    @abstractclassmethod
    def from_config(cls, config: NLConfig) -> "NLComponent":
        pass


NLModel = TypeVar("NLModel", bound=NLComponent)


class NLPipeline(Generic[NLModel], NLComponent, metaclass=ABCMeta):
    preprocessing_pipeline: ProcessingPipeline
    model: NLModel
    postprocessing_pipeline: ProcessingPipeline


Data = TypeVar("Data")


@dataclass
class NLDataContainer(Generic[Data], NLComponent, metaclass=ABCMeta):
    train: Data
    val: Optional[Data]
    test: Optional[Data]


class NLDataCooker(Generic[Data], NLComponent, metaclass=ABCMeta):
    processing_pipeline: ProcessingPipeline

    def cook(
        self, dataset: NLDataContainer, train: bool, val: bool, test: bool
    ) -> None:
        if train:
            dataset.train = self._cook_data(train)
        if val:
            dataset.val = self._cook_data(val)
        if test:
            dataset.val = self._cook_data(test)

    def _cook_data(self, data: Data):
        return self.processing_pipeline.process(data)

    


class NLPipelineOperator(Generic[Data], metaclass=ABCMeta):
    @abstractmethod
    def execute(self, pipeline: NLPipeline, data_mapping: Dict[str, Data]):
        pass


class NLExperimentOutcome(metaclass=ABCMeta):
    pass


class NLExperiment(NLComponent, metaclass=ABCMeta):
    pipeline: NLPipeline
    data_cooker: NLDataCooker
    task: NLPipelineOperator

    def run(
        self,
        *,
        data_container: NLDataContainer,
    ) -> NLExperimentOutcome:
        cooked_data_container = self.data_cooker.cook(data_container)
        self.task.execute(self.pipeline, cooked_data_container)


from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments


class HuggingfaceLoadingModules(str, Enum):
    TRANSFORMERS = "transformers"
    PEFT = "peft"


class HuggingfaceNLModelConfig(NLConfig):
    kind: Literal["hf"] = "hf"
    model_path: str
    loading_module: HuggingfaceLoadingModules
    loading_cls: str
    loading_func: str = "from_pretrained"
    loading_kwargs: Dict
    bnb_config: Optional[Dict]
    bnb_config_key: str = "quantization_config"
    model_config_attr_setting: Optional[Dict]
    lora_config: Optional[Dict]


NLModelConfig = HuggingfaceNLModelConfig


@dataclass
class HuggingfaceNLModel(NLModel):
    model: AutoModel
    tokenizer: AutoTokenizer

    def from_config(cls, cfg: HuggingfaceNLModelConfig) -> "HuggingfaceNLModel":
        loading_module = importlib.import_module(cfg.loading_module.value)
        loading_func = getattr(
            getattr(loading_module, cfg.loading_cls), cfg.loading_func
        )

        try:
            cfg.loading_kwargs["torch_dtype"] = getattr(
                torch, cfg.loading_kwargs["torch_dtype"]
            )
        except KeyError:
            pass

        bnb_config = (
            BitsAndBytesConfig(**cfg.bnb_config) if cfg.bnb_config is not None else None
        )


        model_loading_kwargs = cfg.loading_kwargs

        if bnb_config is not None:
            if cfg.bnb_config_key in model_loading_kwargs:
                raise ValueError("Key collision")
            model_loading_kwargs[cfg.bnb_config_key] = bnb_config

        model = loading_func(cfg.model_path, **cfg.loading_kwargs)
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_config = (
            LoraConfig(**cfg.lora_config) if cfg.lora_config is not None else None
        )

        if lora_config is not None:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

        return cls(model=model, tokenizer=tokenizer)

from datasets import Dataset as HFDataset

class HuggingFaceNLTrainer(NLPipelineOperator[HFDataset], metaclass=ABCMeta):
    train_split = "train"
    eval_split = "eval"
    def execute(self, pipeline: NLPipeline[HuggingfaceNLModel], data_mapping: Dict[str, HFDataset]):
        if set(data_mapping.keys()).issubset({self.train_split, self.eval_split}):
            raise ValueError("invalid data mapping keys")

        model, tokenizer = pipeline.model.model, pipeline.model.tokenizer
        
        trainer_kwargs = {}
        if cfg.metrics_config is not None:
            compute_metrics = get_compute_metrics_func(cfg.metrics_config)
            trainer_kwargs["compute_metrics"] = compute_metrics


        train_dataset = data_mapping[self.train_split]
        eval_dataset = data_mapping.get(self.eval_split, None)

        collator = None
        if cfg.train_on_completion_only:
            response_template = [step.label_field for step in pipeline.steps if isinstance(step, PromptingComponent)][0]
            collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  

        args = TrainingArguments(
            output_dir=cfg.artifacts_dir, logging_dir=cfg.logs_dir, **cfg.training_kwargs
        )


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




class HuggingFaceNLTrainer(NLPipelineOperator):
    pass


class HuggingFaceNLGenerator(NLPipelineOperator):
    pass


# class HuggingfaceModelPatchCatalog(str, Enum):
#     LLAMA2= "llama2"


# @dataclass
# class HuggingfaceModelPatch:
#     model_patch: Dict
#     tokenizer_patch: Dict

# class HuggingfaceModelPatchFactory:
#     def from_catalog(self, patch_type: HuggingfaceModelPatchCatalog) -> HuggingfaceModelPatch:
#         if patch_type == HuggingfaceModelPatchCatalog.LLAMA2:
#             return HuggingfaceModelPatch(model_patch={"config.pretraining_tp": 1},
#                                          tokenizer_patch={"pad_token"})
