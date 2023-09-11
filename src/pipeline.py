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
        required_field = getattr(cls, 'kind', None)
        if required_field is not None and required_field not in attrs:
            raise ValueError(f"{required_field} is a required field in {cls.__name__}")
        super().__init__(name, bases, attrs)


class NLConfig(BaseModel):
    kind: str

class NLComponent(metaclass=ABCMeta):
    @abstractclassmethod
    def from_config(cls, config: NLConfig) -> "NLComponent":
        pass

NLModel = TypeVar('NLModel', bound=NLComponent)

class NLPipeline(NLComponent, metaclass=ABCMeta):
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

    def cook(self, dataset: NLDataContainer,
             train: bool,
             val: bool,
             test: bool) -> None:
        if train:
            dataset.train = self._cook_data(train)
        if val:
            dataset.val = self._cook_data(val)
        if test:
            dataset.val = self._cook_data(test)
    
    def _cook_data(self, data: Data):
        return self.processing_pipeline.process(data)

class NLExperimentOutcome(metaclass=ABCMeta):
    pass

class NLTask(Generic[Data], metaclass=ABCMeta):
    @abstractmethod
    def execute(self, pipeline: NLPipeline, data_container: NLDataContainer[Data]):
        pass

class NLTrainer(Generic[NLModel], NLComponent, metaclass=ABCMeta):
    @abstractmethod
    def train(self, pipeline: NLPipeline, data_container: NLDataContainer):
        pass

class NLTrainingTask(NLTask):
    trainer: NLTrainer

    def execute(self, pipeline: NLPipeline, data_container: NLDataContainer):
        return self.trainer.train(pipeline=pipeline, data_container=data_container)


class NLPredictionTask(NLTask):
    pass


class NLInference:
    pipeline: NLPipeline

    def from_experiment(experiment_outcome: NLExperimentOutcome) -> "NLInference":
        pass

class NLExperiment(NLComponent, metaclass=ABCMeta):
    pipeline: NLPipeline
    data_cooker: NLDataCooker
    task: NLTask
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

class HuggingfaceNLModelConfig(metaclass=NLConfigMeta):
    kind: Literal['hf'] = 'hf'
    model_path: str
    loading_module: HuggingfaceLoadingModules
    loading_cls: str
    loading_func: str = "from_pretrained"
    loading_kwargs: Dict
    bnb_config: Dict
    lora_config: Dict
    bnb_config_key: str = "quantization_config"
    model_config_attr_setting: Optional[Dict]

NLModelConfig = HuggingfaceNLModelConfig


@dataclass
class HuggingfaceNLModel(NLModel):
    model: AutoModel
    tokenizer: AutoTokenizer

    def from_config(cls, cfg: HuggingfaceNLModelConfig) -> "HuggingfaceNLModel":
        loading_module = importlib.import_module(cfg.loading_module.value)
        loading_func = getattr(getattr(loading_module, cfg.loading_cls), cfg.loading_func)

        try:
            cfg.loading_kwargs["torch_dtype"] = getattr(torch,cfg.loading_kwargs["torch_dtype"])
        except KeyError:
            pass


        bnb_config = (
        BitsAndBytesConfig(**cfg.bnb_config)
        if cfg.bnb_config is not None
        else None
    )

        lora_config = (
            LoraConfig(**cfg.lora_config)
            if cfg.lora_config is not None
            else None
        )

        model_loading_kwargs = cfg.loading_kwargs

        if bnb_config is not None:
            if cfg.bnb_config_key in model_loading_kwargs:
                raise ValueError("Key collision")
            model_loading_kwargs[cfg.bnb_config_key] = bnb_config

        model, tokenizer = load_model_and_tokenizer(model_config)


        model = loading_func(cfg.model_path, **cfg.loading_kwargs)

        model.config.pretraining_tp = 1


        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # prepare model for training
        if lora_config is not None:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)


        return cls(model=model, tokenizer=tokenizer)
    

class HuggingFaceNLTrainer(NLTrainer[HuggingfaceNLModel], metaclass=ABCMeta):
    def train(self, pipeline: NLPipeline, data_container: NLDataContainer):
        return super().train(pipeline, data_container)

    
class HuggingfaceModelPatchCatalog(str, Enum):
    LLAMA2= "llama2"
Patch = Dict
class HuggingfaceModelPatcher:
    def get_patch(self, model: HuggingfaceNLModel) -> Tuple[Dict, Dict]