from dataclasses import dataclass
from enum import Enum
import importlib
from typing import Dict, Literal, Optional

from pydantic import BaseModel
from .preprocess import ProcessingPipeline

from abc import abstractclassmethod, ABCMeta



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
    
class NLModel(NLComponent, metaclass=ABCMeta):
    pass

class NLPipeline(NLComponent, metaclass=ABCMeta):
    model: NLModel
    preprocessing_pipeline: ProcessingPipeline
    postprocessing_pipeline: ProcessingPipeline


class NLDataContainer(NLComponent, metaclass=ABCMeta):
    pass


class NLTrainer(NLComponent, metaclass=ABCMeta):
    pass


class NLDataCooker(NLComponent, metaclass=ABCMeta):
    processing_pipeline: ProcessingPipeline

    def cook(dataset: NLDataContainer) -> NLDataContainer:
        pass

class NLExperimentOutcome(metaclass=ABCMeta):
    pass

class NLExperiment(NLComponent, metaclass=ABCMeta):
    def conduct(
        pipeline: NLPipeline,
        train_data: NLDataContainer,
        validation_data: Optional[NLDataContainer],
        eval_data: Optional[NLDataContainer],
        data_cooker: NLDataCooker,
        trainer: NLTrainer,
    ) -> NLExperimentOutcome:
        train_data = data_cooker.cook(train_data)

class NLInference:
    pipeline: NLPipeline

    def from_experiment(experiment_outcome: NLExperimentOutcome) -> "NLInference":
        pass




from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM



class HFLoadingModules(str, Enum):
    TRANSFORMERS = "transformers"
    PEFT = "peft"

class HuggingfaceNLModelConfig(metaclass=NLConfigMeta):
    kind: Literal['hf'] = 'hf'
    model_path: str
    loading_module: HFLoadingModules
    loading_cls: str
    loading_func: str = "from_pretrained"
    loading_kwargs: Dict

NLModelConfig = HuggingfaceNLModelConfig

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

        model = loading_func(cfg.model_path, **cfg.loading_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

        return cls(model=model, tokenizer=tokenizer)
    


