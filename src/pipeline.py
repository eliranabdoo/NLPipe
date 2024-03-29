from dataclasses import dataclass
from enum import Enum
import importlib
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import BaseModel, Field
import torch
from tqdm import tqdm

from src.evaluation import MetricsConfig, get_compute_metrics_func
from .preprocess import ProcessingMode, ProcessingPipeline, PromptingComponent

from abc import abstractclassmethod, ABCMeta, abstractmethod, abstractstaticmethod


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


NLModel = TypeVar("NLModel", bound=(NLComponent, Callable))


class NLModel(NLComponent, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *, x, **kwargs):
        pass


NLModelType = TypeVar("NLModelType", bound=NLModel)


@dataclass
class NLPipeline(Generic[NLModelType], metaclass=ABCMeta):
    preprocessing_pipeline: ProcessingPipeline
    model: NLModelType
    postprocessing_pipeline: ProcessingPipeline

    def predict(self, *, x, processing_mode: ProcessingMode, **kwargs):
        x = self.preprocessing_pipeline.process(x, processing_mode=processing_mode)
        x = self.model(x=x, **kwargs)
        x = self.postprocessing_pipeline.process(x, processing_mode=processing_mode)
        return x


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


class NLPipelineOperator(Generic[Data], NLComponent, metaclass=ABCMeta):
    @abstractmethod
    def execute(self, pipeline: NLPipeline, data_mapping: Dict[str, Data]):
        pass


class NLExperimentOutcome(metaclass=ABCMeta):
    pass


class NLExperiment(NLComponent, metaclass=ABCMeta):
    """An experiment will take a pipeline, a data cooker and an operator,
    and apply the pipeline according to the operator, on the cooked data
    
    An ExperimentOutcome object should contain different information according to the task"""
    pipeline: NLPipeline
    data_cooker: NLDataCooker
    task: NLPipelineOperator

    def run(
        self,
        *,
        data_container: NLDataContainer,
    ) -> NLExperimentOutcome:
        cooked_data_container = self.data_cooker.cook(data_container)
        result = self.task.execute(self.pipeline, cooked_data_container)
        result = self.get_converter(type(self.task))(result)
        self.handle_result(result)

    def get_converter(self, task_type: Type[NLPipelineOperator]) -> Callable:
        if task_type == "Huggingface":

            def convert(result) -> NLExperimentOutcome:
                df = dataset.to_pandas()
                df["preds"] = preds

                preds_dir = os.path.join(prediction_cfg.pred_dir, prediction_cfg.uid)
                os.makedirs(preds_dir)
                df.to_csv(os.path.join(preds_dir, "preds.csv"))
                json.dump(
                    prediction_cfg.model_dump(),
                    open(os.path.join(preds_dir, "config.json"), "w"),
                    cls=HydraConfigEncoder,
                )

                if metrics_cfg is not None:
                    compute_metrics = get_compute_metrics_func(metrics_cfg)
                    metrics_result = compute_metrics(
                        [
                            df["preds"].to_list(),
                            [[ref] for ref in df["description"].to_list()],
                        ]
                    )
                    json.dump(
                        metrics_result,
                        open(os.path.join(preds_dir, "metrics.json"), "w"),
                    )

                return preds

    def handle_result(self, result: NLExperimentOutcome):
        pass


from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


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

    def __call__(
        self,
        x,
        inputs_keys: Set[str],
        tokenization_kwargs: Dict,
        generation_kwargs: Optional[Dict],
    ):
        inputs = self.tokenizer(x, **tokenization_kwargs)

        if generation_kwargs is not None:
            predict_func = self.model.generate
        else:
            predict_func = self.model.forward
            generation_kwargs = {}
        outputs = predict_func(
            **{inputs.get(k) for k in inputs_keys}, **generation_kwargs
        )

        preds = self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )
        return preds

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


class HuggingfaceDataAdapterType(str, Enum):
    TEXT = "text"


class HuggingFaceNLTrainerConfig(NLConfig):
    kind: Literal["hf"] = "hf"
    max_sequence_length: int
    training_args: TrainingArguments
    metrics_config: MetricsConfig
    hf_map_kwargs: Dict
    data_adapter_type: HuggingfaceDataAdapterType


class HuggingfaceDataAdapter(Generic[Data], metaclass=ABCMeta):
    registry = {}

    def __init_subclass__(cls):
        HuggingfaceDataAdapter.registry[cls.get_type()] = cls

    @abstractstaticmethod
    def batch_to_data(batch: Dict) -> Data:
        pass

    @abstractstaticmethod
    def data_to_batch(data: Data) -> Dict:
        pass

    @abstractstaticmethod
    def get_type() -> HuggingfaceDataAdapterType:
        pass

class TextHuggingfaceDataAdapter(HuggingfaceDataAdapter):
    @staticmethod
    def get_type():
        return HuggingfaceDataAdapterType.TEXT
    
    def batch_to_data(batch: Dict) -> Data:
        pass

    def data_to_batch(data: Data) -> Dict:
        pass

    


class HuggingFaceNLTrainer(NLPipelineOperator[HFDataset], metaclass=ABCMeta):
    train_split = "train"
    eval_split = "eval"
    cfg: HuggingFaceNLTrainerConfig

    def from_config(cls, config):
        return cls(config)

    def __init__(self, cfg: HuggingFaceNLTrainerConfig):
        self.cfg = cfg

    @abstractmethod
    def get_trainer(
        self,
        preprocess_pipeline: ProcessingPipeline,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        train_dataset: HFDataset,
        eval_dataset: HFDataset,
        max_seq_length: int,
        training_args: TrainingArguments,
        compute_metrics: Optional[Callable],
    ) -> Dict:
        pass

    def execute(
        self,
        pipeline: NLPipeline[HuggingfaceNLModel],
        data_mapping: Dict[str, HFDataset],
    ):
        if set(data_mapping.keys()).issubset({self.train_split, self.eval_split}):
            raise ValueError("invalid data mapping keys")

        model, tokenizer = pipeline.model.model, pipeline.model.tokenizer

        train_dataset = data_mapping[self.train_split]
        eval_dataset = data_mapping.get(self.eval_split, None)

        max_seq_length = self.cfg.max_sequence_length

        training_args = self.cfg.training_args

        compute_metrics = None
        if self.cfg.metrics_config is not None:
            compute_metrics = get_compute_metrics_func(
                self.cfg.metrics_config, pipeline.postprocessing_pipeline
            )

        data_adapter: HuggingfaceDataAdapter = HuggingfaceDataAdapter.registry[
            self.cfg.data_adapter_type
        ]

        def process_batch(example):
            data = data_adapter.batch_to_data(example)
            data = pipeline.preprocessing_pipeline.process(
                data, processing_mode=ProcessingMode.TRAINING
            )
            example = data_adapter.data_to_batch(data)
            return example

        for split in data_mapping:
            data_mapping[split] = data_mapping[split].map(
                function=process_batch, **self.cfg.hf_map_kwargs
            )

        trainer = self.get_trainer(
            preprocess_pipeline=pipeline.preprocessing_pipeline,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=max_seq_length,
            training_args=training_args,
            compute_metrics=compute_metrics,
        )
        trainer.train()


class HuggingFaceSFTNLTrainerConfig(HuggingFaceNLTrainerConfig):
    kind: Literal["hf_sft"] = "hf_sft"
    train_on_completion_only: bool
    text_field: str


class HuggingFaceSFTNLTrainer(HuggingFaceNLTrainer):
    cfg: HuggingFaceSFTNLTrainerConfig

    def get_trainer(
        self,
        preprocess_pipeline: ProcessingPipeline,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset],
        max_seq_length: int,
        training_args: TrainingArguments,
        compute_metrics: Optional[Callable],
    ) -> Dict:
        collator = None
        if self.cfg.train_on_completion_only:
            response_template = next(
                step.delimiter + step.label_field
                for step in preprocess_pipeline.steps[::-1]
                if isinstance(step, PromptingComponent)
            )
            response_template_ids = tokenizer.encode(
                response_template, add_special_tokens=False
            )
            collator = DataCollatorForCompletionOnlyLM(
                response_template_ids, tokenizer=tokenizer
            )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field=self.cfg.text_field,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        return trainer


class NLPredictorConfig(NLConfig, metaclass=ABCMeta):
    use_tqdm: bool


class HuggingfaceNLPredictorConfig(NLPredictorConfig):
    kind: Literal["hf"] = "hf"
    generation_kwargs: Dict = Field(
        default_factory=lambda: dict(
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
    )
    tokenization_kwargs: Dict = Field(
        default_factory=lambda: dict(
            return_tensors="pt", truncation=True, max_length=512
        )
    )
    data_device: str = "cuda"


class NLPredictor(NLPipelineOperator):
    cfg: NLPredictorConfig

    def execute(self, pipeline: NLPipeline, data_mapping: Dict[str, Any]):
        results = {}
        for data_name, data in data_mapping.values():
            samples = (datum[0] for datum in data)
            # TODO think about split-process-merge paradigm
            if self.cfg.use_tqdm:
                samples = tqdm(samples, total=len(data))
            preds = pipeline.predict(samples, processing_mode=ProcessingMode.INFERENCE)
            results[data_name] = preds

        return results


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
