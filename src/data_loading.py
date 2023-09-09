from enum import Enum
from datasets import load_dataset, DatasetDict, Dataset
from pydantic import BaseModel, Field
from typing import Callable, Dict, List, Tuple
import re

from preprocess import ProcessingPipeline, ProcessingMode, DUMMY_PREPROCESS_PIPELINE



class DataAdapterType(str, Enum):
    HF_TO_SAMPLE = "hf_to_sample"

class DataAdapterConfig(BaseModel):
    data_adapter_type: DataAdapterType
    data_adapter_kwargs: Dict
    data_adapter_result_key: str

DUMMY_DATA_ADAPTER_CONFIG = DataAdapterConfig(data_adapter_type=DataAdapterType.HF_TO_SAMPLE,
                                              data_adapter_kwargs={"label_key": "description",
                                                                   "sample_keys": ["html","html_element"]},
                                              data_adapter_result_key="prompt") 
def hf_to_sample_converter(examples, sample_keys, label_key):
    keys = list(examples.keys())
    size = len(examples[keys[0]])
    for i in range(size):
        yield [[examples[k][i] for k in sample_keys], examples[label_key][i]]


def get_data_adapters(cfg: DataAdapterConfig) -> Tuple[Callable, Callable]:
    if cfg.data_adapter_type == DataAdapterType.HF_TO_SAMPLE:
        return lambda data : hf_to_sample_converter(data, **cfg.data_adapter_kwargs), lambda data: {cfg.data_adapter_result_key: [datum[0] for datum in data]}
    

class DataLoadingConfig(BaseModel):
    paths: List[str]
    test_size: float
    batch_size: int = 1000
    data_adapter_config: DataAdapterConfig
    mode: ProcessingMode
    seed: int


class DataPreparationConfig(BaseModel):
    dataset_config: DataLoadingConfig
    splits_to_preprocess: List[str] = Field(default_factory=lambda: ["train"])


def prepare_data(dataset: Dataset, cfg: DataLoadingConfig, processing_pipeline: ProcessingPipeline):
    processing_pipeline.mode = cfg.mode

    def apply_batch(examples):
        data_adapter, data_readapter = get_data_adapters(cfg.data_adapter_config)
        data = data_adapter(examples)
        data = processing_pipeline.process(data)
        return data_readapter(data)

    dataset = dataset.map(function=apply_batch, batched=True, batch_size=cfg.batch_size, remove_columns=dataset.column_names)
    return dataset

def load_data(cfg: DataLoadingConfig) -> DatasetDict:
    dataset: Dataset = load_dataset("csv", data_files=cfg.paths, split="train")
    split_dataset_dict = dataset.train_test_split(
        seed=cfg.seed, test_size=cfg.test_size
    )
    return split_dataset_dict


def load_and_preprocess_data(cfg: DataPreparationConfig, processing_pipeline: ProcessingPipeline) -> DatasetDict:
    split_dataset_dict = load_data(cfg.dataset_config)
    splits_to_preprocess = cfg.splits_to_preprocess
    for split in splits_to_preprocess:
        split_dataset_dict[split] = prepare_data(
            dataset=split_dataset_dict[split], cfg=cfg.dataset_config,
            processing_pipeline=processing_pipeline
        )
    return split_dataset_dict


DUMMY_DATA_PREP_CONFIG = DataPreparationConfig(
        dataset_config=DataLoadingConfig(
            paths=["./data/data-dummy-nlp.csv"], test_size=0.2, seed=1337,
            data_adapter_config=DUMMY_DATA_ADAPTER_CONFIG, mode=ProcessingMode.TRAINING
        ),
    )

if __name__ == "__main__":
    dataset = load_and_preprocess_data(cfg=DUMMY_DATA_PREP_CONFIG,
                                       processing_pipeline=DUMMY_PREPROCESS_PIPELINE)
    print(dataset)
