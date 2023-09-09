from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import re
import pandas as pd

from pydantic import BaseModel

Sample = TypeVar("Sample")
Label = TypeVar("Label")
Datum = Tuple[Sample, Optional[Label]]
Data = Iterable[Datum]


class ProcessingMode(int, Enum):
    TRAINING = 0
    INFERENCE = 1


class ProcessingComponent(Generic[Sample, Label], ABC):
    @abstractmethod
    def get_max_mode(self) -> ProcessingMode:
        pass

    @abstractmethod
    def process(self, data: Data) -> Data:
        pass


class ProcessingPipelineConfig(BaseModel):
    steps: List[Dict[str, Any]]
    mode: ProcessingMode

    @classmethod
    def from_pipeline(cls, pipeline: "ProcessingPipeline"):
        steps_config = []
        for step in pipeline.steps:
            step_config = {"name": step.__class__.__name__, "params": step.__dict__}
            steps_config.append(step_config)

        return cls(steps=steps_config, mode=pipeline.mode)

    def to_pipeline(self) -> "ProcessingPipeline":
        steps = []
        for step_config in self.steps:
            step_class = globals()[step_config["name"]]
            step_params = step_config["params"]
            step_instance = step_class(**step_params)
            steps.append(step_instance)

        return ProcessingPipeline(steps=steps, mode=self.mode)


class DataFilteringComponent(ProcessingComponent[Sample, Label], ABC):
    def get_max_mode(self) -> ProcessingMode:
        return ProcessingMode.TRAINING

    def process(self, data: Data) -> Data:
        valid_indices = self.get_valid_indices(data)
        return (data[i] for i in valid_indices)

    @abstractmethod
    def get_valid_indices(self, data: Data) -> Iterable[int]:
        pass


class AlteringComponent(ProcessingComponent[Sample, Label], ABC):
    def __init__(self, apply_to_labels: bool, *arg, **kwargs):
        self.apply_to_labels = apply_to_labels

    def get_max_mode(self) -> ProcessingMode:
        return ProcessingMode.INFERENCE

    def process(self, data: Data) -> Data:
        res = []
        apply_idx = 0 if not self.apply_to_labels else 1
        for datum in data:
            datum = list(datum)
            datum[apply_idx] = self._process(datum)
            res.append(datum)
        return res

    @abstractmethod
    def _process(self, datum: Datum) -> Union[Sample, Optional[Label]]:
        pass


class DedupComponent(DataFilteringComponent[str, str]):
    def __init__(self, by_features: bool, by_label: bool, keep_strategy: str):
        self.by_features = by_features
        self.by_label = by_label
        self.keep_strategy = keep_strategy

    def get_valid_indices(self, data: Data) -> Iterable[int]:
        df = pd.DataFrame.from_records(data)
        subset = [i for i, b in enumerate([self.by_features, self.by_label]) if b]
        valid_indices = list(
            df.drop_duplicates(subset=subset, keep=self.keep_strategy).index
        )
        return valid_indices


class PromptingComponent(ProcessingComponent[List[str], str]):
    def __init__(
        self, sample_fields: List[str], label_field: str, delimiter: str = "\n\n"
    ):
        self.delimiter = delimiter
        self.sample_fields = sample_fields
        self.label_field = label_field

    def get_max_mode(self) -> ProcessingMode:
        return ProcessingMode.INFERENCE

    def process(self, data: Data) -> Data:
        fields = self.sample_fields + [self.label_field]
        res = []
        for datum in data:
            comps = list(datum[0])
            if datum[1] is not None:
                comps.append(datum[1])
            else:
                comps.append("")

            prompt = self.delimiter.join(
                [f"{fields[idx]}{comps[idx]}" for idx in range(len(fields))]
            )
            res.append((prompt, None))
        return res


class TextStripProcessor(AlteringComponent[str, str]):
    def _process(self, datum: Datum):
        if self.apply_to_labels and datum[1] is None:
            return None

        payload = datum[int(self.apply_to_labels)]
        if isinstance(payload, str):
            payload = [payload]
        payload = list(map(str.strip, payload))
        if isinstance(payload, str):
            payload = payload[0]
        return payload


class RegexReplaceProcessor(AlteringComponent[str, str]):
    def __init__(self, pattern: str, repl: str, apply_to_labels: bool):
        self.pattern = pattern
        self.repl = repl
        self.apply_to_labels = apply_to_labels

    def _process(self, datum: Datum):
        if self.apply_to_labels and datum[1] is None:
            return None
        payload = datum[int(self.apply_to_labels)]
        if isinstance(payload, str):
            payload = [payload]
        payload = list(
            map(
                lambda text: re.sub(pattern=self.pattern, repl=self.repl, string=text),
                payload,
            )
        )
        if isinstance(payload, str):
            payload = payload[0]
        return payload


class ProcessingPipeline:
    def __init__(self, steps: List[ProcessingComponent], mode: ProcessingMode):
        self.mode = mode
        self.steps = steps

    def process(
        self, samples: Data, processing_mode: Optional[ProcessingMode] = None
    ) -> Data:
        processing_mode = self.mode if processing_mode is None else processing_mode
        for step in self.steps:
            if processing_mode > step.get_max_mode():
                continue
            samples = step.process(samples)
        return samples


DUMMY_PREPROCESS_PIPELINE = ProcessingPipeline(
    steps=[
        PromptingComponent(
            sample_fields=["### HTML Dom - ", "### HTML Element - "],
            label_field="### Description - ",
        ),
        TextStripProcessor(apply_to_labels=True),
        RegexReplaceProcessor(pattern=r"\d+", repl="NUM", apply_to_labels=True),
        DedupComponent(by_features=True, by_label=False, keep_strategy="first"),
    ],
    mode=ProcessingMode.TRAINING,
)

if __name__ == "__main__":
    config = ProcessingPipelineConfig.from_pipeline(DUMMY_PREPROCESS_PIPELINE)
    print("Pipeline Configuration:")
    print(config.model_dump_json())

    loaded_pipeline = config.to_pipeline()
    print("\nLoaded Pipeline:")
    print(
        loaded_pipeline.process(
            [([" Text with spaces ", "more text  "], "Label with 123")]
        )
    )
