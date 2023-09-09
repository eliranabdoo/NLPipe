from enum import Enum
from typing import Dict
from peft import AutoPeftModelForCausalLM
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class ModelLoadingType(str, Enum):
    PEFT = "peft"
    REGULAR = "regular"

class ModelConfig(BaseModel):
    model_path: str
    loading_type: ModelLoadingType
    loading_kwargs: Dict

DUMMY_MODEL_CONFIG = ModelConfig(
    model_path="NousResearch/Llama-2-7b-hf",
    loading_type=ModelLoadingType.REGULAR,
    loading_kwargs=dict(device_map="auto",use_cache=False)
)

def load_model_and_tokenizer(cfg: ModelConfig):
    loading_func = None
    if cfg.loading_type == ModelLoadingType.PEFT:
        loading_func = AutoPeftModelForCausalLM.from_pretrained
    elif cfg.loading_type == ModelLoadingType.REGULAR:
        loading_func = AutoModelForCausalLM.from_pretrained

    try:
        cfg.loading_kwargs["torch_dtype"] = getattr(torch,cfg.loading_kwargs["torch_dtype"])
    except KeyError:
        pass

    model = loading_func(cfg.model_path, **cfg.loading_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    return model, tokenizer