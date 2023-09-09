import json
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
import torch
def xls_to_csv(xls_path, csv_path):
    excel = pd.read_excel(xls_path, index_col=0)
    excel.to_csv(csv_path, index=0)

class HydraConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        if isinstance(obj, torch.dtype):
            return str(obj).split(".")[-1]
        return super().default(obj)