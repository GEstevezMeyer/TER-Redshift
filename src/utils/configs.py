import tomllib
import os
from src.training.architecture import ArchitectureMacSum
import torch

MODEL_REGISTERY = {"ArchitectureMacSum",ArchitectureMacSum}

def import_config(toml_path:str = "config.toml") -> dict: 
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data


def return_device() -> str: 
    return os.getenv("DEVICE")


def load_pth(model_configs:dict,pth_path:str):
    model = MODEL_REGISTERY[model_configs.get("name",ArchitectureMacSum)](**model_configs["model"])
    state_dict = torch.load(pth_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model