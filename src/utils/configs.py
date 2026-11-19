import tomllib
import os
import torch


def import_config(toml_path:str = "config.toml") -> dict: 
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data


def return_device() -> str: 
    return os.getenv("DEVICE")


def load_pth(configs:dict,pth_path:str):
    from src.training.training import get_model

    model = get_model(configs["model"])
    state_dict = torch.load(pth_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model
