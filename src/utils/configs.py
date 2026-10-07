import tomllib
import os

def import_config(toml_path:str = "config.toml") -> dict: 
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data


def return_device() -> str: 
    return os.getenv("DEVICE")