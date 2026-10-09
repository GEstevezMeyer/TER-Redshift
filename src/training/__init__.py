import torch 
import os 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

os.environ["DEVICE"] = device