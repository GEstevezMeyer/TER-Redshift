import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader
from typing import Any

TRAINING_DATA_DEFAULT_PATH = "data/epoch_40_train_results.npy"
VALIDATION_DATA_DEFAULT_PATH = "data/epoch_40_val_results.npy"

class EspaceLatent(Dataset):
    def __init__(self,path):
        super().__init__()
        data:dict[Any] = np.load(path,allow_pickle=True).item()
        self.labels = data["label"]
        self.final_latent = data["final_latent"]

    def __len__(self):
        return len(self.final_latent)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].copy(), dtype=torch.float32)
        y = torch.tensor(self.y[idx].copy(), dtype=torch.float32)
        return x,y


def main_pipeline(batch_size:int) -> tuple[DataLoader]: 
    training_dataset = EspaceLatent(TRAINING_DATA_DEFAULT_PATH)
    validation_dataset = EspaceLatent(VALIDATION_DATA_DEFAULT_PATH)

    return DataLoader(training_dataset,batch_size,shuffle=True), DataLoader(validation_dataset,batch_size,shuffle=True)

if __name__ == "__main__": 
    print(main_pipeline(32))