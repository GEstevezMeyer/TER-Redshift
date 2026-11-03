import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TRAINING_DATA_DEFAULT_PATH = "data/epoch_40_train_results.npy"
VALIDATION_DATA_DEFAULT_PATH = "data/epoch_40_val_results.npy"

TARGET_KEY = "z_true"
LATENT_KEY = "final_latent"

class EspaceLatent(Dataset):
    def __init__(self, path):
        super().__init__()
        data: dict = np.load(path, allow_pickle=True).item()
        x = np.asarray(data[LATENT_KEY])
        y = np.asarray(data[TARGET_KEY])
        valid = np.isfinite(y) & (y > 0)
        self.x = x[valid]
        self.y = y[valid].astype(np.float32)

        x_min = torch.tensor(np.min(self.x), dtype=torch.float32)
        x_max = torch.tensor(np.max(self.x), dtype=torch.float32)
        x_range = x_max - x_min
        if float(x_range) < 1e-6 or not torch.isfinite(x_range):
            x_range = torch.tensor(1.0, dtype=torch.float32)
        y_min = torch.tensor(np.min(self.y), dtype=torch.float32)
        y_max = torch.tensor(np.max(self.y), dtype=torch.float32)
        y_range = y_max - y_min
        if float(y_range) < 1e-6 or not torch.isfinite(y_range):
            y_range = torch.tensor(1.0, dtype=torch.float32)

        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_range
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_range

        self.final_latent = self.x
        self.targets = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].copy(), dtype=torch.float32)
        y = torch.tensor(self.y[idx].copy(), dtype=torch.float32)
        x = (x - self.x_min) / self.x_range
        y = (y - self.y_min) / self.y_range
        return x, y

    @property
    def min(self):
        return self.x_min.tolist(), self.y_min.tolist()

    @property
    def max(self):
        return self.x_max.tolist(), self.y_max.tolist()

    @property
    def range(self):
        return self.x_range.tolist(), self.y_range.tolist()

def main_pipeline(batch_size: int) -> tuple[DataLoader, DataLoader]:
    training_dataset = EspaceLatent(TRAINING_DATA_DEFAULT_PATH)
    validation_dataset = EspaceLatent(VALIDATION_DATA_DEFAULT_PATH)

    return DataLoader(training_dataset, batch_size, shuffle=True), DataLoader(validation_dataset, batch_size, shuffle=False)


if __name__ == "__main__":
    print(main_pipeline(32))
