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

        x_mean = torch.tensor(np.mean(self.x), dtype=torch.float32)
        x_std = torch.tensor(np.std(self.x), dtype=torch.float32)
        if float(x_std) < 1e-6:
            x_std = torch.tensor(1.0, dtype=torch.float32)
        y_mean = torch.tensor(np.mean(self.y), dtype=torch.float32)
        y_std = torch.tensor(np.std(self.y), dtype=torch.float32)
        if float(y_std) == 0 or not torch.isfinite(y_std):
            y_std = torch.tensor(1.0, dtype=torch.float32)

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        self.final_latent = self.x
        self.targets = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].copy(), dtype=torch.float32)
        y = torch.tensor(self.y[idx].copy(), dtype=torch.float32)
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return x, y


def main_pipeline(batch_size: int) -> tuple[DataLoader, DataLoader]:
    training_dataset = EspaceLatent(TRAINING_DATA_DEFAULT_PATH)
    validation_dataset = EspaceLatent(VALIDATION_DATA_DEFAULT_PATH)

    return DataLoader(training_dataset, batch_size, shuffle=True), DataLoader(validation_dataset, batch_size, shuffle=False)


if __name__ == "__main__":
    print(main_pipeline(32))
