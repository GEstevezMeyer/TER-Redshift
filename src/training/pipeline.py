import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

TRAINING_DATA_DEFAULT_PATH = "data/epoch_40_train_results.npy"
VALIDATION_DATA_DEFAULT_PATH = "data/epoch_40_val_results.npy"

TARGET_KEY = "z_true"
LATENT_KEY = "final_latent"

TEST_SIZE_DEFAULT = 0.2


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

    @property
    def mean(self):
        return self.x_mean.tolist(), self.y_mean.tolist()

    @property
    def std(self):
        return self.x_std.tolist(),self.y_std.tolist()

def main_pipeline(batch_size: int, test_size: float = TEST_SIZE_DEFAULT) -> tuple[DataLoader, dict[str, DataLoader]]:

    training_dataset = EspaceLatent(TRAINING_DATA_DEFAULT_PATH)
    validation_dataset = EspaceLatent(VALIDATION_DATA_DEFAULT_PATH)

    n_total = len(training_dataset)
    n_test = int(round(n_total * test_size))
    n_train = n_total - n_test
    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"test_size={test_size} gives an empty split for {n_total} samples"
        )

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        training_dataset, [n_train, n_test], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    eval_dataloaders = {
        "train": DataLoader(train_dataset, batch_size, shuffle=False),
        "validation": DataLoader(validation_dataset, batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size, shuffle=False),
    }

    return train_loader, eval_dataloaders


if __name__ == "__main__":
    train_loader, eval_dataloaders = main_pipeline(32)
    print(f"train batches: {len(train_loader)}")
    for name, loader in eval_dataloaders.items():
        print(f"{name}: {len(loader.dataset)} samples")
