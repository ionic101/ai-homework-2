import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_classification


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32) # type: ignore
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1) # type: ignore
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(count_data=100, count_features: int = 1,
        count_classes: int = 1, source='random') -> tuple[torch.Tensor, torch.Tensor]:
    if source == 'random':
        X, y = make_classification(
            n_samples=count_data,
            n_features=count_features,
            n_informative=min(count_features, count_classes),
            n_redundant=0,
            n_classes=count_classes,
            class_sep=2.0,
            random_state=42
        )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y
    else:
        raise ValueError('Unknown source')

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean().item()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)
