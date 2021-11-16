import torch
from torch.utils.data import Dataset


# https://discuss.pytorch.org/t/dictionary-in-dataloader/40448/2
class DictionaryDataset(Dataset):
    """Dataset that returns X, y and possible extra kwargs."""
    def __init__(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.extra = kwargs

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        extra = {
            kw: val[index] for kw, val in self.extra.items()
        }
        return {'X': X, 'y': y, **extra}

    def __len__(self):
        return self.X.shape[0]


def get_dataset(X, y, **extra):
    dataset = DictionaryDataset(X, y, **extra)
    return dataset


def get_dataloader(dataset, batch_size, **kwargs):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)
