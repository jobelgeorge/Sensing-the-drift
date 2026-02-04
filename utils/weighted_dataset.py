import torch
from torch.utils.data import Dataset


class WeightedDataset(Dataset):
    """
    Wraps an existing dataset and adds a sample-wise weight
    based on the sideslip distribution.

    Original dataset returns:
        X, y

    This wrapper returns:
        X, y, w
    """

    def __init__(self, base_dataset, weighter, return_band=False):
        """
        Parameters
        ----------
        base_dataset : torch.utils.data.Dataset
            Your existing dataset (e.g. sequential_dataset)

        weighter : SideSlipWeighter
            Object that maps y -> band -> weight

        return_band : bool
            If True, also return 'few/medium/many' (useful for evaluation)
        """
        self.base_dataset = base_dataset
        self.weighter = weighter
        self.return_band = return_band

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, y = self.base_dataset[idx]

        # Extract scalar sideslip
        if isinstance(y, torch.Tensor):
            y_val = float(y.item())
        else:
            y_val = float(y)

        # Compute weight
        w = torch.tensor(
            self.weighter.weight_of(y_val),
            dtype=torch.float32
        )

        # 🔑 IMPORTANT: return LIST of weights (uvote-style)
        weights = [w]

        return X, y, weights

