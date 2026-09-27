# src/data/feature_dataset.py
import h5py
import torch
from torch.utils.data import Dataset

class ExtractedFeatureDataset(Dataset):
    """
    Loads pre-extracted 2048-dimensional vectors directly from HDF5.
    Can be loaded entirely into RAM for maximum training speed.
    """
    def __init__(self, h5_path: str, in_memory: bool = True):
        self.h5_path = h5_path
        self.in_memory = in_memory

        if self.in_memory:
            # Load the entire ~270MB array into system RAM at startup
            with h5py.File(h5_path, "r") as f:
                self.features = torch.from_numpy(f["features"][:]).float()
        else:
            self.features = None
            with h5py.File(h5_path, "r") as f:
                self.length = f["features"].shape[0]

    def __len__(self):
        return len(self.features) if self.in_memory else self.length

    def __getitem__(self, idx):
        if self.in_memory:
            return self.features[idx]
        
        # Lazy disk reading fallback
        with h5py.File(self.h5_path, "r") as f:
            feat = torch.from_numpy(f["features"][idx]).float()
        return feat