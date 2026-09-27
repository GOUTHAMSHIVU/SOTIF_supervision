# tests/conftest.py
import pytest
import torch
import numpy as np
import h5py

@pytest.fixture
def dummy_batch_images():
    """Generates synthetic RGB batch: (B=4, C=3, H=256, W=512)"""
    return torch.randn(4, 3, 256, 512, dtype=torch.float32)

@pytest.fixture
def dummy_feature_vectors():
    """Generates synthetic ResNet-50 penultimate vectors: (B=16, 2048)"""
    return torch.randn(16, 2048, dtype=torch.float32)

@pytest.fixture
def mock_h5_file(tmp_path):
    """Creates a temporary HDF5 file with 32 samples for I/O verification"""
    file_path = tmp_path / "test_features.h5"
    num_samples = 32
    with h5py.File(file_path, "w") as f:
        f.create_dataset("features", data=np.random.randn(num_samples, 2048).astype(np.float32))
        f.create_dataset("filenames", data=np.array([f"img_{i}.png" for i in range(num_samples)], dtype="S"))
    return str(file_path)