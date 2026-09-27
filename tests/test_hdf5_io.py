# tests/test_hdf5_io.py
from src.data.feature_dataset import ExtractedFeatureDataset

def test_feature_dataset_loading(mock_h5_file):
    dataset = ExtractedFeatureDataset(mock_h5_file, in_memory=True)
    assert len(dataset) == 32
    
    sample = dataset[0]
    assert sample.shape == (2048,)
    assert sample.dtype.is_floating_point