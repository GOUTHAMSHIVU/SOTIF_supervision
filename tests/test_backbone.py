# tests/test_backbone.py
import torch
from src.models.backbone import ResNet50FeatureExtractor

def test_backbone_output_shape(dummy_batch_images):
    model = ResNet50FeatureExtractor(pretrained=False)
    with torch.no_grad():
        out = model(dummy_batch_images)
    
    assert out.shape == (4, 2048), f"Expected (4, 2048), got {out.shape}"
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any(), "Backbone features contain NaNs"

def test_backbone_parameters_are_frozen():
    model = ResNet50FeatureExtractor(pretrained=False)
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"