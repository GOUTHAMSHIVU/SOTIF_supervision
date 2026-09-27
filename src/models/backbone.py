# src/models/backbone.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50FeatureExtractor(nn.Module):
    """
    Extracts the 2048-dimensional penultimate latent embedding 
    from a frozen ResNet-50 model.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)
        
        # Keep everything up to the Adaptive Average Pooling layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Freeze all backbone parameters completely
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # Output shape: (B, 2048)