# This file defines the architecture of the model used for inference-engine.
import torch
import torch.nn as nn
from torchvision import models

class DRClassifier(nn.Module):
    """ResNet50 backbone with a custom FC head for DR grading."""
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(DRClassifier, self).__init__()

        #Load ResNet50 with pre-trained ImageNet weights
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        #Freeze backbone layers (preserve learned features)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Unfreeze the final residual block (layer4) for fine-tuning
        # This allows the model to adapt its deepest features to retinal images
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
