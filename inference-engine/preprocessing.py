# Image pipelines for preprocessing and postprocessing
from torchvision import transforms
from PIL import Image
import torch

#ImageNet statistics (required for ResNet50 pre-trained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]