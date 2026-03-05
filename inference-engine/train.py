# Training Loop
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from model import DRClassifier
from preprocessing import train_transform, inference_transform

 #── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 32
LR         = 1e-4

# Auto-detect best available device: Apple M4 → MPS, NVIDIA → CUDA, else CPU
# If device is CPU, consider using smaller batch size to avoid long training times
# If device is MPS, ensure you have the latest PyTorch version and macOS updates
# If device is CUDA, ensure you have the correct NVIDIA drivers and CUDA toolkit installed
DEVICE     = torch.device('mps' if torch.backends.mps.is_available() else
             'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {DEVICE}")