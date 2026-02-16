import torch

# Check if Apple MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU via MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (training will be slow)")

print(f'Device: {device}')
