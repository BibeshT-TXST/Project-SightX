# Image pipelines for preprocessing and postprocessing
from torchvision import transforms
from PIL import Image
import torch

#ImageNet statistics (required for ResNet50 pre-trained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----- Training Transforms with data augmentation -----
# Augmentation artifically expands our dataset by creating variations of the original images.
# This helps the model generalize better and reduces overfitting.

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a consistent size
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2),  # Randomly adjust brightness and contrast
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Applying ImageNet Stats
])

# ----- Training Transforms with no augmentation -----
# During predection, apply only deterministic steps
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a consistent size
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ----- Inference Preprocessing Function -----
# This function takes an image path, loads the image, applies the necessary transformations, 
# and returns a tensor ready for inference.
def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    img = Image.open(image_path).convert('RGB') # Load the image and force it into standard RGB
    tensor = inference_transforms(img)  # Apply the deterministic transforms 
    return tensor.unsqueeze(0) # Add the batch dimension and return