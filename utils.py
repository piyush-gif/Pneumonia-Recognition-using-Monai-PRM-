from torchvision.transforms import Compose, Resize, Normalize, ToTensor, Grayscale
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Transformations for training dataset (with data augmentation)
train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize(mean=[0.485], std=[0.229]),
])

# Transformations for test and validation datasets (without data augmentation)
test_val_transform = Compose([
    Resize((224, 224)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize(mean=[0.485], std=[0.229]),
])

# Load the datasets
train_dataset = ImageFolder('C:/Users/LENOVO/Desktop/Production Project/chest_xray/train', transform=train_transform)
val_dataset = ImageFolder('C:/Users/LENOVO/Desktop/Production Project/chest_xray/val', transform=test_val_transform)
test_dataset = ImageFolder('C:/Users/LENOVO/Desktop/Production Project/chest_xray/test', transform=test_val_transform)

# Print the number of samples in each dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


batch_size = 300 # dataset divided into 300 samples for memory usage

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)