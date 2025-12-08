import os
from random import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision import datasets
import random

train_transform = transforms.Compose([
    #Rexize images to 224x224 
    transforms.Resize((224, 224)),
    # flip the images randomly for more variation
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # Convert images to tensor and scale pixel intensities
    transforms.ToTensor(),
    # Normalize images with mean and std deviation
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
    
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset      # Subset
        self.transform = transform

    def __getitem__(self, idx):
        # Get underlying dataset index
        original_idx = self.subset.indices[idx]

        # Fetch image *correctly* from the underlying ConcatDataset
        img, label = self.subset.dataset[original_idx]

        # Apply transform to PIL image
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)
    
def get_train_classes():
    rps_train = datasets.ImageFolder("data/rps")
    classes = rps_train.classes
    return classes

def load_data():
    rps_train = datasets.ImageFolder("data/rps")
    rps_val   = datasets.ImageFolder("data/rps-validation")
    rps_test  = datasets.ImageFolder("data/rps-test-set")

    # Combine into full dataset
    full_dataset = ConcatDataset([rps_train, rps_val, rps_test])

    # Split into train/val/test
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms CORRECTLY using wrapper
    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset   = TransformDataset(val_subset, val_transform)
    test_dataset  = TransformDataset(test_subset, val_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=2, worker_init_fn=worker_init_fn)

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    return train_loader, val_loader, test_loader