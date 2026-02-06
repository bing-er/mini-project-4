# src/utils.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_data_loaders(batch_size=64, val_split=0.1):
    """
    Load Fashion-MNIST and create train/val/test dataloaders.

    Args:
        batch_size: Batch size for training
        val_split: Fraction of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Transform: converts PIL image to tensor and scales to [0, 1]
    transform = transforms.ToTensor()

    # Download datasets
    train_val_data = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Split training into train and validation
    val_size = int(len(train_val_data) * val_split)
    train_size = len(train_val_data) - val_size

    train_data, val_data = random_split(
        train_val_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_loader, val_loader, test_loader


def get_class_names():
    """Return Fashion-MNIST class names."""
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def visualize_samples(data_loader, num_samples=10):
    """Visualize random samples from the dataset."""
    class_names = get_class_names()

    # Get one batch
    images, labels = next(iter(data_loader))

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    plt.show()


# Test the data loading
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    visualize_samples(train_loader)
    print("✅ Data loading works!")