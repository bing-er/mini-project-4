# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Progress bar (install: pip install tqdm)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set to training mode
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        # Move data to device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation/test set.

    Returns:
        avg_loss, accuracy
    """
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Complete training loop.

    Returns:
        Dictionary with training history
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model = model.to(device)

    # Track history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    print(f"Training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    return history


# Test training
if __name__ == "__main__":
    from model import FashionClassifier
    from utils import get_data_loaders

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Create model
    model = FashionClassifier(hidden_size=128)

    # Train for just 2 epochs as a test
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=2,
        learning_rate=0.001,
        device=device
    )

    print("\n✅ Training works!")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")