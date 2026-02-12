# src/model.py
import torch
import torch.nn as nn

class FashionClassifier(nn.Module):
    """
    Simple feedforward neural network for Fashion-MNIST classification.
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(FashionClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DeepFashionClassifier(nn.Module):
    """
    Deeper neural network with 3 hidden layers, dropout, and batch normalization.

    Args:
        input_size: Size of flattened input (784 for 28x28 images)
        hidden_sizes: List of hidden layer sizes [256, 128, 64]
        num_classes: Number of output classes (10)
        dropout_prob: Dropout probability for regularization
    """

    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10, dropout_prob=0.3):
        super(DeepFashionClassifier, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        # Layer 3
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)

        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output layer (logits)
        x = self.fc4(x)

        return x


# Test both models
if __name__ == "__main__":
    # Test baseline model
    model1 = FashionClassifier()
    dummy_input = torch.randn(32, 1, 28, 28)
    output1 = model1(dummy_input)
    print(f"Baseline model output shape: {output1.shape}")
    print(f"Baseline model parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Test deep model
    model2 = DeepFashionClassifier()
    output2 = model2(dummy_input)
    print(f"\nDeep model output shape: {output2.shape}")
    print(f"Deep model parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("\n✅ Both models work!")