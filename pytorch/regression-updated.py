import torch

# Seed for reproducibility
torch.manual_seed(1)

# Initialize parameters with requires_grad=True to compute gradients
weight = torch.randn(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

def loss_fn(input, target):
    """Calculates the mean squared error between input and target."""
    loss = (input - target).pow(2).mean()
    print(f"Loss calculation: Input={input}, Target={target}, Loss={loss}")
    return loss

def model(xb):
    """Defines the linear regression model using matrix multiplication."""
    return xb @ weight + bias

# Sample dataset: features (x) and labels (y)
# Note: In a real scenario, replace these tensors with your actual dataset.
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# Training hyperparameters
learning_rate = 0.001
num_epochs = 2

for epoch in range(num_epochs):
    # Assuming x and y are your data and labels
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()  # Compute gradients

    # Update parameters
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate

        # Zero gradients after updating
        weight.grad.zero_()
        bias.grad.zero_()

    print(f'Epoch {epoch + 1}: Loss={loss.item():.4f}')

# Output logs will show loss calculations and parameter updates
