import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

learning_rate = 0.001
num_epochs = 2
log_epochs = 1

# Mock dataset 
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Features
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Labels
train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)

# Model definition
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss function
loss_fn = nn.MSELoss(reduction='mean')

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch).squeeze()  # Adjusted for batch processing

        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)

        # 3. Compute gradients
        loss.backward()

        # 4. Update parameters using gradients
        optimizer.step()

        # 5. Reset the gradients to zero
        optimizer.zero_grad()
        
    # Logging
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')




# Data Preparation: Before the loop, the code creates a mock dataset using TensorDataset and wraps it in a DataLoader called train_dl.

# Model: nn.Linear(input_size, output_size) defines a simple linear model with one input and one output feature. It's a building block for linear regression.

# Loss Function: nn.MSELoss(reduction='mean') defines the Mean Squared Error loss, which is standard for regression tasks. It calculates the average of the squared differences between predictions and actual values.

# Optimizer: torch.optim.SGD(model.parameters(), lr=learning_rate) initializes the Stochastic Gradient Descent optimizer with the model parameters and a learning rate.
