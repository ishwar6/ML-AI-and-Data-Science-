# make regression without torch.nn module

torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)
 
def loss_fn(input, target):
    print("loss calculation", input, target)
    return (input-target).pow(2).mean()

# Defines the linear regression model. xb @ weight uses the matrix multiplication (@) operator to apply the weight to the input batch xb, and then adds the bias. 
# The @ operator is used for matrix multiplication.

def model(xb):
    return xb @ weight + bias

# When we call loss.backward(), PyTorch computes the derivatives (gradients) of the loss function with respect to the model parameters 
# (in this case, weight and bias) that have requires_grad=True. However, the gradients are not stored in loss.grad 
# but rather in the .grad attribute of each parameter tensor (weight and bias).
# This is a key concept in automatic differentiation frameworks like PyTorch.

learning_rate = 0.001
num_epochs = 2
log_epochs = 10

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        print(pred)
        loss = loss_fn(pred, y_batch)
        print("loss", loss)
        loss.backward()

        with torch.no_grad():
            print("weight", weight)
            # we are using weight.grad here to update the weights
            weight -= weight.grad * learning_rate
            print("weight  now", weight)
            bias -= bias.grad * learning_rate
            weight.grad.zero_()
            bias.grad.zero_()
 

    print(f'Epoch {epoch}  Loss {loss.item():.4f}')


# Why Compute Gradients?
# The gradients represent the slope of the loss function with respect to each parameter. 
# In other words, they tell us how to change the parameters to minimize the loss.
# By knowing the direction in which each parameter should be adjusted (increase or decrease) 
# and how significant the adjustment should be, we can update the parameters to make our model more accurate.
 
