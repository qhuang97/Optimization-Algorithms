import numpy as np
import matplotlib.pyplot as plt

# Generate some fake data to fit
np.random.seed(0)
x = np.linspace(-1, 1, 100)
y = x ** 2 + np.random.normal(0, 0.1, 100)

# Define the model
w = np.random.normal(0, 0.1)
b = np.zeros(1)

# Define the loss function
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def predict(x, w, b):
    return x * w + b

# Define the gradients
def gradient_w(x, y_pred, y_true):
    return 2 * np.mean((y_pred - y_true) * x)

def gradient_b(y_pred, y_true):
    return 2 * np.mean(y_pred - y_true)

# Define the SGD function
def sgd(x, y, learning_rate, w, b, epochs):
    losses = []
    for i in range(epochs):
        for j in range(x.shape[0]):
            # Get the current learning rate
            current_learning_rate = learning_rate / (1 + i * j / epochs)
            # Make a prediction with current parameters
            y_pred = predict(x[j], w, b)
            # Compute the loss
            loss = mean_squared_error(y_pred, y[j])
            # Append the loss to the list of losses
            losses.append(loss)
            # Compute the gradients
            dw = gradient_w(x[j], y_pred, y[j])
            db = gradient_b(y_pred, y[j])
            # Update the parameters
            w = w - current_learning_rate * dw
            b = b - current_learning_rate * db
    return losses, w, b

# Run SGD
losses, w, b = sgd(x, y, 0.01, w, b, 100)

# Plot the loss over time
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
