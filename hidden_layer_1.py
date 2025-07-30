import numpy as np
import matplotlib.pyplot as plt

# True function: sin(x)
def actual_function(x):
    return np.sin(x)

# Data generator
def true_output():
    rand = np.random.rand(100) * 5
    return rand, actual_function(rand)

# Plotting
def plot(actual, predicted=None):
    x, y = actual
    plt.scatter(x, y, color='blue', s=10, alpha=0.6, label='true_y')
    if predicted is not None:
        pred_x, pred_y = predicted
        plt.scatter(pred_x, pred_y, color='red', s=10, alpha=0.6, label='pred_y')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Loss function: LMS
def least_mean_square(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent
def gradient_descent(x, error):
    grad_w = -2 * np.mean(x * error)
    grad_b = -2 * np.mean(error)
    return grad_b, grad_w


# Hidden Unit
class HiddenUnit_1x1():
    def __init__(self, bias, weight):
        self.bias = bias
        self.weight = weight

    def update(self, bias, weight):
        self.bias = bias
        self.weight = weight

    def output(self, input):
        return self.bias + self.weight * input


# Hidden Layer with 1 Hidden Unit
class HiddenLayer_1x1x1():
    def __init__(self, hu1_parameter, loss_function, optimizer):
        self.hu1 = HiddenUnit_1x1(hu1_parameter[0], hu1_parameter[1])
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward_pass(self, x):
        h1 = self.hu1.output(x)
        return h1 

    def train(self, x_train, y_train, epochs=1000, lr=0.01):
        loss_history = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward_pass(x_train)
            loss = self.loss_function(y_train, y_pred)
            loss_history.append(loss)
            # Compute error
            error = y_train - y_pred

            # Compute gradients for hidden unit
            error = y_train - self.hu1.output(x_train)
            grad_w = -2 * np.mean(x_train * error)
            grad_b = -2 * np.mean(error)

            # Update hidden unit weights
            self.hu1.weight -= lr * grad_w
            self.hu1.bias -= lr * grad_b
        
        return loss_history







# Generate training data
x_train, y_train = true_output()

# Create the model with:
# - output bias = 0
# - hidden unit (bias=0, weight=0)
# - LMS loss and gradient descent optimizer
model = HiddenLayer_1x1x1(
    hu1_parameter=(np.random.randn() * 0.1, np.random.randn() * 0.1),
    loss_function=least_mean_square
)

# Train the model
losses = model.train(x_train, y_train, epochs=1000, lr=0.01)

# Predict using the trained model
y_pred = model.forward_pass(x_train)

# Plot actual vs predicted
plot((x_train, y_train), (x_train, y_pred))

# Plot loss over training epochs
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
