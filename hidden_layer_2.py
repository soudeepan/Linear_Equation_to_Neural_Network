import numpy as np
import matplotlib.pyplot as plt

# True function: sin(x)
def actual_function(x):
    return np.sin(x)

# Generate synthetic data
def true_output():
    rand = np.random.rand(100) * 5  # x in [0,10]
    return rand, actual_function(rand)

# Plot actual and predicted values
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

# Loss function: Least Mean Square
def least_mean_square(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Activation function
def tanh(x):
    return np.tanh(x)

# Derivative of tanh for backpropagation
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Single hidden unit
class HiddenUnit_1x1:
    def __init__(self, bias, weight):
        self.bias = bias
        self.weight = weight

    def output(self, input):
        z = self.bias + self.weight * input
        self.last_z = z  # Save pre-activation for backprop
        return tanh(z)

# Neural network: 1x2x1
class HiddenLayer_1x2x1:
    def __init__(self, hu1_params, hu2_params, out_bias, out_weights, loss_function, lr=0.01):
        self.hu1 = HiddenUnit_1x1(*hu1_params)
        self.hu2 = HiddenUnit_1x1(*hu2_params)
        self.out_bias = out_bias
        self.out_weights = np.array(out_weights)  # [w1, w2]
        self.loss_function = loss_function
        self.lr = lr

    def forward_pass(self, x):
        h1 = self.hu1.output(x)
        h2 = self.hu2.output(x)
        self.last_h = np.vstack((h1, h2))  # Shape: (2, N)
        y = self.out_bias + self.out_weights[0] * h1 + self.out_weights[1] * h2
        return y

    def train(self, x_train, y_train, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward_pass(x_train)
            loss = self.loss_function(y_train, y_pred)
            loss_history.append(loss)

            # Error
            error = y_train - y_pred

            # Gradients for output layer
            grad_w_out = -2 * np.mean(self.last_h * error, axis=1)  # shape: (2,)
            grad_b_out = -2 * np.mean(error)

            # Update output weights and bias
            self.out_weights -= self.lr * grad_w_out
            self.out_bias -= self.lr * grad_b_out

            # Gradients for hidden units (backprop through tanh)
            d_h1 = error * self.out_weights[0] * tanh_derivative(self.hu1.last_z)
            d_h2 = error * self.out_weights[1] * tanh_derivative(self.hu2.last_z)

            grad_w1 = -2 * np.mean(d_h1 * x_train)
            grad_b1 = -2 * np.mean(d_h1)

            grad_w2 = -2 * np.mean(d_h2 * x_train)
            grad_b2 = -2 * np.mean(d_h2)

            # Update hidden weights and biases
            self.hu1.weight -= self.lr * grad_w1
            self.hu1.bias -= self.lr * grad_b1

            self.hu2.weight -= self.lr * grad_w2
            self.hu2.bias -= self.lr * grad_b2

        return loss_history

# ==== Run Training ====

# Generate training data
x_train, y_train = true_output()

# Create model
model = HiddenLayer_1x2x1(
    hu1_params = (np.random.randn(), np.random.randn()),
    hu2_params = (np.random.randn(), np.random.randn()),
    out_bias = np.random.randn(),
    out_weights = [np.random.randn(), np.random.randn()],
    loss_function=least_mean_square,
    lr=0.01
)

# Train the model
losses = model.train(x_train, y_train, epochs=1000)

# Predict using trained model
y_pred = model.forward_pass(x_train)

# Plot results
plot((x_train, y_train), (x_train, y_pred))

# Plot training loss
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
