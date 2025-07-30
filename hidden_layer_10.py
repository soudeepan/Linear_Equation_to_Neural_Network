import numpy as np
import matplotlib.pyplot as plt

# True function: sin(x)
def actual_function(x):
    return np.sin(x)

# Generate synthetic data
def true_output():
    rand = np.random.rand(100) * 5
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

class HiddenLayer_1xNx1:
    def __init__(self, hu_params, out_bias, out_weights, loss_function, lr=0.01):
        self.hidden_units = [HiddenUnit_1x1(b, w) for b, w in hu_params]
        self.out_bias = out_bias
        self.out_weights = np.array(out_weights)
        self.loss_function = loss_function
        self.lr = lr

    def forward_pass(self, x):
        h_outputs = [hu.output(x) for hu in self.hidden_units]
        self.last_h = np.vstack(h_outputs)  # Shape: (N, batch)
        y = self.out_bias + np.dot(self.out_weights, self.last_h)
        return y

    def train(self, x_train, y_train, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward_pass(x_train)
            loss = self.loss_function(y_train, y_pred)
            loss_history.append(loss)

            # Error
            error = y_train - y_pred

            # Output layer gradients
            grad_w_out = -2 * np.mean(self.last_h * error, axis=1)
            grad_b_out = -2 * np.mean(error)

            # Update output layer
            self.out_weights -= self.lr * grad_w_out
            self.out_bias -= self.lr * grad_b_out

            # Hidden layer gradients and updates
            for i, hu in enumerate(self.hidden_units):
                d_hi = error * self.out_weights[i] * tanh_derivative(hu.last_z)
                grad_wi = -2 * np.mean(d_hi * x_train)
                grad_bi = -2 * np.mean(d_hi)
                hu.weight -= self.lr * grad_wi
                hu.bias -= self.lr * grad_bi

        return loss_history


# Generate training data
x_train, y_train = true_output()


# 5 hidden units
model_10 = HiddenLayer_1xNx1(
    hu_params=[(np.random.randn(), np.random.randn()) for _ in range(10)],
    out_bias=np.random.randn(),
    out_weights=np.random.randn(10),
    loss_function=least_mean_square,
    lr=0.01
)

loss_10 = model_10.train(x_train, y_train, epochs=1000)
y_pred_10 = model_10.forward_pass(x_train)

plot((x_train, y_train), (x_train, y_pred_10))
plt.plot(loss_10)
plt.title("1x10x1 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
