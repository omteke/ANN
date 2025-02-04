#Exp2:Simple Implementation of Gradient descent
import numpy as np
import matplotlib.pyplot as plt

# Sample data: x (input), y (actual output)
x = np.array([1, 2, 3, 4, 5])  # Input features
y = np.array([2, 4, 6, 8, 10]) # Actual outputs (for example, y = 2 * x)

# Initialize weight and bias
w = 0  # initial guess for weight
b = 0  # initial guess for bias

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Gradient Descent
for i in range(num_iterations):
    y_pred = w * x + b  # Predicted output
    error = y_pred - y  # Error (difference between predicted and true values)

    # Compute gradients (derivatives of cost function)
    dw = (2/len(x)) * np.sum(error * x)  # Gradient with respect to weight
    db = (2/len(x)) * np.sum(error)      # Gradient with respect to bias

    # Update the weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Optionally print progress
    if i % 100 == 0:
        print(f"Iteration {i}: w = {w:.4f}, b = {b:.4f}, MSE = {np.mean(error**2):.4f}")

# Plot the result
plt.scatter(x, y, color='blue', label='True values')
plt.plot(x, w * x + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

print(f"Final weight (w) = {w:.4f}, Final bias (b) ={b:.4f}")
