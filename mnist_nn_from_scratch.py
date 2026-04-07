# Handwritten Digit Classifier from Scratch
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 1. LOAD THE MNIST DATASET

print("Downloading MNIST dataset...")

# Download dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Convert dataset from pandas DataFrame to NumPy arrays
X = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int32)

# Normalize pixel values from [0,255] → [0,1]
X = X / 255.0

# Split dataset into training and testing sets
X_train = X[:60000]
X_test = X[60000:]

y_train = y[:60000]
y_test = y[60000:]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 2. ONE-HOT ENCODING FOR LABELS

# Convert labels into one-hot encoded vectors
# Example: digit 3 → [0 0 0 1 0 0 0 0 0 0]

def one_hot(y):
    encoded = np.zeros((y.size, 10))
    encoded[np.arange(y.size), y] = 1
    return encoded

y_train_ohe = one_hot(y_train)

# 3. ACTIVATION FUNCTIONS

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU used in backpropagation
def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax activation for output layer
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # improve numerical stability
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


# 4. NETWORK ARCHITECTURE

# 784 input neurons (28x28 pixels)
# 128 hidden neurons
# 10 output neurons (digits 0-9)

input_size = 784
hidden_size = 128
output_size = 10

# He initialization for weights
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))

# 5. TRAINING PARAMETERS

learning_rate = 0.01
epochs = 15
batch_size = 128

loss_history = []
accuracy_history = []

# 6. TRAINING LOOP

for epoch in range(epochs):

    # Shuffle training data each epoch
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train_ohe = y_train_ohe[permutation]
    y_train = y_train[permutation]

    # Mini-batch training
    for i in range(0, X_train.shape[0], batch_size):

        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_ohe[i:i+batch_size]

        # ---------- Forward Propagation ----------

        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # ---------- Cross-Entropy Loss ----------

        loss = -np.mean(np.sum(y_batch * np.log(A2 + 1e-8), axis=1))

        # ---------- Backpropagation ----------

        dZ2 = A2 - y_batch
        dW2 = A1.T @ dZ2 / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)

        dW1 = X_batch.T @ dZ1 / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        # ---------- Gradient Descent Weight Update ----------

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # ---------- Calculate Training Accuracy ----------

    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    predictions = np.argmax(A2, axis=1)
    train_accuracy = np.mean(predictions == y_train)

    loss_history.append(loss)
    accuracy_history.append(train_accuracy)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Accuracy: {train_accuracy*100:.2f}%")

# 7. TESTING PHASE

Z1 = X_test @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

test_predictions = np.argmax(A2, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

print("\nFinal Test Accuracy:", round(test_accuracy * 100, 2), "%")

# VISUALIZATION
# TRAINING LOSS AND ACCURACY

fig1, axes = plt.subplots(1, 2, figsize=(12,5))

# Plot training loss
axes[0].plot(loss_history)
axes[0].set_title("Training Loss vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Plot training accuracy
axes[1].plot(accuracy_history)
axes[1].set_title("Training Accuracy vs Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

fig1.tight_layout()

# FIVE SAMPLE PREDICTIONS

fig2, axes = plt.subplots(1, 5, figsize=(12,3))

for i in range(5):
    axes[i].imshow(X_test[i].reshape(28,28), cmap="gray")
    axes[i].set_title(f"P:{test_predictions[i]}\nT:{y_test[i]}")
    axes[i].axis("off")

fig2.tight_layout()

# Display both windows
plt.show()
