import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

# --- Hyperparameters ---
epochs = 50
alpha = 0.01
batch_size = 32  # Keeping small for now
input_size = 128 * 128
hidden_size = 128
output_size = 360  # 360 classes for angles

error = []

# --- Transform pipeline ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Load dataset ---
dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)

# --- Utility functions ---
def deriv_ReLU(Z):
    return Z > 0

def one_hot_angle(angle):
    one_hot = np.zeros((output_size, 1))
    one_hot[angle] = 1
    return one_hot

def parameters():
    W1 = np.random.rand(hidden_size, input_size) - 0.5
    b1 = np.random.rand(hidden_size, 1) - 0.5
    W2 = np.random.rand(output_size, hidden_size) - 0.5
    b2 = np.random.rand(output_size, 1) - 0.5
    return W1, b1, W2, b2

# --- Forward & Backward Pass Classes ---
class ForwardPass:
    def __init__(self, W1, b1, W2, b2, input_data):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.input_data = input_data

    def forward(self):
        Z1 = np.dot(self.W1, self.input_data) + self.b1
        A1 = np.maximum(Z1, 0)  # ReLU
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.maximum(Z2, 0)  # ReLU (or softmax if needed)
        return Z1, A1, Z2, A2

class BackwardPass:
    def __init__(self, j, W1, b1, W2, b2, Z1, A1, Z2, A2, X, angle, alpha=0.01):
        self.j = j
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.Z1 = Z1
        self.A1 = A1
        self.Z2 = Z2
        self.A2 = A2
        self.X = X
        self.angle = angle
        self.alpha = alpha

    def back_propagation(self):
        Y = one_hot_angle(self.angle)
        m = self.X.shape[1]

        dZ2 = 2 * (self.A2 - Y)
        dW2 = (1 / m) * dZ2.dot(self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * deriv_ReLU(self.Z1)
        dW1 = (1 / m) * dZ1.dot(self.X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2

        return self.W1, self.b1, self.W2, self.b2

# --- Training loop ---
W1, b1, W2, b2 = parameters()

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for i in range(batch_size):
        image_tensor, _ = dataset[i]
        angle = np.random.randint(0, 360)

        # Rotate image and flatten
        rotated_tensor = torchvision.transforms.functional.rotate(image_tensor, angle)
        rotated_np = rotated_tensor.permute(1, 2, 0).numpy()
        X = rotated_np.flatten().reshape(-1, 1) / 255.0

        # Forward + Backward
        forward = ForwardPass(W1, b1, W2, b2, X)
        Z1, A1, Z2, A2 = forward.forward()

        backward = BackwardPass(i, W1, b1, W2, b2, Z1, A1, Z2, A2, X, angle, alpha)
        W1, b1, W2, b2 = backward.back_propagation()

        if i % 2 == 0:
            predicted_angle = np.argmax(A2)
            print(f"[{i}] Predicted: {predicted_angle}, Actual: {angle}")

print("\n✅ Training complete!")

# --- Testing some images ---
print("\n🔍 Testing predictions...")
for i in range(100):
    image_tensor, _ = dataset[i]
    angle = np.random.randint(0, 360)
    rotated_tensor = torchvision.transforms.functional.rotate(image_tensor, angle)
    rotated_np = rotated_tensor.permute(1, 2, 0).numpy()
    X = rotated_np.flatten().reshape(-1, 1) / 255.0

    forward = ForwardPass(W1, b1, W2, b2, X)
    _, _, _, A2 = forward.forward()
    predicted_angle = np.argmax(A2)

    print(f"Test {i+1}: Predicted angle = {predicted_angle}, Actual angle = {angle}")
    
    error.append(abs(angle-predicted_angle) )

    print(f"error,{ error[i]}")

"""
    # Show image
    plt.imshow(rotated_np)
    plt.title(f"Predicted: {predicted_angle}° | Actual: {angle}°")
    plt.axis('off')
    plt.show()
"""
average_error = sum(error) / len(error)
print(f"\nAverage error across test images: {average_error:.2f}°")

"""
I asked GPT to edit my code and it made it worse. 
ave error of 115 degrees is not good.

"""