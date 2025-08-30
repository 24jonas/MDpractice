from turtle import color
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torchvision
import keras
import tensorflow as tf

# Constants

epochs = 6
alpha = 0.01
batch_size = 100

angles = np.arange(0, 360, 1)  # Angles from 0 to 359
error = []
angle = []
for i in range(72):
    angle.append(i * 5)



# Data preprocessing

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((128, 128)),               
    transforms.ToTensor()                        
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)

def rotate_image(image, angle):
    return torchvision.transforms.functional.rotate(image, angle)
def preprocess_dataset(dataset, max_samples=None):
    X, Y = [], []
    count = 0
    for image in dataset:
        A = np.random.randint(0, 72)    # random index 0–71
        B = angle[A]                    # actual rotation angle (0, 5, 10, …, 355)
        
        current_image = dataset[count][0]   # PyTorch tensor [1, H, W]
        rotated_image = torchvision.transforms.functional.rotate(current_image, B)
        
        rotated_image_np = rotated_image.permute(1, 2, 0).numpy()  # [H, W, 1]
        X.append(rotated_image_np)
        
        # Store class index (0–71), not raw angle
        Y.append(A)  
    
        count += 1
        if max_samples and count >= max_samples:
            break
    
    # Normalize images only
    X = np.array(X) / 255.0   # (num_samples, 128,128,1), values in [0,1]
    Y = np.array(Y)           # integer labels 0–71
    
    return X, Y

X_all, Y_all = preprocess_dataset(dataset)

# Split train/dev
split = batch_size
X_dev, Y_dev = X_all[:split], Y_all[:split]
X_train, Y_train = X_all[split:], Y_all[split:]

# CNN Model

model = keras.Sequential([
    keras.layers.Input(shape=(128,128,1)),

    keras.layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(72, activation='softmax')  # 72 angle classes
])

# Loss + optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

# Metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(batch_size)

train_acc_history, val_acc_history = [], []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_loss = 0.0

    # Training
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, predictions)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track loss + accuracy
        epoch_loss += loss_value.numpy()
        train_acc_metric.update_state(y_batch, predictions)

    train_acc = train_acc_metric.result().numpy()
    train_acc_metric.reset_state()
    print(f"Train loss: {epoch_loss/len(train_dataset):.4f}, Train acc: {train_acc:.4f}")

    # Validation
    for x_batch, y_batch in val_dataset:
        val_preds = model(x_batch, training=False)
        val_acc_metric.update_state(y_batch, val_preds)
    val_acc = val_acc_metric.result().numpy()
    val_acc_metric.reset_state()
    print(f"Validation acc: {val_acc:.4f}")

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

# Plot training curves

plt.plot(train_acc_history, label="Train Acc")
plt.plot(val_acc_history, label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
