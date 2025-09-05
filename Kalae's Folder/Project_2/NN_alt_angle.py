import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torchvision
import keras
import tensorflow as tf

# Constants
epochs = 30
alpha = 0.001
batch_size = 32

# Create angle mappings
angle_values = [i * 5 for i in range(72)]  # [0, 5, 10, ..., 355]

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((64, 64)),               
    transforms.ToTensor()                        
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)

def preprocess_dataset(dataset, train_ratio=0.8):
    """
    Proper train/val split: completely separate images in train vs validation
    No data leakage - validation images never seen during training
    """
    total_images = len(dataset)
    train_size = int(train_ratio * total_images)
    
    # Split images (not rotations) into train and validation
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_images))
    
    print(f"Using {len(train_indices)} base images for training")
    print(f"Using {len(val_indices)} base images for validation")
    
    def create_rotated_dataset(image_indices, rotations_per_image=10):
        X, Y = [], []
        
        for img_idx in image_indices:
            base_image = dataset[img_idx][0]  # Get the original image tensor
            
            # Create multiple random rotations of this image
            for _ in range(rotations_per_image):
                angle_idx = np.random.randint(0, 72)
                rotation_angle = angle_values[angle_idx]
                
                rotated_image = torchvision.transforms.functional.rotate(base_image, rotation_angle)
                rotated_image_np = rotated_image.permute(1, 2, 0).numpy()
                
                X.append(rotated_image_np)
                Y.append(angle_idx)
        
        return np.array(X) / 255.0, np.array(Y)
    
    # Create training set with more augmentation
    X_train, Y_train = create_rotated_dataset(train_indices, rotations_per_image=15)
    
    # Create validation set with fewer rotations (different base images!)
    X_val, Y_val = create_rotated_dataset(val_indices, rotations_per_image=10)
    
    return X_train, Y_train, X_val, Y_val

X_train, Y_train, X_val, Y_val = preprocess_dataset(dataset)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Improved CNN Model focused on rotation-invariant features
model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 1)),

    # First block - detect basic edges/features
    keras.layers.Conv2D(32, (5,5), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (5,5), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),

    # Second block - more complex patterns
    keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

    # Third block - high-level features
    keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

    # Fourth block - even higher level
    keras.layers.Conv2D(256, (3,3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.4),

    # Dense layers for classification
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(72, activation='softmax')  # 72 angle classes
])

# Compile with better optimizer settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Enhanced callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=8, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=4, 
    min_lr=1e-7,
    verbose=1
)

# Create TensorFlow datasets for manual training loop
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)

# Manual training loop
train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

# Metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

print("Starting manual training loop...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    new_training_set = train_dataset.shuffle(10000).batch(batch_size)
    # Reset metrics at start of epoch
    train_acc_metric.reset_state()
    val_acc_metric.reset_state()
    
    epoch_train_loss = 0.0
    num_batches = 0
    
    # Training phase
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, predictions)
        
        # Compute gradients and update weights
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Update metrics
        epoch_train_loss += loss_value.numpy()
        train_acc_metric.update_state(y_batch, predictions)
        num_batches += 1
        
        # Print progress every 50 batches
        if step % 50 == 0:
            current_acc = train_acc_metric.result().numpy()
            print(f"  Step {step}: loss = {loss_value:.4f}, acc = {current_acc:.4f}")
    
    # Get final training metrics for this epoch
    train_acc = train_acc_metric.result().numpy()
    avg_train_loss = epoch_train_loss / num_batches
    
    # Validation phase
    epoch_val_loss = 0.0
    val_batches = 0
    
    for x_batch, y_batch in val_dataset:
        val_predictions = model(x_batch, training=False)
        val_loss = loss_fn(y_batch, val_predictions)
        
        epoch_val_loss += val_loss.numpy()
        val_acc_metric.update_state(y_batch, val_predictions)
        val_batches += 1
    
    val_acc = val_acc_metric.result().numpy()
    avg_val_loss = epoch_val_loss / val_batches
    
    # Store history
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    
    # Print epoch summary
    print(f"  Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.4f}")
    print(f"  Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.4f}")
    
    # Simple early stopping check
    if epoch > 5 and val_acc < max(val_acc_history[:-1]) - 0.05:

        print(f"  Early stopping triggered - validation accuracy decreased significantly")
        break
    
    # Simple learning rate reduction
    if epoch > 0 and val_loss_history[-1] > val_loss_history[-2]:
        new_lr = optimizer.learning_rate * 0.9
        optimizer.learning_rate.assign(new_lr)
        print(f"  Reduced learning rate to {new_lr:.6f}")

# Create history dictionary for compatibility with plotting code
history = {
    'history': {
        'accuracy': train_acc_history,
        'val_accuracy': val_acc_history,
        'loss': train_loss_history,
        'val_loss': val_loss_history
    }
}

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label="Train Acc", linewidth=2)
plt.plot(history.history['val_accuracy'], label="Val Acc", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.grid(True, alpha=0.3)
