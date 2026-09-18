import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Dataset paths
# -----------------------------
TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

# -----------------------------
# Image settings
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load training dataset
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# -----------------------------
# Normalize images
# -----------------------------
normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# -----------------------------
# Build CNN
# -----------------------------
model = models.Sequential([

    layers.Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(224,224,3)
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )

])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Model Summary
# -----------------------------
model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("\n===== TRAINING RESULTS =====")
print(f"Training Accuracy : {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# -----------------------------
# Save Model
# -----------------------------
model.save("tb_model.keras")

print("\nTraining Complete!")
print("Model saved as tb_model.keras")