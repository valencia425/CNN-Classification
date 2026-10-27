# ==============================================================
# PROJECT 2:
# DEVELOPMENT OF A CNN FOR DOG-CAT IMAGE CLASSIFICATION
# USING THREE CONVOLUTIONAL LAYERS
#
# Dataset: Hugging Face - louiecerv/cats_dogs_dataset
# Environment: Google Colab
# ==============================================================

# 1. INSTALL AND IMPORT LIBRARIES

!pip install -q datasets

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling
)

from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)


# 2. LOAD DATASET FROM HUGGING FACE

print("\nDownloading dataset from Hugging Face...")

dataset = load_dataset("louiecerv/cats_dogs_dataset")

train_data = dataset["train"]
test_data = dataset["test"]

print("\nDataset loaded successfully!")
print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

IMG_SIZE = 128
BATCH_SIZE = 32

# Class labels: 0 = cats, 1 = dogs
class_names = ["cats", "dogs"]


# 3. ROBUST IMAGE CONVERSION FUNCTION

def convert_to_pil(image):
    """
    Convert image data from Hugging Face into a PIL RGB image.
    Supports PIL images, dictionaries, NumPy arrays, and lists.
    """

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    elif isinstance(image, dict):

        if image.get("bytes") is not None:
            return Image.open(
                io.BytesIO(image["bytes"])
            ).convert("RGB")

        elif image.get("path") is not None:
            return Image.open(
                image["path"]
            ).convert("RGB")

        else:
            raise ValueError(
                "Image dictionary contains neither bytes nor path."
            )

    elif isinstance(image, np.ndarray):

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image).convert("RGB")

    elif isinstance(image, list):

        image_array = np.asarray(image)

        if image_array.dtype != np.uint8:
            image_array = np.clip(
                image_array, 0, 255
            ).astype(np.uint8)

        return Image.fromarray(image_array).convert("RGB")

    else:
        raise TypeError(
            f"Unsupported image format: {type(image)}"
        )


# 4. PREPARE IMAGES AND LABELS

def prepare_data(split):
    images = []
    labels = []

    for item in split:

        image = convert_to_pil(item["image"])

        # Resize images to 128 × 128
        image = image.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.LANCZOS
        )

        images.append(
            np.asarray(image, dtype=np.uint8)
        )

        labels.append(int(item["label"]))

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int32)

    return X, y


print("\nPreparing training images...")

X_train, y_train = prepare_data(train_data)

print("Preparing testing images...")

X_test, y_test = prepare_data(test_data)

print("\nDATASET PREPARATION COMPLETED")
print("-" * 45)
print("Training image shape:", X_train.shape)
print("Training label shape:", y_train.shape)
print("Testing image shape:", X_test.shape)
print("Testing label shape:", y_test.shape)

print("\nTraining class distribution:")
print("Cats:", np.sum(y_train == 0))
print("Dogs:", np.sum(y_train == 1))


# 5. DISPLAY SAMPLE IMAGES

plt.figure(figsize=(10, 7))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(X_train[i])

    plt.title(class_names[y_train[i]])

    plt.axis("off")

plt.suptitle("Sample Images from the Cats and Dogs Dataset")
plt.tight_layout()
plt.show()


# 6. BUILD CNN WITH EXACTLY THREE CONVOLUTIONAL LAYERS

model = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Normalize pixel values from 0-255 to 0-1
    Rescaling(1.0 / 255),

    # Convolutional Layer 1
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    MaxPooling2D((2, 2)),

    # Convolutional Layer 3
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    MaxPooling2D((2, 2)),

    # Classification layers
    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.5),

    # Binary classification output
    Dense(1, activation="sigmoid")
])


# 7. COMPILE MODEL

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN MODEL SUMMARY")

model.summary()


# 8. CREATE VALIDATION SET FROM TRAINING DATA

# Shuffle training indices reproducibly
rng = np.random.default_rng(42)

indices = rng.permutation(len(X_train))

validation_size = int(0.20 * len(X_train))

val_indices = indices[:validation_size]
train_indices = indices[validation_size:]

X_val = X_train[val_indices]
y_val = y_train[val_indices]

X_train_final = X_train[train_indices]
y_train_final = y_train[train_indices]

print("\nDATA SPLIT")
print("-" * 45)
print("Training images:", len(X_train_final))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))


# 9. TRAIN MODEL

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

EPOCHS = 15

print("\nTRAINING CNN MODEL...")

history = model.fit(
    X_train_final,
    y_train_final,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed successfully!")


# 10. EVALUATE MODEL ON UNSEEN TEST DATA

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nMODEL EVALUATION")
print("-" * 45)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# 11. GENERATE TEST PREDICTIONS

print("\nGenerating predictions...")

y_prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()

y_pred = (y_prob >= 0.5).astype(np.int32)


# 12. CLASSIFICATION REPORT

print("\nCLASSIFICATION REPORT")
print("-" * 45)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )
)

print(
    "Classification Accuracy:",
    f"{accuracy_score(y_test, y_pred) * 100:.2f}%"
)


# 13. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("CNN Dog-Cat Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm)


# 14. PLOT TRAINING AND VALIDATION ACCURACY / LOSS

plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)


# Loss graph
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 15. PREDICT A SAMPLE TEST IMAGE

sample_index = 0

sample_image = X_test[sample_index]
actual_label = int(y_test[sample_index])

probability = float(
    model.predict(
        sample_image[np.newaxis, ...],
        verbose=0
    )[0][0]
)

predicted_label = int(probability >= 0.5)

confidence = (
    probability
    if predicted_label == 1
    else 1.0 - probability
)

plt.figure(figsize=(5, 5))

plt.imshow(sample_image)

plt.title(
    f"Actual: {class_names[actual_label]}\n"
    f"Predicted: {class_names[predicted_label]}\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")
plt.show()

print("\nSAMPLE PREDICTION")
print("-" * 45)
print("Actual class:", class_names[actual_label])
print("Predicted class:", class_names[predicted_label])
print(f"Prediction confidence: {confidence * 100:.2f}%")


# 16. SAVE TRAINED MODEL

model.save("dog_cat_cnn.keras")

print("\nTrained model saved as dog_cat_cnn.keras")


# 17. SAVE TEST PREDICTIONS

import pandas as pd

results = pd.DataFrame({
    "Actual_Label": y_test,
    "Predicted_Label": y_pred,
    "Dog_Probability": y_prob
})

results.to_csv(
    "dog_cat_predictions.csv",
    index=False
)

print("Predictions saved as dog_cat_predictions.csv")


# 18. FINAL RESULT

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Project: Dog-Cat Image Classification")
print("Dataset: Hugging Face")
print("Dataset Name: louiecerv/cats_dogs_dataset")
print("Training Images:", len(X_train_final))
print("Validation Images:", len(X_val))
print("Testing Images:", len(X_test))
print("Convolutional Layers: 3")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Saved Model: dog_cat_cnn.keras")
print("Saved Predictions: dog_cat_predictions.csv")

print("=" * 60)
