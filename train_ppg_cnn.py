import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load segmented PPG dataset.
# Expected X shape: (2892, 300, 1)
# Expected y shape: (2892,)
X = np.load("X.npy").astype("float32")
y = np.load("y.npy").astype("int32")

# Stratified 80:20 train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# CNN architecture preserved from the supplied script.
# Note: The accompanying report shows a three-class output and dropout,
# whereas this recovered script specifies two output classes and no dropout.
model = Sequential([
    Input(shape=(300, 1)),
    Conv1D(filters=32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.20,
    verbose=1,
)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test, predictions[:, 1])

print("\n===== CNN PERFORMANCE =====")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model.save("PPG_CNN_Model.keras")
print("\nModel saved successfully.")
