import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix
import numpy as np

# Load test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)  # Add channel dimension

# Load model
model = load_model("mnist_cnn_model.h5")

# Predict
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MNIST")
plt.show()
