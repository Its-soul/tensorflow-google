import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np

# Load dataset
(x_train, _), _ = mnist.load_data()
x_train = np.expand_dims(x_train, -1)  # Add channel dimension

# Define augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Fit generator
datagen.fit(x_train)

# Show some augmented examples
fig, ax = plt.subplots(1, 5, figsize=(10, 2))
for images, _ in datagen.flow(x_train, np.zeros(len(x_train)), batch_size=5):
    for i in range(5):
        ax[i].imshow(images[i].reshape(28, 28), cmap='gray')
        ax[i].axis('off')
    break  # Stop after one batch
plt.show()
