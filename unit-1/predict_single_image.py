import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model = tf.keras.models.load_model('unit-1/saved_model/my_model.h5')  # Update this if saved elsewhere

# Load and preprocess the sample image
img_path = img_path = 'cats_and_dogs_filtered/validation/dogs/dog.2373.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Predict
prediction = model.predict(x)
if prediction[0] > 0.5:
    print("It's a dog.")
else:
    print("It's a cat.")
