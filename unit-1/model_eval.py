import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your saved model
model = tf.keras.models.load_model('unit-1/saved_model/my_model.h5')

# Path to your test (validation) directory
test_dir = 'cats_and_dogs_filtered/validation'

# Image data preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

# Generating test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Evaluating the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
