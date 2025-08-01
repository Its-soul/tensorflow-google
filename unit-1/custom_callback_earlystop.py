import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.callbacks import Callback, EarlyStopping # type: ignore

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Custom callback to stop at 95% accuracy
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.95:
            print("\nReached 95% accuracy, stopping training!")
            self.model.stop_training = True

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Using EarlyStopping + custom callback
earlystop = EarlyStopping(monitor='val_loss', patience=3)
callbacks = [MyCallback(), earlystop]

model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=callbacks)
