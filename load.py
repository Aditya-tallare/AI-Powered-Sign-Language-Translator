import tensorflow as tf
from tensorflow import keras

# Load the Keras model
model = keras.models.load_model("keras_model.h5")

# Print model summary
model.summary()
