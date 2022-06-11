import os

import numpy as np
import tensorflow as tf

# Parameters
model_dirname = os.path.join(os.curdir, "model_outputs")
model_filename = "MNIST_MLP"
model_path = os.path.join(model_dirname, model_filename)
quant_model_path = os.path.join(model_dirname, model_filename + "_quant.tflite")
intermediate_output_dirname = os.path.join(model_dirname, "intermediates")

if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)

if not os.path.exists(intermediate_output_dirname):
    os.makedirs(intermediate_output_dirname)

# Get dataset and re-scale the dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train / 255.0

# Define the layers
layers = list()
layers.append(tf.keras.layers.Flatten(input_shape=(28, 28)))
layers.append(tf.keras.layers.Dense(128, activation='relu'))
layers.append(tf.keras.layers.Dropout(0.2))
layers.append(tf.keras.layers.Dense(10, activation='softmax'))

# Generate model and compile
model = tf.keras.models.Sequential(layers)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callback for saving model (only weights)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True, verbose=1)

# Fitting model and evaluate the accuracy
model.fit(x_train, y_train, epochs=5, callbacks=[cp_callback])
model.evaluate(x_test,  y_test, verbose=2)

