import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Parameters
model_dirname = os.path.join(os.curdir, "model_outputs")
model_filename = "CIFAR10_CNN"
model_path = os.path.join(model_dirname, model_filename)
quant_model_path = os.path.join(model_dirname, model_filename + "_quant.tflite")
intermediate_output_dirname = os.path.join(model_dirname, "intermediates")

if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)

if not os.path.exists(intermediate_output_dirname):
    os.makedirs(intermediate_output_dirname)

# Get dataset and re-scale the dataset
dataset = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

# Define the layers and sequential model
model = tf.keras.models.Sequential(name=model_filename)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.1,)

# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Generate model and compile
model_for_pruning.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

model_for_pruning.summary()

# Define callback for saving model (only weights)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1),
    tfmot.sparsity.keras.UpdatePruningStep(),
]

# Fitting model and evaluate the accuracy
model_for_pruning.fit(x_train, y_train,epochs=5, callbacks=callbacks)
model_for_pruning.evaluate(x_test,  y_test, verbose=2)

# Save pruned model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# Dataset type casting
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.uint8)

# Convert trained model into tflite model to quantize paramters
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

# Quantize the pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
quantized_and_pruned_tflite_model = converter.convert()

with open(quant_model_path, 'wb') as quant_model_file:
  quant_model_file.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quant_model_path)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(quant_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

evaluation, cnt = 0, 0

input_images = tf.data.Dataset.from_tensor_slices(x_test).batch(1)
output_labels = tf.data.Dataset.from_tensor_slices(y_test).batch(1)

for xt, yt in zip(input_images, output_labels):
    cnt += 1
    print(f"\riteration: {cnt}", end='')
    interpreter.set_tensor(input_details[0]['index'], xt)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == yt:
        evaluation += 1

print(f"\nquantized model acccuracy: {evaluation / cnt}")

# Save first intermediate output tensors
xt = list(input_images.as_numpy_iterator())[0]
interpreter.set_tensor(input_details[0]['index'], xt)
interpreter.invoke()

for tensor_details in interpreter.get_tensor_details():
    if 'Conv2D' in tensor_details['name'] or 'conv2d' in tensor_details['name']:
        tensor_name = tensor_details['name'].replace(';', '_').replace('/', '_').replace(':', '_')
        tensor = interpreter.get_tensor(tensor_details["index"])
        np.save(os.path.join(intermediate_output_dirname, f" {tensor_name}"), tensor)

print(f"intermediate layer outputs are saved at {intermediate_output_dirname}")