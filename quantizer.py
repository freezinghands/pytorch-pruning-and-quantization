import tensorflow as tf
import numpy as np
import os


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
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.uint8)

# Convert trained model into tflite model to quantize paramters
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(model_dirname, model_filename))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
quant_model = converter.convert()

with open(os.path.join(model_dirname, model_filename + "_quant.tflite"), 'wb') as quant_model_file:
    quant_model_file.write(quant_model)

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
    tensor_name = tensor_details["name"].replace('/', '_')
    if len(tensor_name) < 20:
        tensor = interpreter.get_tensor(tensor_details["index"])
        np.save(os.path.join(intermediate_output_dirname, f"test0_{tensor_name}"), tensor)

print(f"intermediate layer outputs are saved at {intermediate_output_dirname}")