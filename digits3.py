import tensorflow as tf
import numpy as np

# Define the input shape
#input_shape = (4, 28, 28, 3)
input_shape = (1, 2, 2, 1)
tf.random.set_seed(1234)
# Generate random input data
x = tf.random.normal(input_shape)
x = tf.constant(np.array([[[[1.0], [0.0]], [[1.0], [0.0]]]], dtype=np.float32))
# Create a Conv2D layer with 2 filters, a kernel size of 3, 'relu' activation,
# and using the input_shape specified
kernel_values = [[1, 2], [-4, 4]]  # Пример значений ядра
conv_layer = tf.keras.layers.Conv2D(1, 2, activation='relu', input_shape=input_shape[
                                                                         1:],
                                                                   kernel_initializer=tf.constant_initializer  (kernel_values))

# Apply the Conv2D layer to the input data
y = conv_layer(x)
print(y)

# Print the shape of the output tensor
print(y.shape)
