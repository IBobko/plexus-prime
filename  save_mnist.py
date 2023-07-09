import os
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Directory to save the images
directory = "mnist_img"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Iterate over images in the test dataset
for i in range(len(x_test)):
    image = x_test[i]
    label = y_test[i]

    # Create a filename based on the class label
    filename = f"{directory}/mnist_image_{i}_label_{label}.png"

    # Save the image
    plt.imsave(filename, image, cmap='gray')
