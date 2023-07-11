import traceback

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the saved model
model = load_model('digits.h5')


def process_digit(digit):
    digit = digit.reshape(1, 28, 28, 1)
    return np.argmax(model.predict(digit)[0])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        digit = np.array(data)
        prediction = process_digit(digit)
        return jsonify({'prediction': prediction.item()})
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.', 'trace': traceback.format_exc()})


@app.route('/picture', methods=['GET'])
def picture():
    custom_image = tf.keras.preprocessing.image.load_img('./mnist_img/mnist_image_0_label_7.png',
                                                         color_mode='grayscale',
                                                         target_size=(28, 28))
    # Convert the image to numpy array
    custom_image_array = tf.keras.preprocessing.image.img_to_array(custom_image)
    # Convert to 28x28 array
    custom_image_array_28x28 = np.squeeze(custom_image_array)
    return jsonify({'data': custom_image_array_28x28.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
