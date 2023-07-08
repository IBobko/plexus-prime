import numpy as np
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Загружаем сохраненную модель
model = load_model('digits.h5')

# Загружаем или создаем ваш 28x28 массив
# Здесь мы создаем заглушку для примера
digit = np.zeros((28,28))

# Не забудьте нормализовать ваши данные, если это было сделано во время обучения
digit = digit / 255.0

# Измените форму массива, чтобы он соответствовал форме входных данных модели
digit = digit.reshape(1, 28, 28, 1)

# Делаем предсказание
prediction = model.predict(digit)

# Получаем наиболее вероятный класс
predicted_class = np.argmax(prediction)

print("Predicted class is: ", predicted_class)






@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(data)
        digit = np.array(data)
        digit = 1-digit
        print(digit)
        digit = digit.reshape(1, 28, 28, 1)
        # Не забудьте нормализовать ваши данные, если это было сделано во время обучения
        #digit = digit / 255.0

        # Измените форму массива, чтобы он соответствовал форме входных данных модели

        # Делаем предсказание
        prediction = model.predict(digit)

        # Получаем наиболее вероятный класс
        predicted_class = np.argmax(prediction)

        print("Predicted class is: ", predicted_class)


        return jsonify({'prediction': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.', 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
