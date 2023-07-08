import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Загружаем MNIST датасет
(_, _), (x_test, y_test) = mnist.load_data()

# Нормализуем значения пикселей
x_test = x_test / 255.0

# Преобразуем целевую переменную в категориальный формат
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Добавляем еще одно измерение к нашим данным
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Загружаем модель
model = tf.keras.models.load_model('digits.h5')

# Оцениваем модель на тестовых данных MNIST
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
