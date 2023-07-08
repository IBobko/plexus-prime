from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Загружаем MNIST датасет
(_, _), (x_test, y_test) = mnist.load_data()

# Перебираем изображения в тестовом наборе данных
for i in range(len(x_test)):
    image = x_test[i]
    label = y_test[i]

    # Создаем имя файла, основываясь на метке класса
    filename = f"mnist_img/mnist_image_{i}_label_{label}.png"

    # Сохраняем изображение
    plt.imsave(filename, image, cmap='gray')