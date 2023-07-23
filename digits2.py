import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# Создаем модель
model = Sequential()

# Слой свертки, 2D массив с 32 выходами
model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', input_shape=(9, 9, 1), ))


# Создаем пример изображения (9x9)
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Добавляем размерность к изображению
image = image.reshape(1, 9, 9, 1)

# Проходим изображение через сверточный слой
output = model.predict(image)

# Выводим результат свертки
print(output)