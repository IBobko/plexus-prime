from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# Создаем модель
model = Sequential()

# Добавляем входной слой размером 4 и скрытый слой размером 8
model.add(Dense(8, input_dim=4, activation='relu'))

# Добавляем выходной слой размером 1
model.add(Dense(1, activation='linear'))

# Входные данные
X = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

# Целевые значения
y = [1, 2, 3, 4]

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=100)

# Входные данные
test_X = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

# Предсказания модели
predictions = model.predict(test_X)

print(predictions)
