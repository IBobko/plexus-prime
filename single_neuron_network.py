import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Создаем модель с одним нейроном
model_single_neuron = Sequential()
model_single_neuron.add(Dense(1, input_dim=1, activation='linear', kernel_initializer='ones'))

# Входные данные
X_single = np.array([[1], [2], [3], [4], [5]])

# Целевые значения (удвоенные входные значения)
y_single = X_single * 2

# Обучение модели с отслеживанием функции потерь и обновлений параметров
losses = []
weights = []

model_single_neuron.compile(loss='mean_squared_error', optimizer='adam')
for epoch in range(500):
    # Один проход обучения
    history = model_single_neuron.fit(X_single, y_single, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    weights.append(model_single_neuron.get_weights())
    losses.append(loss)
    print(f"Epoch {epoch+1}/{500} - Loss: {loss}")

# Проверка модели
test_X_single = np.array([[10], [20], [30]])
predictions_single = model_single_neuron.predict(test_X_single)

print(predictions_single)
