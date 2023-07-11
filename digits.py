import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загружаем MNIST датасет
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем значения пикселей
x_train, x_test = x_train / 255.0, x_test / 255.0

# Преобразуем целевую переменную в категориальный формат
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Добавляем еще одно измерение к нашим данным
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Создаем модель
model = Sequential()

# Слой свертки, 2D массив с 32 выходами
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Слой подвыборки, выбирает лучший пиксель из (2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой свертки, 2D массив с 64 выходами
model.add(Conv2D(64, (3, 3), activation='relu'))

# Еще один слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())

# Полносвязный слой
model.add(Dense(128, activation='relu'))

# Еще один слой регуляризации Dropout
model.add(Dropout(0.5))

# Выходной слой с функцией активации softmax
model.add(Dense(10, activation='softmax'))

# Компилируем модель
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),  # Используем оптимизатор Adam
              metrics=['accuracy'])

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=10,  # Угол поворота изображений
    width_shift_range=0.1,  # Сдвиг по горизонтали
    height_shift_range=0.1,  # Сдвиг по вертикали
    zoom_range=0.1  # Масштабирование изображений
)

# Обучаем модель с использованием аугментации данных
model.fit(datagen.flow(x_train, y_train, batch_size=128),
          steps_per_epoch=len(x_train) / 128,  # Количество шагов на эпоху
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

model.save('digits.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])
