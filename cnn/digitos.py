import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Pré-processamento
# Redimensionar para incluir o canal (necessário para a CNN)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encoding das classes (0 a 9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Criar o modelo CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Treinar o modelo
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

# 6. Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc:.2f}")
