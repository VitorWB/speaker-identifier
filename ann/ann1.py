import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Carregar os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalizar os dados (0 a 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encoding nos rótulos
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Criar o modelo ANN
model = Sequential([
    Flatten(input_shape=(28, 28)),   # camada de entrada (achatamento da imagem 28x28)
    Dense(128, activation='relu'),   # camada escondida com 128 neurônios
    Dense(64, activation='relu'),    # outra camada escondida
    Dense(10, activation='softmax')  # camada de saída com 10 neurônios (0 a 9)
])

# 5. Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinar o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 7. Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc:.2f}")
