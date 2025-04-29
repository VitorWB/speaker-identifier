import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalizar os dados (0 a 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encoding nos rótulos
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 4. Criar o modelo ANN
model = Sequential([
    Flatten(input_shape=(28, 28)),   
    Dense(128, activation='relu'),   
    Dense(64, activation='relu'),    
    Dense(10, activation='softmax')  
])

# 5. Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinar o modelo
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)

# 7. Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Acurácia no teste: {test_acc:.2f}")

# 8. Fazer previsões
predictions = model.predict(x_test)

# 9. Visualizar 10 exemplos com imagem, rótulo real e previsão
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Real: {y_test[i]}\nPrevisto: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()
