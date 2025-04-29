# Importação das bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.datasets import mnist  # Dataset com imagens de dígitos manuscritos
from tensorflow.keras.models import Sequential  # Modelo sequencial (camadas em sequência)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Camadas da CNN
from tensorflow.keras.utils import to_categorical  # Para converter os rótulos em one-hot
import matplotlib.pyplot as plt  # Para plotar imagens
import numpy as np  # Para manipulação numérica

# 1. Carregar o dataset MNIST (imagens 28x28 de dígitos 0 a 9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Pré-processamento
# Adiciona o canal (1 para escala de cinza) e normaliza os valores de pixel para o intervalo [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Converte os rótulos para o formato one-hot (ex: 3 → [0,0,0,1,0,0,0,0,0,0])
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 3. Criar o modelo CNN (camadas em sequência)
model = Sequential([
    # Primeira camada convolucional: 32 filtros 3x3 + ReLU
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Camada de pooling: reduz a imagem pela metade (28x28 → 14x14)
    MaxPooling2D(pool_size=(2, 2)),

    # Segunda camada convolucional: 64 filtros 3x3 + ReLU
    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # Outro pooling (14x14 → 7x7)
    MaxPooling2D(pool_size=(2, 2)),

    # Achata a imagem 3D em vetor 1D para alimentar a camada densa
    Flatten(),

    # Camada densa com 128 neurônios e ativação ReLU
    Dense(128, activation='relu'),

    # Camada de saída com 10 neurônios (um para cada dígito) e ativação softmax
    Dense(10, activation='softmax')
])

# 4. Compilar o modelo
# - Otimizador: Adam (adaptativo)
# - Função de perda: cross-entropy categórica (para classificação com one-hot)
# - Métrica: acurácia
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Treinar o modelo
# - batch_size: 64 amostras por vez
# - epochs: repetir o treinamento 5 vezes sobre todos os dados
# - validation_split: 10% dos dados de treino serão usados para validação
model.fit(x_train, y_train_cat, batch_size=64, epochs=5, validation_split=0.1)

# 6. Avaliar o modelo com os dados de teste
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Acurácia no teste: {test_acc:.2f}")

# 7. Fazer previsões com os dados de teste
# Cada previsão é um vetor com 10 probabilidades (uma para cada classe)
predictions = model.predict(x_test)

# 8. Visualizar 10 imagens com seus rótulos reais e previstos
plt.figure(figsize=(12, 5))  # Tamanho da figura

for i in range(10):  # Mostrar as 10 primeiras imagens de teste
    plt.subplot(2, 5, i + 1)  # Grade 2x5
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  # Mostra a imagem em tons de cinza
    plt.title(f"Real: {y_test[i]}\nPrevisto: {np.argmax(predictions[i])}")  # Título com real e previsto
    plt.axis('off')  # Remove os eixos para visualização mais limpa

plt.tight_layout()  # Ajusta os espaçamentos automaticamente
plt.show()  # Exibe as imagens
