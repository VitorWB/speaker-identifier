import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ===================== CONFIGURAÇÕES =====================
base_dir = r'C:\Users\vitor\Documents\TCC\imagens mfcc'
img_height = 128
img_width = 128
modelo_path = 'modelo_identificador_vitor.h5'
# =========================================================

X = []
y = []

# Nome da pasta com os áudios do Vitor
nome_vitor = "vwb-flac"  # ou "84", se for o prefixo de Vitor

# Loop pelas subpastas
for nome_pasta in os.listdir(base_dir):
    pasta_completa = os.path.join(base_dir, nome_pasta)

    if not os.path.isdir(pasta_completa):
        continue

    # Marcar 1 para Vitor, 0 para outros
    rotulo = 1 if nome_pasta == nome_vitor else 0

    for nome_arquivo in os.listdir(pasta_completa):
        if nome_arquivo.endswith(".png"):
            caminho_imagem = os.path.join(pasta_completa, nome_arquivo)
            img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            img = img.reshape((img_height, img_width, 1))

            X.append(img)
            y.append(rotulo)

# Conversão para numpy arrays
X = np.array(X).astype("float32") / 255.0
y = np.array(y).astype("float32")  # binário

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== MODELO =====================
if os.path.exists(modelo_path):
    print("🔁 Carregando modelo existente...")
    model = load_model(modelo_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    print("🆕 Criando novo modelo...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Saída binária
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===================== TREINAMENTO =====================
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stop])

# ===================== AVALIAÇÃO =====================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Acurácia no teste: {test_acc:.2f}")

# ===================== SALVAR MODELO =====================
model.save(modelo_path)
print(f"💾 Modelo salvo em: {modelo_path}")

# ===================== VISUALIZAÇÃO DE PREVISÕES =====================
predictions = model.predict(X_test)

for i in range(10):
    plt.figure(figsize=(3, 3))
    plt.imshow(X_test[i].reshape(img_height, img_width), cmap='gray')
    real = int(y_test[i])
    pred = int(predictions[i] >= 0.5)  # binariza saída
    plt.title(f'Real: {real} | Previsto: {pred}')
    plt.axis('off')
    plt.show()
