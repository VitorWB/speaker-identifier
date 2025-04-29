import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

# Função para extrair MFCC de um arquivo de áudio
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=None)  # Carregar áudio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extrair MFCC
    pad_width = max_pad_len - mfcc.shape[1]
    
    if pad_width > 0:
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]  # Cortar para tamanho fixo
    
    return mfcc

# Simulação de base de dados
audio_files = ["voz1.wav", "voz2.wav", "voz3.wav", "voz4.wav"] 
labels = [0, 1, 0, 1]  # 0 = Pessoa 1, 1 = Pessoa 2

# Carregar e processar os dados
X = np.array([extract_mfcc(f) for f in audio_files])
y = np.array(labels)

# Normalizar os MFCCs
X = (X - np.mean(X)) / np.std(X)

# Redimensionar para formato necessário (CNN + LSTM)
X = X[..., np.newaxis]  # Adiciona canal para CNN (Formato: [amostras, altura, largura, canais])

# Transformar rótulos em categorias (One-hot encoding)
num_classes = len(set(labels))
y = to_categorical(y, num_classes=num_classes)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo CNN + LSTM
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Reshape((X.shape[1], -1)),  # Transformar saída da CNN para entrada da LSTM
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Saída com softmax para classificação
])

# Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar modelo
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Avaliar modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {acc * 100:.2f}%")
