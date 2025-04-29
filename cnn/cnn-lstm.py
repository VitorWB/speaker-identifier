import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# ===================== CONFIGURAÇÕES =====================
base_dir = r'C:\Users\vitor\Documents\TCC\LibriSpeech\dev-clean'
max_len = 100        # Número máximo de frames MFCC por amostra (tempo)
n_mfcc = 13          # Número de coeficientes MFCC
# =========================================================

X = []
y = []

def extrair_mfcc_sequencia(audio_path, n_mfcc=13, max_len=100):
    y_audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T  # (n_frames, n_mfcc)

# Percorrer todas as pastas
for pasta in os.listdir(base_dir):
    pasta_completa = os.path.join(base_dir, pasta)
    if not os.path.isdir(pasta_completa):
        continue

    for subpasta in os.listdir(pasta_completa):
        subpasta_completa = os.path.join(pasta_completa, subpasta)

        # Define o rótulo com base no prefixo da pasta
        if pasta.startswith("84"):
            rotulo = 0
        elif pasta.startswith("174"):
            rotulo = 1
        else:
            continue

        for arquivo in os.listdir(subpasta_completa):
            if arquivo.endswith(".flac"):
                caminho_audio = os.path.join(subpasta_completa, arquivo)
                try:
                    mfcc_seq = extrair_mfcc_sequencia(caminho_audio, n_mfcc=n_mfcc, max_len=max_len)
                    X.append(mfcc_seq)
                    y.append(rotulo)
                except Exception as e:
                    print(f"Erro em {arquivo}: {e}")

# Convertendo para arrays numpy
X = np.array(X).astype("float32")  # shape: (n_amostras, n_frames, n_mfcc)
y = to_categorical(np.array(y))    # one-hot

# Divisão em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== MODELO CNN + LSTM =====================
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(max_len, n_mfcc)),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # n_classes = número de locutores
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Acurácia no teste: {test_acc:.2f}")
