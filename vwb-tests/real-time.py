import os
import cv2
import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import librosa.display

# ======== CONFIGURAÇÕES ========
model_path = 'modelo_identificador_vitor.h5'
temp_audio = 'temp.wav'
temp_img = 'temp.png'
sample_rate = 16000
duration = 3  # segundos de gravação
img_height = 128
img_width = 128
# ===============================

# Gravar áudio do microfone
print("🎙️ Gravando... fale agora.")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
wav.write(temp_audio, sample_rate, recording)
print("✅ Gravação finalizada.")

# Extrair MFCC e salvar imagem
y, sr = librosa.load(temp_audio, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(2, 2))
librosa.display.specshow(mfcc, x_axis=None, sr=sr)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(temp_img, bbox_inches='tight', pad_inches=0)
plt.close()

# Carregar imagem MFCC e preparar para entrada do modelo
img = cv2.imread(temp_img, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img_width, img_height))
img = img.reshape((1, img_height, img_width, 1)).astype("float32") / 255.0

# Carregar modelo e prever
model = load_model(model_path)
pred = model.predict(img)[0][0]

if pred >= 0.5:
    print(f"✅ É o Vitor (confiança: {pred:.2f})")
else:
    print(f"❌ Não é o Vitor (confiança: {1 - pred:.2f})")
