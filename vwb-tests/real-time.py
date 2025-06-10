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
model_path = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/modelo_identificador_vitor.h5'
temp_audio = 'temp.wav'
temp_img = 'temp.png'
sample_rate = 16000
duration = 10
img_height = 128
img_width = 128
n_mfcc = 13
n_frames_pad = 130
# ===============================

# 🎙️ Gravar áudio do microfone
print("🎙️ Gravando... fale agora.")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
wav.write(temp_audio, sample_rate, recording)
print("✅ Gravação finalizada.")

# 🎧 Carregar áudio e extrair MFCC
y, sr = librosa.load(temp_audio, sr=None)
y, _ = librosa.effects.trim(y, top_db=20)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
mfcc = librosa.util.fix_length(mfcc, size=n_frames_pad, axis=1)

# Salvar imagem MFCC com o mesmo estilo do treinamento
plt.figure(figsize=(10, 4))  # mesmo formato usado no treino
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(label='Amplitude')
plt.title('MFCC - tempo real')
plt.xlabel('Tempo')
plt.ylabel('Coeficiente MFCC')
plt.savefig(temp_img)  # sem recortes ou compressão
plt.close()

# 📷 Carregar imagem gerada e preparar para modelo
img = cv2.imread(temp_img, cv2.IMREAD_GRAYSCALE)

# ⚠️ IMPORTANTE: fazer resize para garantir compatibilidade com o modelo
img = cv2.resize(img, (img_width, img_height))
img = img.reshape((1, img_height, img_width, 1)).astype("float32") / 255.0

# 🤖 Carregar modelo e prever
model = load_model(model_path)
pred = model.predict(img)

conf = float(pred[0][0])
if conf >= 0.5:
    print(f"✅ É o Vitor (confiança: {conf:.2f})")
else:
    print(f"❌ Não é o Vitor (confiança: {1 - conf:.2f})")
