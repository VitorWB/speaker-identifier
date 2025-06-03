import os
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime

# ========== CONFIGURAÇÕES ==========
saida_base = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/LibriSpeech/dev-clean/vwb-flac'
os.makedirs(saida_base, exist_ok=True)

sample_rate = 16000
duration = 30  # segundos
n_mfcc = 13
# ====================================

# 🎙️ Gravar áudio
print("🎙️ Gravando... fale agora.")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("✅ Gravação finalizada.")

# 🗂️ Gerar nome baseado no horário
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
nome_base = f'vitor_{timestamp}'
caminho_audio = os.path.join(saida_base, nome_base + '.flac')

# 💾 Salvar áudio em FLAC
sf.write(caminho_audio, audio, sample_rate, format='FLAC')
print(f"💾 Áudio salvo em: {caminho_audio}")

# 🎧 Carregar áudio e gerar MFCC
y, sr = librosa.load(caminho_audio, sr=None)
y, _ = librosa.effects.trim(y, top_db=20)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

# 📊 Plotar MFCC com estilo completo
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(label='Amplitude')
plt.title(f'MFCC - {nome_base}')
plt.xlabel('Tempo')
plt.ylabel('Coeficiente MFCC')

# # 💾 Salvar imagem
# caminho_img = os.path.join(saida_base, nome_base + '.png')
# plt.savefig(caminho_img)
# plt.close()
# print(f"🖼️ Imagem MFCC salva em: {caminho_img}")
