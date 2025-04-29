import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ===================== CONFIGURAÇÕES =====================
# Caminho base onde estão os áudios
pasta_base = r'C:\Users\vitor\Documents\TCC\LibriSpeech\dev-clean'

subpasta = r'174\168635'

# Caminho completo da pasta de entrada
entrada_dir = os.path.join(pasta_base, subpasta)

prefixo_arquivo = '-'.join(subpasta.split(os.sep))

# Caminho de saída onde serão salvas as imagens
saida_dir = os.path.join(r'C:\Users\vitor\Documents\TCC\imagens mfcc', prefixo_arquivo)

# Número de coeficientes MFCC
n_mfcc = 13
# =========================================================

# Cria pasta de saída, se necessário
os.makedirs(saida_dir, exist_ok=True)

# Lista os arquivos .flac na pasta
arquivos_flac = sorted([f for f in os.listdir(entrada_dir) if f.endswith('.flac')])

print(f"Encontrados {len(arquivos_flac)} arquivos .flac na pasta {entrada_dir}")

# Loop pelos arquivos
for nome_base in arquivos_flac:
    caminho_audio = os.path.join(entrada_dir, nome_base)

    try:
        # Carregar o áudio
        y, sr = librosa.load(caminho_audio, sr=None)

        # Extrair MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Plotar MFCC
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time', sr=sr)
        plt.colorbar(label='Amplitude')
        plt.title(f'MFCC - {nome_base}')
        plt.xlabel('Tempo')
        plt.ylabel('Coeficiente MFCC')

        # Caminho de saída da imagem
        nome_imagem = nome_base.replace('.flac', '.png')
        caminho_saida = os.path.join(saida_dir, nome_imagem)

        # Salvar e fechar
        plt.savefig(caminho_saida)
        plt.close()

        print(f"Imagem salva: {caminho_saida}")

    except Exception as e:
        print(f"Erro ao processar {nome_base}: {e}")
