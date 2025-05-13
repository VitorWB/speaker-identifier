import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ===================== CONFIGURAÇÕES =====================
# Caminho base onde estão os áudios
pasta_base = r'C:\Users\vitor\Documents\Projects\speaker-identifier\LibriSpeech\dev-clean'

# Caminho base para salvar as imagens
saida_base = r'C:\Users\vitor\Documents\Projects\speaker-identifier\imagens mfcc'

# Número de coeficientes MFCC
n_mfcc = 13
# =========================================================

# Loop recursivo por todas as subpastas e arquivos .flac
for root, dirs, files in os.walk(pasta_base):
    arquivos_flac = sorted([f for f in files if f.endswith('.flac')])

    if not arquivos_flac:
        continue  # pula se não houver arquivos .flac

    # Determina o caminho relativo da subpasta para organizar as saídas
    caminho_relativo = os.path.relpath(root, pasta_base)
    prefixo_arquivo = '-'.join(caminho_relativo.split(os.sep))

    # Caminho de saída
    saida_dir = os.path.join(saida_base, prefixo_arquivo)
    os.makedirs(saida_dir, exist_ok=True)

    print(f"Processando {len(arquivos_flac)} arquivos em {root}")

    for nome_base in arquivos_flac:
        caminho_audio = os.path.join(root, nome_base)

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
