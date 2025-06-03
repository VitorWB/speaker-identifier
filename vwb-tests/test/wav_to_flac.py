import os
import subprocess

# Pasta com os arquivos .wav
entrada_dir = r'/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/test'

# Pasta onde os arquivos .flac serão salvos (pode ser a mesma)
saida_dir = r'/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/test'

# Cria a pasta de saída se necessário
os.makedirs(saida_dir, exist_ok=True)

# Lista arquivos .wav
arquivos_wav = [f for f in os.listdir(entrada_dir) if f.endswith('.wav')]

for arquivo in arquivos_wav:
    caminho_entrada = os.path.join(entrada_dir, arquivo)
    nome_base = os.path.splitext(arquivo)[0]
    caminho_saida = os.path.join(saida_dir, nome_base + '.flac')

    try:
        comando = ['ffmpeg', '-y', '-i', caminho_entrada, caminho_saida]
        subprocess.run(comando, check=True)
        print(f"Convertido: {arquivo} → {nome_base}.flac")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter {arquivo}: {e}")
