import os

# Caminho base
base_dir = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/imagens mfcc'
nome_vitor = 'vwb-flac'

# Contadores
total_vitor = 0
total_outros = 0

# Loop pelas subpastas
for nome_pasta in os.listdir(base_dir):
    pasta_completa = os.path.join(base_dir, nome_pasta)
    if not os.path.isdir(pasta_completa):
        continue

    imagens = [f for f in os.listdir(pasta_completa) if f.endswith('.png')]
    count = len(imagens)

    if nome_pasta == nome_vitor:
        total_vitor += count
    else:
        total_outros += count

# Resultado final
print(f"mfcc outros: {total_outros}")
print(f"mfcc vitor: {total_vitor}")
