import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

img_dir = r'C:\Users\vitor\Documents\Projects\speaker-identifier\teste'
modelo_path = 'modelo_identificador_vitor.h5'
img_height, img_width = 128, 128

model = load_model(modelo_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Opcional: evita warning

# Percorre todas as subpastas
for root, _, files in os.walk(img_dir):
    for arquivo in files:
        if not arquivo.lower().endswith('.png'):
            continue

        img_path = os.path.join(root, arquivo)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.reshape((1, img_height, img_width, 1)).astype("float32") / 255.0

        pred = model.predict(img)[0][0]

        if pred >= 0.5:
            print(f'{arquivo}: ✅ É o Vitor! (confiança: {pred:.2f})')
        else:
            print(f'{arquivo}: ❌ Não é o Vitor. (confiança: {1 - pred:.2f})')
