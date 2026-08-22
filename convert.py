from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob, cv2, numpy as np

saved_model = keras.models.load_model("models/final/modelo_gestos.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

caminhos = glob.glob("data/processed/**/*.png", recursive=True)[:200]

def representative_dataset():
    for caminho in caminhos:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        yield [img.reshape(1, 96, 96, 1)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

with open("models/final/modelo_gestos.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversão concluída!")