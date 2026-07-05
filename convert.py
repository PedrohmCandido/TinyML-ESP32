from tensorflow import keras
import tensorflow as tf

saved_model = keras.models.load_model("models/final/modelo_gestos.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("models/final/modelo_gestos.tflite", "wb") as f:
    f.write(tflite_model)