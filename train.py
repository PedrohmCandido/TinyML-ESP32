import tensorflow as tf
from tensorflow import keras

# Semente fixa: reduz a variacao entre execucoes de ~8 p.p. para ~1,4 p.p.
# (a variacao residual vem de operacoes nao deterministicas da GPU).
tf.keras.utils.set_random_seed(42)

BASE = "data/processed/rps-real"

# ---------------------------------------------------------------------
# Carregamento das TRES particoes
#
# Diferente da versao anterior (dataset Moroney), agora existe uma
# particao de validacao propria. Isso corrige o vazamento de selecao
# que havia antes, quando o conjunto de teste era usado tanto como
# validation_data quanto como base do early stopping.
#
#   train -> ajusta os parametros
#   val   -> early stopping e decisoes de projeto
#   test  -> avaliacao final, tocado apenas pelo avaliar_modelo.py
# ---------------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    f"{BASE}/train",
    image_size=(96, 96),
    color_mode="grayscale"
)

val_ds = keras.utils.image_dataset_from_directory(
    f"{BASE}/val",
    image_size=(96, 96),
    color_mode="grayscale"
)

test_ds = keras.utils.image_dataset_from_directory(
    f"{BASE}/test",
    image_size=(96, 96),
    color_mode="grayscale"
)

# Conferencia obrigatoria: esta ordem define os indices 0, 1, 2 e precisa
# corresponder ao vetor CLASSES[] do firmware.
print("Classes (treino):", train_ds.class_names)
print("Classes (val)   :", val_ds.class_names)
print("Classes (teste) :", test_ds.class_names)

# Normalizacao: pixels de 0-255 para 0,0-1,0.
# Aplicada nas TRES particoes. O firmware replica esta operacao com
# 'fb->buf[i] / 255.0f' -- as duas precisam sempre coincidir.
normalizar = keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalizar(x), y))
val_ds = val_ds.map(lambda x, y: (normalizar(x), y))
test_ds = test_ds.map(lambda x, y: (normalizar(x), y))

# ---------------------------------------------------------------------
# Data augmentation
#
# As tres camadas fotometricas (contraste, brilho, ruido) simulam a
# variacao que o sensor OV2640 introduz e que o dataset nao possui.
# Elas nao alteram a geometria, portanto nao correm o risco de recortar
# dedos e descaracterizar o gesto.
#
# As geometricas ficam em nivel moderado: com valores agressivos
# (rotacao 0.15 + zoom 0.2 + translacao), observou-se queda de 33 p.p.
# na classe 'paper', por corte dos dedos.
#
# fill_mode="nearest" estende o pixel da borda ao preencher areas
# vazias apos rotacao/zoom. Escolhido porque o fundo deste dataset eh
# uniforme: repetir a borda reproduz o fundo, enquanto o padrao
# "reflect" espelharia a imagem e criaria dedos fantasma nas quinas.
# ---------------------------------------------------------------------
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1, fill_mode="nearest"),
    keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    keras.layers.RandomContrast(0.4),
    keras.layers.RandomBrightness(0.3, value_range=(0.0, 1.0)),
    keras.layers.GaussianNoise(0.03),
])

# Augmentation SOMENTE no treino. Aplicar em val ou test mediria o
# desempenho sobre imagens artificialmente distorcidas.
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# ---------------------------------------------------------------------
# Arquitetura
#
# Dropout nao adiciona parametros (o total segue 996.291) e desaparece
# na conversao para TFLite -- o firmware e os 7 operadores do resolver
# permanecem inalterados.
# ---------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,   # <- particao propria, nao mais o teste
    callbacks=[early_stopping]
)

model.save("models/final/modelo_gestos.h5")

# Avaliacao rapida no teste, apenas para registro no log do treino.
# A analise completa (matriz de confusao, comparacao com o .tflite)
# fica no avaliar_modelo.py.
perda_teste, acc_teste = model.evaluate(test_ds, verbose=0)
print(f"\nConjunto de teste -- perda: {perda_teste:.4f}  "
      f"acuracia: {acc_teste:.4f} ({acc_teste * 100:.2f}%)")
