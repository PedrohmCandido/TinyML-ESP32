from tensorflow import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/processed/Rock-Paper-Scissors-processado",
    image_size=(96, 96),
    color_mode="grayscale"
)

test_ds = keras.utils.image_dataset_from_directory(
    "data/processed/test-processado",
    image_size=(96, 96),
    color_mode="grayscale"
)

print(train_ds.class_names)
print(test_ds.class_names)

train_ds = train_ds.map(lambda x, y: (keras.layers.Rescaling(1./255)(x), y))
test_ds = test_ds.map(lambda x, y: (keras.layers.Rescaling(1./255)(x), y))

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
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
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    callbacks=[early_stopping]
)

model.save("models/final/modelo_gestos.h5")