import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------------------- Configuration ----------------------
image_size = (64, 64)
batch_size = 32
epochs = 10
train_dir = "processed_dataset"  # Folder with preprocessed A-Z subfolders
model_save_path = "model/isl_model.h5"

# ---------------------- Data Preparation ----------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ---------------------- Model Definition ----------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

# ---------------------- Compile Model ----------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Save best model
checkpoint = ModelCheckpoint(
    model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
)

# ---------------------- Train ----------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Save final model (in case you skip checkpoint)
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")
