#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:59:50 2025
@author: chaari
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing.image import img_to_array
import matplotlib.image as mpimg 
from skimage import transform
import os

#----------------------------------------------------------------- #
# -----------------    Préparation des données  ------------------ #
#----------------------------------------------------------------- #
base_dir = './PetImagesGray/PetImagesGray/'
img_width, img_height = 200, 200

Batch_Size = 32

train_datagen = image_dataset_from_directory(
    base_dir,
    image_size=(img_height, img_width),
    subset='training',
    seed=1,
    validation_split=0.1,
    batch_size=Batch_Size
)

test_datagen = image_dataset_from_directory(
    base_dir,
    image_size=(img_height, img_width),
    subset='validation',
    seed=1,
    validation_split=0.1,
    batch_size=Batch_Size
)

# Normalisation pour [0,1]
normalization_layer = layers.Rescaling(1./255)
train_datagen = train_datagen.map(lambda x, y: (normalization_layer(x), y))
test_datagen = test_datagen.map(lambda x, y: (normalization_layer(x), y))
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# ------------------------    Modèle profond  -------------------- #
#----------------------------------------------------------------- #
model = tf.keras.models.Sequential([
    # 1 couche conv 32 filtres
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),

    # 3 couches conv 64 filtres
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # 3 couches denses 512 neurones + batchnorm + dropout
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Couche de sortie
    layers.Dense(1, activation='sigmoid')
])

model.summary()
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -------------------    Compilation & Entraînement -------------- #
#----------------------------------------------------------------- #
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

Nb_epochs = 5
history = model.fit(train_datagen, epochs=Nb_epochs, validation_data=test_datagen)
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----------   Sauvegarde & affichage des métriques  ------------ #
#----------------------------------------------------------------- #
output_dir = './results_deep'
os.makedirs(output_dir, exist_ok=True)

history_df = pd.DataFrame(history.history)

# Courbe de perte
plt.figure()
history_df.loc[:, ['loss', 'val_loss']].plot(title="Courbe de perte")
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'courbe_perte.png'))
plt.close()

# Courbe de précision
plt.figure()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Courbe de précision")
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'courbe_precision.png'))
plt.close()

# Sauvegarde du modèle
model.save(os.path.join(output_dir, 'model_cats_dogs_deep.h5'))
print("✅ Modèle sauvegardé dans", output_dir)
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----------------    Tester quelques exemples  ----------------- #
#----------------------------------------------------------------- #
def predict_image(image_path, output_dir='./results_deep'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger l'image
    img = mpimg.imread(image_path)
    img = img[0:img_height, 0:img_width]
    plt.imshow(img)
    plt.axis("off")
    
    # Préparer l'image
    img_array = img_to_array(img)
    img_array = transform.resize(img_array, (img_height, img_width, 3))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisation
    
    # Prédiction
    result = model.predict(img_array, batch_size=1)
    label = "Dog " if result >= 0.5 else "Cat "
    print(f"Résultat brut : {result} → {label}")
    
    # Afficher et sauvegarder
    plt.title(f"Prédiction : {label}")
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"pred_{filename}")
    plt.savefig(save_path)
    plt.close()

predict_image(base_dir + 'Cat/20.jpg')
predict_image(base_dir + 'Dog/20.jpg')
predict_image(base_dir + 'Cat/30.jpg')
predict_image(base_dir + 'Dog/30.jpg')
#----------------------------------------------------------------- #
