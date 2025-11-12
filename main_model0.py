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


#----------------------------------------------------------------- #
# -----------------    préparation des données  ------------------ #
#----------------------------------------------------------------- #
base_dir = './PetImagesGray/PetImagesGray/'
img_width, img_height = 200, 200

# 🔁 Paramètres à tester
Batch_Size = 32          # Exemple : essaye aussi 16, 64, 128
nb_filters = 32          # Exemple : essaye 16, 32, 64
taille_filtre = (5, 5)   # Exemple : essaye (3,3), (5,5)
nb_neurones_dense = 64   # Exemple : essaye 32, 64, 128, 256

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
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# ------------------------    le modèle  ------------------------- #
#----------------------------------------------------------------- #
model = tf.keras.models.Sequential([
    # On fait varier nb_filters et taille_filtre
    layers.Conv2D(nb_filters, taille_filtre, activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    # On fait varier nb_neurones_dense
    layers.Dense(nb_neurones_dense, activation='relu'),

    layers.Dropout(0.2),
    # Couche de sortie binaire
    layers.Dense(1, activation='sigmoid')
])

model.summary()
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -------------------    compilation du modèle  ------------------ #
#----------------------------------------------------------------- #
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #

#----------------------------------------------------------------- #
# -------------------    Entrainement du modèle  ----------------- #
#----------------------------------------------------------------- #
Nb_epochs = 10
history = model.fit(train_datagen, epochs=Nb_epochs, validation_data=test_datagen)
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----    Affichage des métriques d'apprentissage  -------------- #
#----------------------------------------------------------------- #
import os

# Créer un dossier pour enregistrer les graphiques si nécessaire
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# Convertir l'historique en DataFrame
history_df = pd.DataFrame(history.history)

# --- Courbe de perte ---
plt.figure()
history_df.loc[:, ['loss', 'val_loss']].plot(title="Courbe de perte")
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'courbe_perte.png'))  # sauvegarde
plt.close()  # ferme la figure pour éviter chevauchement

# --- Courbe de précision ---
plt.figure()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Courbe de précision")
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'courbe_precision.png'))  # sauvegarde
plt.close()

#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----------------    Tester quelques exemples  ----------------- #
#----------------------------------------------------------------- #
def predict_image(image_path, output_dir='./results'):
    # Créer le dossier si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger et afficher l'image
    img = mpimg.imread(image_path)
    img = img[0:img_height, 0:img_width]
    plt.imshow(img)
    plt.axis("off")
    
    # Faire la prédiction
    img_array = img_to_array(img)
    img_array = transform.resize(img_array, (img_height, img_width, 3))
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array, batch_size=1)
    
    label = "Dog" if result >= 0.5 else "Cat"
    print(result)
    print(label)
    
    # Ajouter le label au titre
    plt.title(f"Prédiction: {label}")
    
    # Enregistrer l'image
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"pred_{filename}")
    plt.savefig(save_path)
    plt.close()
    
predict_image(base_dir + 'Cat/20.jpg')
predict_image(base_dir + 'Dog/20.jpg')


#----------------------------------------------------------------- #
#----------------------------------------------------------------- #
