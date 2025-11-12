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


#----------------------------------------------------------------- #
# -----------------    préparation des données  ------------------ #
#----------------------------------------------------------------- #
base_dir = './PetImagesGray/'
img_width, img_height = 200,200
Batch_Size = 64

train_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(img_height,img_width),
                                                  subset='training',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= Batch_Size)
test_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(img_height,img_width),
                                                  subset='validation',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= Batch_Size)
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #

#----------------------------------------------------------------- #
# ------------------------    le modèle  ------------------------- #
#----------------------------------------------------------------- #
model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width,3)),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),


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
history = model.fit(train_datagen, epochs=Nb_epochs,validation_data=test_datagen)

#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----    Affichage des métriques d'apprentissage  -------------- #
#----------------------------------------------------------------- #
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #


#----------------------------------------------------------------- #
# -----------------    Tester quelques exemples  ----------------- #
#----------------------------------------------------------------- #
from skimage import transform
def predict_image(image_path):
    img = mpimg.imread(image_path)
    img = img[0:img_height,0:img_width]
    plt.imshow(img)
    img = img_to_array(img)
    img = transform.resize(img, (img_height, img_width, 3))
    img = np.expand_dims(img, axis=0)
    result = model.predict(img,batch_size=1)
    print(result)
    print("Dog" if result >= 0.5 else "Cat")
    
predict_image(base_dir + 'Cat/20.jpg')
predict_image(base_dir +'Dog/20.jpg')
#----------------------------------------------------------------- #
#----------------------------------------------------------------- #
