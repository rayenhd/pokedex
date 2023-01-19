
"""""""""

 -------------------------------------------------------------------------------------------------------------------
|                                                                                                                   |
|               Voici mon IA 'pokédex' capble de reconnître certains pokemons de la première génération             |
|                                                                                                                   |
 -------------------------------------------------------------------------------------------------------------------


"""""""""



# -------------------------------------- Import Modules --------------------------------------- #


import pathlib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import PIL.Image
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # debug tensorflow


# ---------------------------------------- import des données --------------------------------------- #

train_dir = pathlib.Path("pokedex" + r"\train")                    # chemin du dataset d'entrainement
#print(train_dir)


test_dir = pathlib.Path("pokedex" + r"\test")                      # chemin du dataset de test
#print(test_dir)

image_train_count = len(list(train_dir.glob('*/*')))                  # nombre d'images d'entrainement
#print(image_train_count)

image_test_count = len(list(test_dir.glob('*/*')))                    # nombre d'images de test
#print(image_test_count)

# -------------------------------- initialisation des données / preprocessing --------------------------------- #

batch_size = 7                      # taille du batch (images qui seront analysés en même temps)

# initialisation des images

img_height = 200
img_width = 200


# -------------------------------- récupération des données --------------------------------- #

train_data = tf.keras.preprocessing.image_dataset_from_directory(           # récupération des images d'entrainement qui génère un 'tf.data.Dataset'
  train_dir,                                                                # chemin d'accès
  validation_split=0.4,                                                     # répartition des données
  subset="training",                                                        # Subset retourné
  seed=4,                                                                   # index de l'aléatoir afin d'avoir les mêmes valeurs à chaque relance
  image_size=(img_height, img_width),                                       # taille de chaque image
  batch_size=batch_size,                                                    # taille du batch
  )



val_data = tf.keras.preprocessing.image_dataset_from_directory(            # récupération des images de validation (idem)
  test_dir,
  validation_split=0.4,
  subset="validation",
  seed=4,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_data.class_names                                       # list contenant le nom de toutes les classes
print(class_names)

num_classes = len(class_names)                                             # nombre total de class


# ------------------------------------- creation du modèle ----------------------------------------- #


model = tf.keras.Sequential([                                             # creation model Sequentiel
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),         # Normalisation des données
    tf.keras.layers.Conv2D(128, 4, activation='relu'),                    # couche de convolution
    tf.keras.layers.MaxPooling2D(),                                       # couche de pooling
    tf.keras.layers.Conv2D(64, 4, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 4, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 4, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),                                           # Creation de la couche Flatten
    tf.keras.layers.Dense(64, activation='relu'),                        # couche de 64 neuronnes
    tf.keras.layers.Dense(num_classes, activation='softmax')             # couche de sortie avec autant de neuronnes que de class
])


model.compile(                                                           # compilation du modèle
    loss="sparse_categorical_crossentropy",                              # choix de la fonction d'erreur
    optimizer='adam',                                                     # choix de l'optimizer
    metrics=['accuracy'])                                                # choix de la metrique (pourcentage du taux de réussite)

#model.fit(train_data, validation_data= val_data, epochs=20)             # entrainement du modèle sur 20 epochs


# -------------------------------- choix du pokemon et prédiction de la machine --------------------------------- #


file_path = input("select picture")
file_to_predict = pathlib.Path(file_path)                               # chemin de l'image à prédire
image_to_predict = PIL.Image.open(file_to_predict)                      # ouverture de l'image
image_to_predict = np.array(image_to_predict)                           # conversion en tableau numpy
img_to_predict = np.expand_dims(cv2.resize(image_to_predict,(200,200)), axis=0)

predict_x = model.predict( img_to_predict)                              # matrice affichant la proba des prédictions pour chaque classe
classes_x=np.argmax(predict_x,axis=1).astype(int)                       # argument du max de la matrice

plt.imshow(image_to_predict)                                            # affichage de l'image selectionnée
if np.array(class_names)[classes_x][0][0] == 'A':
    plt.title(f"IT'S AN {np.array(class_names)[classes_x][0]}")         # affichage du nom du pokemon prédit par l'IA
else:
    plt.title(f"IT'S A {np.array(class_names)[classes_x][0]}")
plt.show()





