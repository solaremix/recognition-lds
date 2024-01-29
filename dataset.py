import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import keras.datasets as tfds
import pathlib

data_dir = pathlib.Path(
    r"C:\Users\Junior\Desktop\lds-project\recognition-lds\img")

image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

# Definiendo parámetros para el cargador
batch_size = 32
img_height = 180
img_width = 180


# Definiendo el conjunto de datos de entrenamiento
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
