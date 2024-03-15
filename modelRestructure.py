import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
from os import environ
import tensorflow as tf
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

#cant forget the "dont blow up" code
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
environ["OMP_NUM_THREADS"] = "4"

#gets directory of faces
dataDir = pathlib.Path("people").with_suffix('')
imageCount = len(list(dataDir.glob('*/*.jpg')))
print(imageCount)

batchSize = 128
imgHeight = 192
imgWidth = 108

#selects a portion of the faces for training data
trainDs = tf.keras.utils.image_dataset_from_directory(
    dataDir,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize
)

#selects a portion of the faces for validation data
valDs = tf.keras.utils.image_dataset_from_directory(
    dataDir,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize
)

def mapFunction(features, labels):
    return features, labels

classNames = trainDs.class_names

plt.figure(figsize=(10, 10))
for images, labels in trainDs.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classNames[labels[i]])
    plt.axis("off")

plt.show()

print(type(valDs))

sns.countplot(x=mapValDs.numpy());
plt.xlabel('Digits')
plt.title("People Data Distribution");

plt.show()