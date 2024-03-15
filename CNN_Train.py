import matplotlib.pyplot as plt
import os
from os import environ
import pathlib
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import random
import numpy as np


def train(values, suffix):

    #Declare hyperparameters
    seed = random.randint(0, 200)
    dataset = values["-DATADROP-"]
    batchsize = values["-BATCHDROP-"]
    imagesize = values["-SIZEDROP-"]
    epochs = values["-EPOCHDROP-"]
    dropout = values["-DROPOUTDROP-"]

    #Restrict CPU useage to prevent very bad things
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    environ["OMP_NUM_THREADS"] = "4"


    #Gets directory of faces
    dataDir = pathlib.Path(f"datasets/{dataset}").with_suffix('')
    imageCount = len(list(dataDir.glob('*/*.jpg')))


    #Selects a portion of the faces for training data
    trainDs = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(imagesize, imagesize),
        batch_size=batchsize
    )

    #Selects a portion of the faces for validation data
    valDs = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(imagesize, imagesize),
        batch_size=int(batchsize)
    )

    #Displays names of people's faces being trained on
    classNames = trainDs.class_names
    print(classNames)

    plt.figure(figsize=(10, 10))


    # for images, labels in valDs.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(classNames[labels[i]])
    #         plt.axis("off")
    #
    # plt.show()

    AUTOTUNE = tf.data.AUTOTUNE

    #Randomise the order of the datasets I think
    trainDs = trainDs.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE)


    numClasses = len(classNames)

    #Creates a layer which applies modifications to the images to increase variety
    dataAugmentation = keras.Sequential(
      [
          layers.RandomRotation(0.2, input_shape=(imagesize, imagesize, 3)),
          layers.RandomZoom(0.1),

      ]
    )

    #Declares the structure of the Convolutional Neural Network itself
    model = Sequential([
        #Augment data, rotate/zoom and whatnot
        dataAugmentation,
        #Rescale pixel BGR values to between 1 and 255
        layers.Rescaling(1./255),
        #Convolute and maxpool a few times
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        #Drop out a percentage of the input values to the fully connected layer
        layers.Dropout(dropout),

        #Flatten to 1D array
        layers.Flatten(),
        #Multi-layer Perceptron
        layers.Dense(128, activation='relu'),
        layers.Dense(numClasses, name="outputs")
    ])


    #Compile the CNN
    model.compile(optimizer="adam",
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"]
    )

    model.summary()

    #Train the CNN based on the dataset
    history = model.fit(
        trainDs,
        validation_data=valDs,
        epochs=int(epochs)
    )

    #Save the trained CNN
    model.save(f"keras_models/faceRecognition_{imagesize}_{suffix}.keras")

    #Display a graph of the training progress throughout the epochs
    acc = history.history["accuracy"]
    valAcc = history.history["val_accuracy"]

    loss = history.history["loss"]
    valLoss = history.history["val_loss"]

    epochsRange = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochsRange, acc, label="Training Accuracy")
    plt.plot(epochsRange, valAcc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochsRange, loss, label="Training Loss")
    plt.plot(epochsRange, valLoss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.show()

#Just put this here for testing purposes without the need to run it through the GUI
#Runs the train function with preset hyperparameters
if __name__ == "__main__":
    values = {'-DATADROP-': '256_cv2', '-BATCHDROP-': 128, '-SIZEDROP-': 256, '-EPOCHDROP-': 5, '-DROPOUTDROP-': 0.2}
    train(values, "test")