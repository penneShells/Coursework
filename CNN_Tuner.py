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
from keras_tuner import RandomSearch
import numpy as np

#This program is a variation of CNN_Train designed to optimize the hyperparameters used in model training, then train
#a model subject to those optimized hyperparameters
#Theoretically this code works. I have not tested it. It will take hours to run at least.

#It uses the keras tuner to create varying models with a range of hyper parameter values an assess which one
#achieves the highest validation score.

def train(values, suffix):
    
    seed = random.randint(0, 200)
    dataset = values["-DATADROP-"]
    batchsize = values["-BATCHDROP-"]
    imagesize = values["-SIZEDROP-"]
    epochs = values["-EPOCHDROP-"]
    dropout = values["-DROPOUTDROP-"]

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    environ["OMP_NUM_THREADS"] = "4"


    #gets directory of faces
    dataDir = pathlib.Path(f"datasets/{dataset}").with_suffix('')
    imageCount = len(list(dataDir.glob('*/*.jpg')))


    #selects a portion of the faces for training data
    trainDs = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(imagesize, imagesize),
        batch_size=batchsize
    )


    #selects a portion of the faces for validation data
    valDs = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(imagesize, imagesize),
        batch_size=int(batchsize)
    )


    #displays names of people's faces being trained on
    classNames = trainDs.class_names
    print(classNames)

    plt.figure(figsize=(10, 10))
    for images, labels in valDs.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classNames[labels[i]])
            plt.axis("off")

    plt.show()

    AUTOTUNE = tf.data.AUTOTUNE

    trainDs = trainDs.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE)

    trainList = list(trainDs)
    valList = list(valDs)

    trainFeatures = np.concatenate([trainList[n][0] for n in range(0, len(trainList))])
    trainLabels = np.concatenate([trainList[n][1] for n in range(0, len(trainList))])

    valFeatures = np.concatenate([trainList[n][0] for n in range(0, len(trainList))])
    valLabels = np.concatenate([trainList[n][1] for n in range(0, len(trainList))])

    numClasses = len(classNames)

    #redundant
    dataAugmentation = keras.Sequential(
      [
          layers.RandomRotation(0.2, input_shape=(imagesize, imagesize, 3)),
          layers.RandomZoom(0.1),

      ]
    )


    def buildModel(hp):
        #model structure
        model = Sequential([
            #Augment data, rotate/zoom and whatnot
            dataAugmentation,
            #Rescale pixel BGR values to between 1 and 255
            layers.Rescaling(1./255),
            #Convolute and maxpool a few times
            layers.Conv2D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values=[2, 5]),
                padding="same",
                activation='relu'
            ),
            layers.MaxPooling2D(),

            layers.Conv2D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values=[2, 5]),
                padding="same",
                activation='relu'
            ),
            layers.MaxPooling2D(),

            layers.Dropout(dropout),

            #Flatten to 1D array
            layers.Flatten(),
            #Multi-layer Perceptron
            keras.layers.Dense(
                units=hp.Int('dense_1_units', min_value=64, max_value=192, step=16),
                activation='relu'
            ),
            layers.Dense(numClasses, name="outputs")
        ])

        model.compile(optimizer="adam",
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"]
        )

        model.summary()

        return model

    tuner = RandomSearch(buildModel,
                         objective='val_accuracy',
                         max_trials=5)

    tuner.search(trainFeatures, trainLabels, epochs=3, validation_data=(valFeatures, valLabels))


if __name__ == "__main__":
    values = {'-DATADROP-': '256_cv2', '-BATCHDROP-': 128, '-SIZEDROP-': 256, '-EPOCHDROP-': 5, '-DROPOUTDROP-': 0.2}
    train(values, "test")
