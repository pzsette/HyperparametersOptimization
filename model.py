from tensorflow import keras
import tensorflow as tf
import numpy as np
import random as rn
import os

os.environ['PYTHONHASHSEED'] = str(19)
np.random.seed(19)
tf.random.set_seed(19)
rn.seed(19)


class Model:

    def __init__(self, learning_rate, decay, train_images, train_labels, test_images, test_labels):
        self.learning_rate = learning_rate
        self.decay = decay
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.model = self.create_model()

    def create_model(self):

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        opt = keras.optimizers.SGD(lr=self.learning_rate, nesterov=True, decay=self.decay)
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(self.train_images, self.train_labels, epochs=5, verbose=0)

        return model

    def evaluate_model(self):

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=0)

        return test_loss, test_acc
