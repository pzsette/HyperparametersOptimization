import os
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow_core.python import set_random_seed

os.environ['PYTHONHASHSEED'] = str(19)
np.random.seed(19)
rn.seed(19)

from keras import optimizers, Sequential
from keras.layers import Dense

set_random_seed(19)


class Model2:

    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.decay = decay
        self.testX = None
        self.testY = None
        self.model = self.create_model2()

    def create_model2(self):

        dataset = pd.read_csv('diabetes.csv')

        # creating input features and target variables
        x = dataset.iloc[:, 0:8]
        y = dataset.iloc[:, 8]

        # standardizing the input feature
        sc = StandardScaler()
        x = sc.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        self.testX = x_test
        self.testY = y_test

        classifier = Sequential()
        classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
        classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
        classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        opt = optimizers.SGD(lr=self.learning_rate, nesterov=True, decay=self.decay)

        classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        classifier.fit(x_train, y_train, batch_size=10, epochs=100, verbose=0)

        return classifier

    def evaluate_model(self):

        eval_model = self.model.evaluate(self.testX, self.testY, verbose=0)
        return eval_model
