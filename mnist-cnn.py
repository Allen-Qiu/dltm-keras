# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:38:13 2020

recognize digits in mnist using CNN

@author: Allen
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape
from tensorflow.keras import Sequential
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
n,m = x_train.shape[1], x_train.shape[2]

model = Sequential()
model.add(Reshape((n,m,1), input_shape=(n,m)))
model.add(Conv2D(filters=32, kernel_size=5, activation='relu', padding="same"))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding="same"))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='Nadam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=100)
model.evaluate(x_test,  y_test, verbose=2)

for layer in model.layers:
    print(('%s - %s')%(layer.input_shape, layer.output_shape))
