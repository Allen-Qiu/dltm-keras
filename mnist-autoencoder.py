"""
Created on Tue Sep  6 17:14:46 2022

Building Autoencoder to visulize MNIST dataset

@author: Allen Qiu
"""

#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

encoding_dim = 2
epochs       = 0
lr           = 0.005
batch_size   = 128

#%% building model
input_img = keras.Input(shape=(784,))
n_layer1 = BatchNormalization()(input_img)
encoded1 = Dense(512, activation='relu')(n_layer1)
encoded2 = Dense(128, activation='relu')(encoded1)
encoded3 = Dense(64, activation='relu')(encoded2)
encoded = Dense(encoding_dim)(encoded3)

decoded1 = Dense(64, activation='relu')(encoded)
decoded2 = Dense(128, activation='relu')(decoded1)
decoded3 = Dense(512, activation='relu')(decoded2)
n_layer2 = BatchNormalization()(decoded3)
decoded = Dense(784, name="decoder", activation='sigmoid')(n_layer2)

autoencoder = Model(input_img, decoded)
encoder     = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder = tf.keras.Sequential([encoded_input,
                               autoencoder.layers[-5],
                               autoencoder.layers[-4],
                               autoencoder.layers[-3],
                               autoencoder.layers[-2],
                               autoencoder.layers[-1]])

mse = tf.keras.losses.MeanSquaredError()
op  = tf.keras.optimizers.Adamax(learning_rate=lr)
autoencoder.compile(optimizer=op, loss=mse)

(x_train, y_train), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

#%% draw images
def draw_imgs(n, imgs):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

n = 10
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
draw_imgs(n, x_test)
draw_imgs(n, decoded_imgs)

#%% generator
z = np.random.normal(size=(n, encoding_dim))
decoded_imgs2 = decoder.predict(z, verbose=0)
draw_imgs(n, decoded_imgs2)

#%% visulization
encoded = encoder(x_train)
x_encoded = encoder(x_train).numpy()
labels = list(set(y_train))
plt.figure()

for i in np.arange(len(labels)):
    idx = y_train == labels[i]
    plt.scatter(x_encoded[idx,0], x_encoded[idx,1], s=0.1)
plt.show()
