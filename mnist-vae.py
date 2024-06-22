"""
Building variational Autoencoder to visulize MNIST dataset
@author: Allen Qiu
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

encoding_dim = 2
epochs       = 30
lr           = 0.0007
batch_size   = 128
epsilon_std = 1.0

# building model
input_img = keras.Input(shape=(784,))
n_layer1 = BatchNormalization()(input_img)
encoded1 = Dense(512, activation='relu')(n_layer1)
encoded2 = Dense(128, activation='relu')(encoded1)
h = Dense(64, activation='relu')(encoded2)

# both mu and log_sigma are diagonal matrix
z_mean = Dense(encoding_dim)(h)
z_log_var = Dense(encoding_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, encoding_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(encoding_dim,))([z_mean, z_log_var])
decoded1 = Dense(64, activation='relu')(z)
decoded2 = Dense(128, activation='relu')(decoded1)
decoded3 = Dense(512, activation='relu')(decoded2)
n_layer2 = BatchNormalization()(decoded3)
decoded = Dense(784, name="decoder", activation='sigmoid')(n_layer2)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, z)

encoded_input = Input(shape=(encoding_dim,))

decoder = tf.keras.Sequential([encoded_input,
                               autoencoder.layers[-5],
                               autoencoder.layers[-4],
                               autoencoder.layers[-3],
                               autoencoder.layers[-2],
                               autoencoder.layers[-1]])

kl_loss = 0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var -2 - z_log_var ), axis=-1)
kloss = K.mean(kl_loss)
autoencoder.add_loss(0.0002 * kloss)
mse = tf.keras.losses.MeanSquaredError()
op  = tf.keras.optimizers.Adamax(learning_rate=lr)
autoencoder.compile(optimizer=op, loss=mse)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size, drop_remainder=True)
autoencoder.fit(train_dataset, epochs=epochs, shuffle=True, verbose=1)

# draw images
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
encoded_imgs = encoder.predict(test_dataset)
decoded_imgs = decoder.predict(encoded_imgs)
draw_imgs(n, x_test)
draw_imgs(n, decoded_imgs)

# generator
z = np.random.normal(size=(n, encoding_dim))
decoded_imgs2 = decoder.predict(z, verbose=0)
draw_imgs(n, decoded_imgs2)

# visulization
x_encoded = encoder.predict(train_dataset)
labels = list(set(y_train))
plt.figure()

for i in np.arange(len(labels)):
    idx = y_train[:x_encoded.shape[0]] == labels[i]
    plt.scatter(x_encoded[idx,0], x_encoded[idx,1], s=0.1)
plt.show()
