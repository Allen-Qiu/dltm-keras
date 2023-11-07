"""
using Adversarial Autoencoder visulizes MNIST dataset
@author: Allen Qiu
"""

#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

encoding_dim = 2
epochs       = 20
lr           = 0.001
batch_size   = 256
lam          = 0.2

#%% building model

lrelu = tf.keras.layers.LeakyReLU(0.2)
input_img = keras.Input(shape=(784,))
n_layer1 = tf.keras.layers.BatchNormalization()(input_img)
encoded1 = layers.Dense(512, activation=lrelu)(n_layer1)
encoded2 = layers.Dense(128, activation=lrelu)(encoded1)
encoded3 = layers.Dense(64, activation=lrelu)(encoded2)
encoded = layers.Dense(encoding_dim)(encoded3)

decoded1 = layers.Dense(64, activation=lrelu)(encoded)
decoded2 = layers.Dense(128, activation=lrelu)(decoded1)
decoded3 = layers.Dense(512, activation=lrelu)(decoded2)
n_layer2 = tf.keras.layers.BatchNormalization()(decoded3)
decoded = layers.Dense(784, name="decoder", activation='sigmoid')(n_layer2)

autoencoder = keras.Model(input_img, decoded)
encoder     = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(encoding_dim,))

decoder = tf.keras.Sequential([encoded_input,
                               autoencoder.layers[-5],
                               autoencoder.layers[-4],
                               autoencoder.layers[-3],
                               autoencoder.layers[-2],
                               autoencoder.layers[-1]])

mse = tf.keras.losses.MeanSquaredError()
op  = tf.keras.optimizers.Adamax(learning_rate=lr)
autoencoder.compile(optimizer=op, loss=mse)

# discriminator
discriminator = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(encoding_dim,)),
    Dense(64, activation='relu',input_shape=(encoding_dim,)),
    Dense(1, activation="sigmoid")
    ]);
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# loss function of discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = lam*real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss

#%% preparing dataset
(x_train, y_train), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

#%% fitting model with train_on_batch
dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

for epoch in np.arange(epochs):
    dataset = dataset.shuffle(buffer_size=len(x_train), seed=0)
    for x in dataset:
        z = np.random.normal(size=(len(x), encoding_dim))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            x_encode = encoder(x, training=True)
            x_decode = decoder(x_encode, training=True)
            f_output = discriminator(x_encode)
            r_output = discriminator(z)

            gen_loss = mse(x, x_decode) + 0.01*generator_loss(f_output)
            dis_loss = discriminator_loss(r_output, f_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, autoencoder.trainable_variables)
        op.apply_gradients(zip(gradients_of_generator, autoencoder.trainable_variables))
        gradients_of_disc = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
        op.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

    v_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
    print(f'epoch: {epoch}, gen_loss: {gen_loss}, gen_val: {v_loss}; dis_loss:{dis_loss}')

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

#%% visulization
encoded = encoder(x_train)
x_encoded = encoder(x_train).numpy()
labels = list(set(y_train))
plt.figure()

for i in np.arange(len(labels)):
    idx = y_train == labels[i]
    plt.scatter(x_encoded[idx,0], x_encoded[idx,1], s=0.05, label=labels[i])
plt.legend()
plt.show()

