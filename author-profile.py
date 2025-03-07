"""
sentence embedding with self attention for author profile
@author: Allen
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Input, LSTM, Dense, Dot, Flatten, Lambda
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import json

# hyparameters
lr = 0.002
da = 128
embed_size = 64
hidden_size = 256
vocab_size = 5000
epoch = 20
time_steps = 512

# build dataset
text = []
gender = []
age = []

with open('../data/author-profile.json') as f:
    for line in f.readlines():
        dic = json.loads(line)
        text.append(dic['conversation'])
        gender.append(dic['gender'])
        age.append(dic['age_group'])

t = Tokenizer(num_words=vocab_size, oov_token=None)
t.fit_on_texts(text)
encoded_docs = t.texts_to_sequences(text)
x = pad_sequences(encoded_docs, maxlen=time_steps, padding='post')
dic_age = {item: id for id, item in enumerate(set(age))}
y = [dic_age[item] for item in age]
y = np.array(y)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

# building model
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    inputs = Input(shape=(time_steps,))
    embed = Embedding(vocab_size, embed_size)
    embed_input = embed(inputs)
    H = Bidirectional(LSTM(hidden_size, return_sequences=True), name='H')(embed_input)
    ws = Dense(da, activation='tanh', use_bias=False, name='ws')(H)
    A = Dense(2*hidden_size,
              activation='softmax',
              use_bias=False, name='A')(ws)
    M = Dot(axes=2, name='M')([H, A])

    # mid_layer = Lambda(lambda x: K.mean(x, axis=1))(M)
    mid_layer = Flatten()(M)
    fc1 = Dense(1024, activation='relu')(mid_layer)
    fc2 = Dense(128, activation='tanh')(fc1)
    outputs = Dense(len(set(age)), activation='softmax')(fc2)
    model = Model(inputs=inputs, outputs=outputs)

    penal = tf.matmul(tf.transpose(A, [0, 2, 1]), A)
    penal = penal - K.eye(penal.shape[-1])
    model.add_loss(tf.norm(penal))
    # model.summary()

    opt = Adamax(learning_rate=lr)
    loss = SparseCategoricalCrossentropy()
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epoch, validation_split=0.1, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(x_dev, y_dev, verbose=0)
print('Accuracy: %f' % (accuracy))


