# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:25:29 2020

sentiment classifier of combining lstm cell and peephole cell
and add pretrained embeddings

@author: dell
"""

import numpy as np
from tensorflow.keras.layers import RNN, Input
import data_helpers
from peepholecell import PeepholeLSTMCell
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean
from tensorflow.keras.layers import LSTM,Lambda, Concatenate
from tensorflow.keras.initializers import Constant

#parameter
vocab_size=5000
hidden_size=100
embed_size = 100
epoch=10
pname='D:/data/glove.twitter.27B/glove.twitter.27B.%sd.txt'%(embed_size)

# load text
x_text, y = data_helpers.load_data_and_labels()
y=np.argmax(y, axis=1, out=None)

# encoding
t=Tokenizer(num_words=vocab_size,oov_token=None)
t.fit_on_texts(x_text)
encoded_docs=t.texts_to_sequences(x_text)
doc_length = max([len(x) for x in encoded_docs])
x = pad_sequences(encoded_docs, maxlen=doc_length, padding='post')
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# loading pretrained embeddings
embeddings_index = {}
with open(pname, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embed_size))
for word, i in t.word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Split train/test set
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

# build model
inputs=Input(shape=(doc_length,))
embed=Embedding(vocab_size, embed_size, input_length=doc_length)
preembed = Embedding(input_dim=vocab_size, output_dim=embed_size,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=doc_length,
                            trainable=True)
embed_layer1=embed(inputs)
embed_layer2=preembed(inputs)

# combining two type cells
lstm1=LSTM(hidden_size)(embed_layer1)
cell=PeepholeLSTMCell(hidden_size)
lstm2=RNN(cell)(embed_layer2)
conc_lstm=Concatenate()([lstm1,lstm2])
hidden=Dense(100)(conc_lstm)
dropout=Dropout(0.5)(hidden)
output=Dense(1, activation='sigmoid')(hidden)
model = Model(inputs=inputs, outputs=output)

opt = Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=epoch, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(x_dev, y_dev, verbose=0)
print('Accuracy: %f' % (accuracy))



