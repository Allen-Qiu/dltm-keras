# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:19:04 2020
sentence embedding with self attention for author profile
@author: Allen
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dot, Flatten
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import json

#hyparameters
da=30
r=50
embed_size=50
hidden_size=200
vocab_size=5000
epoch=2
time_steps=1000

#build dataset
text=[]
gender=[]
age=[]

with open('author-profile.json') as f:
    for line in f.readlines():
        dic=json.loads(line)
        text.append(dic['conversation'])
        gender.append(dic['gender'])
        age.append(dic['age_group'])


t=Tokenizer(num_words=vocab_size,oov_token=None)
t.fit_on_texts(text)
encoded_docs=t.texts_to_sequences(text)
x = pad_sequences(encoded_docs, maxlen=time_steps, padding='post')
dic_age={item:id for id,item in enumerate(set(age))}
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
inputs = Input(shape=(time_steps,))
embed = Embedding(vocab_size, embed_size)
embed_input = embed(inputs)
H =  Bidirectional(LSTM(hidden_size, return_sequences=True), name='H')(embed_input)
ws=Dense(da,activation='tanh',use_bias=False, name='ws')(H)
A=Dense(r, 
        activation='softmax',
        use_bias=False, name='A')(ws)
M=Dot(axes=1, name='M')([H,A])

flat=Flatten()(M)
outputs=Dense(len(set(age)),activation='softmax')(flat)
model=Model(inputs=inputs,outputs=outputs)

penal=tf.matmul(tf.transpose(A,[0,2,1]),A)
penal=penal-K.eye(penal.shape[-1])
model.add_loss(tf.norm(penal))
model.summary()

loss=SparseCategoricalCrossentropy()
model.compile(loss=loss, 
              optimizer='adam', 
              metrics=['categorical_accuracy'])
    
model.fit(x_train, y_train, epochs=epoch, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(x_dev, y_dev, verbose=0)
print('Accuracy: %f' % (accuracy))



