# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:12:50 2020

char-level language model

@author: Allen
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# parameters
hidden_size = 1000 # size of hidden layer of neurons
seq_length = 25   # number of steps to unroll the RNN for
learning_rate = 0.001
epoch_size = 20
batch_size = 100

# data
data = open('t1.txt', 'r').read() 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
idx_data=[char_to_ix[item]for item in data]
encoded_data=to_categorical(idx_data)
one_input = encoded_data[0 : data_size-2]
one_target = idx_data[1 : data_size-1]

x_train = []
y_train = []
p = 0
while p + seq_length < len(one_input)-seq_length:
    x_train.append(one_input [p:p+seq_length])
    y_train.append(one_target[p:p+seq_length])
    p = p + seq_length+1

x_train = np.array(x_train)
y_train = np.array(y_train)

# model
model= Sequential()
#model.add(Input(shape=(seq_length,vocab_size), name='input'))
model.add(Dense(hidden_size, input_shape=(seq_length,vocab_size), name='hidden'))
model.add(SimpleRNN(hidden_size, activation='relu', return_sequences=True, name='rnn'))
model.add(Dense(vocab_size, activation='softmax', name='output'))
model.summary()
opt=Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_size)

# check
res=model.predict(x_train)

i=2
x_=''.join([ix_to_char[item] for item in np.argmax(x_train[i],axis=1)])
print(x_)
y_=''.join([ix_to_char[item] for item in np.argmax(res[i],axis=1)])
print(y_)



