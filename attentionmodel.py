from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
from keras import regularizers
import keras.backend as K
import numpy as np
from faker import Faker
import random
import csv
import sys  
import tensorflow as tf
from nltk import sent_tokenize, word_tokenize
import os
from gensim.models import Word2Vec, KeyedVectors
import os
import h5py
import pydot 
import graphviz
vocab_list = np.load("/Users/parthpatel/Desktop/vocabs2.npy")
vocab = sorted(set(vocab_list))
vocab_size = len(vocab)
word_to_id = dict.fromkeys(vocab,0)
id_to_word = dict.fromkeys(np.arange(vocab_size),"")
x = 0;
for i in vocab:
	word_to_id[i] = x
	id_to_word[x] = i
	x=x+1
word_to_id['<unk>'] = vocab_size
id_to_word[vocab_size] = '<unk>'


x = np.load("/Users/parthpatel/Desktop/en6.npy")
y = np.load("/Users/parthpatel/Desktop/de6.npy")
y_target = np.load("/Users/parthpatel/Desktop/tar6.npy")
# Defined shared layers as global variables
repeator = RepeatVector(20)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation('softmax', name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas,a])
    return context

n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True, recurrent_dropout=0.5)
output_layer = Dense(vocab_size+1, activation='softmax')

# GRADED FUNCTION: model

def model(Tx, Ty):
    X = Input(shape=(20,))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    #outputs = np.zeros((10,vocab_size+1))
    outputs = []
    embed = Embedding(input_dim=vocab_size+1,output_dim=32,mask_zero=True,input_length=20)(X)
    a = Bidirectional(LSTM(n_a,return_sequences=True))(embed)
    for t in range(10):
        context = one_step_attention(a,s)
        s, _, c = post_activation_LSTM_cell(context,initial_state = [s,c])
        out = output_layer(s)
        outputs.append(out)
        #out = np.asarray(out)
#        sess = tf.Session()
#        with sess.as_default():
#       		outputs.append(out.eval(feed_dict={c0: np.zeros((1,n_s))}))
    print(outputs)
    
    model = Model(inputs=[X,s0,c0], outputs=outputs)  
    return model
model = model(20, 10)
opt = Adam(lr=0.01)
model.compile(opt, loss='categorical_crossentropy')
s1 = np.zeros((len(x), n_s))
c1 = np.zeros((len(x), n_s))
o = list(y_target.swapaxes(0,1))
model.fit([x, s1, c1], o, epochs=30, batch_size=100)
model.save_weights("weights.h5")













