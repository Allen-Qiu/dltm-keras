# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:22:53 2020
word2vec
@author: Allen
"""
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dot
from tensorflow.keras.models import Model

import collections
import random
import zipfile

import numpy as np
import tensorflow as tf

# Step 1: build dataset
def build_dataset(filename, n_words):
    """Process raw inputs into a dataset."""
    with zipfile.ZipFile(filename) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


filename = 'text8.zip'
vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(
    filename, vocabulary_size)

#step 2: generate trainset
window_size = 1
vector_dim = 300
epochs = 10
batch_size=1000
vocab_size=len(dictionary)

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

couples, labels = skipgrams(data, vocab_size, window_size=window_size, negative_samples=0.1)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

#step 3: build model
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)
dot_product = Dot(axes=1, normalize=False)([target, context])
dot_product = Reshape((1,))(dot_product)

similarity = Dot(axes=1, normalize=False)([target, context])

# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)

# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

#callback
class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim[:,0,0]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def _get_sim(self,valid_word_idx):
        in_arr1 = np.ones((vocab_size,))*valid_word_idx
        in_arr2 = np.arange(vocab_size)
        sim = validation_model.predict_on_batch([in_arr1, in_arr2])
        return sim
sim_cb = SimilarityCallback()

#step 4: training
ndlabels=np.array(labels)
for cnt in range(epochs):
    idx = random.sample(range(vocabulary_size),batch_size)
    loss = model.train_on_batch([word_target[idx],word_context[idx]], ndlabels[idx])
    if cnt % 1000 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))

    sim_cb.run_sim()

        
        