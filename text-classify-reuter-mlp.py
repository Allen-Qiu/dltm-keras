# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:03:33 2021

@author: Allen
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

from sklearn.preprocessing import LabelEncoder

vocab_size = 5000

dir        = 'D:/qjt/data/reuter R8/'
train_file = dir+'r8-train-no-stop.txt'
test_file  = dir+'r8-test-no-stop.txt'

X_train = list()
y_train = list()
X_test  = list()
y_test  = list()
    
fin = open(train_file, "r")
s= fin.readlines()

for line in s:
    items = line.split('\t')
    y_train.append(items[0])
    X_train.append(items[1])
        
fin.close()

fin = open(test_file, "r")
s= fin.readlines()

for line in s:
    items = line.split('\t')
    y_test.append(items[0])
    X_test.append(items[1])
    
t=Tokenizer(num_words=vocab_size, oov_token='unk', split=' ')
t.fit_on_texts(X_train + X_test)
X_train_coded = t.texts_to_sequences(X_train)
X_test_coded  = t.texts_to_sequences(X_test)

max_len = max([len(line) for line in X_train_coded])
X_train_padded_docs = pad_sequences(X_train_coded, maxlen=max_len, padding='post')
X_test_padded_docs  = pad_sequences(X_test_coded, maxlen=max_len, padding='post')
                            
y = LabelEncoder().fit_transform(y_train+y_test)
idx = len(y_train)
size = len(y)

y_train = y[range(idx)]
y_test = y[range(idx, size)]


model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_padded_docs, y_train, epochs=50, verbose=1)
loss, accuracy = model.evaluate(X_test_padded_docs, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))










