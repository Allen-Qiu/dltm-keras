# -*- coding: utf-8 -*-
"""
predict stock price
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# building dataset
dataset_train = pd.read_csv("google-stock/trainset.csv")
dataset_test  = pd.read_csv("google-stock/testset.csv")
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
real_stock_price = dataset_test['Open']

sc = MinMaxScaler(feature_range = (0,1))
dataset_total  = dataset_total.values
dataset_total  = dataset_total.reshape(-1,1)
dataset_scaled = sc.fit_transform(dataset_total)

trainset = dataset_scaled[:len(dataset_train)]
testset  = dataset_scaled[-len(dataset_test)-60:]

x_train = []
y_train = []

for i in range(60,1259):
    x_train.append(trainset[i-60:i, 0])
    y_train.append(trainset[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# building model
regressor = Sequential()
regressor.add(GRU(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(GRU(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(GRU(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)

# predict in test set

x_test = []
for i in range(60,185):
    x_test.append(testset[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
predicted_price = regressor.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)


plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



