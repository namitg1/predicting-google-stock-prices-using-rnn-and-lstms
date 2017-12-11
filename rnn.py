# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing training set
dataset_training=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_training.iloc[:,1:2].values
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
#reshape
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#building rnn
#importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
#initialising rnn
regressor=Sequential()
#first layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#second layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#third layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#fourth layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#adding the output layer
regressor.add(Dense(units=1))
#compile
regressor.compile(optimizer='adam',loss='mean_squared_error')
#fitting the training data
regressor.fit(X_train,y_train,epochs=100,batch_size=32)
#whats our test data
dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2].values
#Getting the Predicted Stock Price
dataset_total=pd.concat((dataset_training['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_training)-60:].values
inputs=inputs.reshape(-1,1)#becoz we did not use iloc method
inputs=sc.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price= regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
#making visualizations
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
