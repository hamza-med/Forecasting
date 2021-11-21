#importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Part1-Data preprocessing
#importing the training set
training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values#extract the column of the open

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()#create object of minMaxClass
training_set = sc.fit_transform(training_set)

#Getting the inputs and the outputs for a time step equals to 1
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#Reshape [samples,timestep,feature]
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

#Part2 - Building the RNN

#import the keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#initialising the RNN
regressor = Sequential()

#adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4 , activation = "sigmoid", input_shape = (1,1)))
#adding the output layer
regressor.add(Dense(units = 1))
#compiling the RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")
#fitting the RNN to the training set
regressor.fit(X_train,y_train,batch_size = 32 , epochs = 200)

# Part3 - Making the predictions and vusualisations
#Getting the real stock price
test_set = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:,1:2].values

#Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = inputs.reshape((20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualisations
plt.plot(real_stock_price, color ="red" , label ="Actual")
plt.plot(predicted_stock_price, color ="green" , label ="Predicted")
plt.title("google_stock_price_prediction")
plt.xlabel("Time")
plt.ylabel("Google stock price")
plt.legend()
plt.show()

# Part4 - Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
