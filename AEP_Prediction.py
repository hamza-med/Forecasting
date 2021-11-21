import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("AEP_hourly.csv")

dataset = df
dataset["Month"] = pd.to_datetime(df["Datetime"]).dt.month
dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
dataset["Week"] = pd.to_datetime(df["Datetime"]).dt.week
dataset["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
dataset = df.set_index("Datetime")
dataset.index = pd.to_datetime(dataset.index)

#Data set we are going to work with
daily_data = dataset.resample('D').mean()
daily_data.shape

#Define the training set and the test set
training_set = daily_data.iloc[:-60,0:1].values
test_set = daily_data.tail(100)
real_AEP = test_set

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Number of days we want to use to predict the future
n_past = 60
# Number of days we want to predict into the future
n_future = 1


#Getting the inputs and the outputs for a time step equals to 60
X_train = []
y_train = []
for i in range(n_past,len(training_set) - n_future + 1):
    X_train.append(training_set[i-n_past:i])#list
    y_train.append(training_set[i + n_future - 1])
X_train = np.array(X_train)
y_train = np.array(y_train)    
X_train,y_train

#Reshape [samples,timestep,feature]
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

#Part2 - Building the RNN

#import the keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.core import Dropout

#initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))#that is what the model expects as input
#for each sample in terms of the number of time steps and the number of features.
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the RNN to the training set
regressor.fit(X_train,y_train,batch_size = 32 , epochs = 50)

# Part3 - Making the predictions and vusualisations
#Getting the predicted AEP
Df_Total = pd.concat((daily_data[["AEP_MW"]], test_set[["AEP_MW"]]), axis=0)
inputs = Df_Total.iloc[training_set.shape[0]:].values
inputs = sc.transform(inputs)
X_test = []
for i in range(n_past, inputs.shape[0]):
    X_test.append(inputs[i-n_past:i])
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predicted_AEP = regressor.predict(X_test)
predicted_AEP = sc.inverse_transform(predicted_AEP)
predicted_AEP = [x[0] for x in predicted_AEP ]
real_AEP = test_set["AEP_MW"].tolist()
dates = test_set.index.tolist()
Machine = pd.DataFrame({"date":dates,"Real":real_AEP,"Predicted":predicted_AEP})

#visualisations
x = Machine["date"]
y = Machine["Real"]
y1 = Machine["Predicted"]

plt.plot(x,y, color="green",label="Actual")
plt.plot(x,y1, color="red",label="Predicted")
# beautify the x-labels
plt.gcf().autofmt_xdate()
plt.xlabel('Dates')
plt.ylabel("Power in MW")
plt.title("Machine Learned the Pattern Predicting Future Values ")
plt.legend()
plt.show() 









