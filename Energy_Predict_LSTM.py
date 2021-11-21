import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('household_power_consumption.txt',
                   sep=';',                            # separateur = ;
                   header=0,                           # ligne des header = 1ère ligne
                   low_memory=False,                   # parsing complet
                   infer_datetime_format=True,         # parsing des dates
                   parse_dates={'datetime':[0,1]},     # les dates se parsent via les 2 premières colonnes
                   index_col=['datetime']) 

dataset = dataset.dropna()
dataset = dataset.astype('float32')   

dataset = dataset[['Global_active_power', 'Global_reactive_power', 'Voltage',
       'Global_intensity', 'Sub_metering_2', 'Sub_metering_1','Sub_metering_3']]

hourly_data = dataset.resample('h').mean() 

#Scaling the entire data
data = hourly_data.values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data_scaled = sc.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled , index=hourly_data.index.tolist(),columns=
                          hourly_data.columns.tolist())


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data,index=hourly_data.index.tolist())
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] 
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))#dans ce cas 0 ce sont les colonnes du output
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)] 
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
from sklearn.preprocessing import MinMaxScaler
values = hourly_data.values
#normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
r = list(range(hourly_data.shape[1]+1, 2*hourly_data.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)

values = reframed.values
n_train_time = 30000
training_set = values[:n_train_time, :]
testing_set = values[n_train_time:, :]
X_train, y_train = training_set[:, :-1], training_set[:,-1]
X_test, y_test = testing_set [:, :-1], testing_set [:,-1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))



from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
model = Sequential()
model.add(LSTM(32,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))

#model.add(LSTM(32,return_sequences=True))
#model.add(Dropout(0.1))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

# Network fitting
history = model.fit(X_train, y_train, epochs=50, batch_size=70, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# Loss history plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


from sklearn.metrics import mean_squared_error
size = hourly_data.shape[1]
# Prediction test
predicted_value_scaled = model.predict(X_test)
predicted_value_scaled = predicted_value_scaled.reshape((predicted_value_scaled.shape[0],predicted_value_scaled.shape[1]))
print(predicted_value_scaled.shape)
X_test = X_test.reshape((X_test.shape[0], size))
print(X_test.shape)

# invert scaling for prediction
all_values_scaled = np.concatenate((predicted_value_scaled, X_test[:, 1-size:]), axis=1)
all_values = scaler.inverse_transform(all_values_scaled)
predicted_value = all_values[:,0]

# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
dataset_scaled = np.concatenate((y_test, X_test[:, 1-size:]), axis=1)
all_dataset = scaler.inverse_transform(dataset_scaled)
actual_value = all_dataset[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))
print('Test RMSE: %.3f' % rmse)


aa=[x for x in range(500)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, actual_value[:500], marker='.', label="actual",color="green")
plt.plot(aa, predicted_value[:500], 'r', label="prediction",color="red")
plt.ylabel(dataset.columns[0], size=15)
plt.xlabel('Time step for first 500 hours', size=15)
plt.legend(fontsize=15)
plt.show()



