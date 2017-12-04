# Multivariate Time Series Forecasting with LSTMs

from math import sqrt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from matplotlib import pyplot
from numpy import array, reshape, concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas import concat,DataFrame, read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation

fileName = 'Zunzunegi'

def print_smth(description, x):
    print ""
    print description
    print "-------------------------------------------------------------------------------------------"
    print x
    print "-------------------------------------------------------------------------------------------"

# Data reading
##############

dataset = read_csv(fileName + '.txt')

# Drop unwanted columns (bike station ID, bike station name...)
dataset.drop(dataset.columns[[0,2,5,6,7]], axis = 1, inplace = True)

dataset.columns    = ['month','hour', 'weekday', 'free_bikes']

print_smth("Read dataset", dataset.head())

# Save the modified dataset into a new file
dataset.to_csv(fileName + '_parsed.txt')

# Data Preparation
####################

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


dataset = read_csv(fileName + '_parsed.txt', header=0, index_col = 0)
values  = dataset.values

print_smth("Imported dataset", dataset.head())

encoder     = LabelEncoder()                     # Encode columns that are not numbers
values[:,2] = encoder.fit_transform(values[:,2]) # Encode WEEKDAY
values[:,1] = encoder.fit_transform(values[:,1]) # Encode HOUR
values      = values.astype('float32')           # Convert al values to floats
scaler      = MinMaxScaler(feature_range=(0,1))  # Normalize values
scaled      = scaler.fit_transform(values)
reframed    = series_to_supervised(scaled,1,1)

# Drop columns that I don't want to predict
# (0) Month(t-1) | (1) Hour(t-1) | (2) Weekday(t-1) | (3) Free Bikes (t-1) 
# (4) Month(t)   | (5) Hour(t)   | (6) Weekday(t)   | (7) Free Bikes (t) 
reframed.drop(reframed.columns[[4, 5, 6]], axis = 1, inplace = True)

print_smth("reframed dataset", reframed.head())

values     = reframed.values
train_size = int(len(values) * 0.67) # Train on 67% test on 33%

print_smth("Training set size (samples)", train_size)

train = values[0:train_size,:]
test  = values[train_size:len(values), :]

train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y   = test[:, :-1], test[:, -1]

# reshape input to be [samples, time_steps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x  = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print train_x.shape, train_y.shape, test_x.shape, test_y.shape

lstm_neurons = 30

model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_x.shape[1]))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])

batch_size = 90
epochs     = 30

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)
model.save('test_1.h5')

print model.summary()

pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()

# make a prediction
yhat   = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y  = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y  = scaler.inverse_transform(inv_y)
inv_y  = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.title("RMSE " +  str(rmse) + " batch size " + str(batch_size) + " epochs " + str(epochs) + " LSTM N " + str(lstm_neurons))
pyplot.savefig("loss.png")
