# Multivariate Time Series Forecasting with LSTMs

from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import array, reshape, concatenate, argmax
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from pandas import concat,DataFrame, read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import to_categorical

fileName = 'Zunzunegi'

################################################################################
# Function definition
################################################################################

def print_smth(description, x):
    print ""
    print description
    print "-------------------------------------------------------------------------------------------"
    print x
    print "-------------------------------------------------------------------------------------------"


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

################################################################################
# Data preparation
################################################################################

# Data reading
#--------------------------------------------------------------------------------

dataset = read_csv(fileName + '.txt')

# Drop unwanted columns (bike station ID, bike station name...)
dataset.drop(dataset.columns[[0,2,5,6,7]], axis = 1, inplace = True)
dataset.columns = ['month','hour', 'weekday', 'free_bikes']
dataset.to_csv(fileName + '_parsed.txt')

# Data Preparation
#--------------------------------------------------------------------------------

dataset = read_csv(fileName + '_parsed.txt', header=0, index_col = 0)
values  = dataset.values

print_smth("Imported dataset", dataset.head())

encoder                = LabelEncoder()                     # Encode columns that are not numbers
integerEncoded_weekday = encoder.fit_transform(values[:,2]) # Encode WEEKDAY as an integer value
values                 = numpy.delete(values,2,1)
values[:,1]            = encoder.fit_transform(values[:,1]) # Encode HOUR as int

print_smth("Integer encoded values", integerEncoded_weekday)

encoded_weekday      = to_categorical(integerEncoded_weekday)
integerEncoded_bikes = to_categorical(values[:,2])
#values               = numpy.delete(values,2,1) # Delete original BIKES column

values = numpy.append(values, encoded_weekday, axis = 1)

print_smth("To categorical", values)

values      = values.astype('float32')           # Convert al values to floats

values[:,2]      = values[:,2].astype(int)           # Convert bikes to int
scaler      = MinMaxScaler(feature_range=(0,1))  # Normalize values
scaled      = scaler.fit_transform(values)

reframed    = series_to_supervised(scaled,1,1)

# Drop columns that I don't want to predict
# (0) Month(t-1) | (1) Hour(t-1) | (2) Weekday(t-1) | (3) Free Bikes (t-1) 
# (4) Month(t)   | (5) Hour(t)   | (6) Weekday(t)   | (7) Free Bikes (t) 
reframed.drop(reframed.columns[[10,11,13,14,15,16,17,18,19]], axis = 1, inplace = True)

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

################################################################################
# Model Training
################################################################################

# nn parameters
lstm_neurons = 30
batch_size   = 90
epochs       = 10


model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_x.shape[1]))
#model.add(Dense(train_x.shape[1], activation = 'softmax'))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)
model.save('test_1.h5')

print model.summary()

min_y = min(history.history['loss'])
max_y = max(history.history['loss'])

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


model.save('test_2.h5')

################################################################################
# Plot styling
################################################################################


ax = plt.subplot(111)    

def prepare_plot():
    plt.figure(figsize=(12, 9))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 


    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Loss
#--------------------------------------------------------------------------------

prepare_plot()
plt.plot(history.history['loss'], label = 'train', color = 'blue')
plt.plot(history.history['val_loss'], label = 'test', color = 'teal')

plt.text((len(history.history['loss']) - 1) * 1.005, 
         history.history['loss'][len(history.history['loss']) - 1], 
         "loss", color = 'blue')

plt.text((len(history.history['val_loss']) - 1) * 1.005, 
         history.history['val_loss'][len(history.history['val_loss']) - 1], 
         "val_loss", color = 'teal')


texto = "RMSE " +  str(rmse) + " batch size " + str(batch_size) + " epochs " + str(epochs) + " LSTM N " + str(lstm_neurons)
plt.text(0, 0, texto , fontsize=9)

x = history.history['loss']


plt.xticks(numpy.arange(min(x), len(x), 1.0))
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

for y in numpy.linspace(min_y, max_y, 9):    
    plt.plot(range(0, epochs), [y] * len(range(0, epochs)), "--", lw=0.5, color="black", alpha=0.3) 

plt.savefig("loss.png", bbox_inches="tight")

plt.savefig("loss.png")
plt.close()

#--------------------------------------------------------------------------------
# Accuracy
#--------------------------------------------------------------------------------

prepare_plot()
plt.plot(test_y)
plt.plot(inv_y)
plt.plot(yhat)
plt.savefig("acc.png")

