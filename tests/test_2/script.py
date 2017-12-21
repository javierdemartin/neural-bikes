# neural-bikes test_2
# javier de martin @ december 2017

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

def print_smth(description, x):
    print ""
    print description
    print "-------------------------------------------------------------------------------------------"
    print x
    print "-------------------------------------------------------------------------------------------"

# Data reading
#-------------

dataset = read_csv(fileName + '.txt')

# Drop unwanted columns (bike station ID, bike station name...)
dataset.drop(dataset.columns[[0,2,5,6,7]], axis = 1, inplace = True)
dataset.columns = ['month','hour', 'weekday', 'free_bikes']

print_smth("Read dataset", dataset.head())

# Save the modified dataset into a new file
dataset.to_csv(fileName + '_parsed.txt')

# Data Pre-processing
#--------------------

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
aux         = encoder.fit_transform(values[:,2]) # Encode WEEKDAY as an integer value
values      = numpy.delete(values,2,1)           # Remove WEEKDAY column
values[:,1] = encoder.fit_transform(values[:,1]) # Encode HOUR

print_smth("Integer encoded values", aux)

encoded_weekday = to_categorical(aux)

values = numpy.append(values, encoded_weekday, axis = 1)
values = values.astype('float32') # Convert al values to floats

bikes_categorical = to_categorical(values[:,2])           # Convert bikes to int
values            = numpy.append(values, bikes_categorical, axis = 1)
values            = numpy.delete(values,2,1)           # Remove FREE_BIKES column

# create a new arrey w all the daata

#numpy.set_printoptions(threshold='nan')
print_smth("HEHE", values[0])
print values.shape

scaler      = MinMaxScaler(feature_range=(0,1))  # Normalize values
scaled      = scaler.fit_transform(values)
reframed    = series_to_supervised(scaled,1,1)

print reframed.shape

# Drop columns that I don't want to predict
reframed.drop(reframed.columns[[30,31,32,33,34,35,36,37,38]], axis = 1, inplace = True)


print reframed.shape

#numpy.set_printoptions(threshold='nan')
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

print "Shapes\n-----------------------"
print train_x.shape, train_y.shape, test_x.shape, test_y.shape

# Neural Net parameter definitions and training
#----------------------------------------------

lstm_neurons = 30
batch_size   = 90
epochs       = 5

# NN model definition
model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_x.shape[1], activation = 'softmax'))
#model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)

model.save('test_2.h5')

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


# Plot styling
#-------------

plt.figure(figsize=(12, 9))
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

ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

x = history.history['loss']

plt.xticks(numpy.arange(min(x), len(x), 1.0))

plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

for y in numpy.linspace(min_y, max_y, 9):    
    plt.plot(range(0, epochs), [y] * len(range(0, epochs)), "--", lw=0.5, color="black", alpha=0.3) 

plt.savefig("loss.png", bbox_inches="tight")

plt.close()


#plt.title("RMSE " +  str(rmse) + " batch size " + str(batch_size) + " epochs " + str(epochs) + " LSTM N " + str(lstm_neurons))
#plt.savefig("loss.png")
#plt.plot(test_y)
#plt.plot(yhat)
#plt.savefig("acc.png")

