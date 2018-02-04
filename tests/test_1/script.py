# Multivariate Time Series Forecasting with LSTMs
#
# Summary
#-------------------------------------------------------------------------------
# Import the data and witouth doing any transformation predict


################################################################################
# Libraries and Imports
################################################################################

from math import sqrt
import numpy
import pandas
import matplotlib.pyplot as plt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame, read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from datetime import datetime
from keras.utils import to_categorical

stationToRead = 'AYUNTAMIENTO'

################################################################################
# Classes and Functions
################################################################################

# Colorful prints in the terminal
class col:
    HEADER    = '\033[95m'
    blue      = '\033[94m'
    green     = '\033[92m'
    yellow    = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

# Formatted output
def print_smth(description, x):
    print "", col.yellow
    print description
    print "----------------------------------------------------------------------------", col.ENDC
    print x
    print col.yellow, "----------------------------------------------------------------------------", col.ENDC


# Convert series to supervised learning
# Arguments
#  * [Columns]> Array of strings to name the supervised transformation
def series_to_supervised(columns, data, n_in=1, n_out=1, dropnan=True):

        """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""

        n_vars = 1 if type(data) is list else data.shape[1]
	dataset = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dataset.shift(i))
        names += [(columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dataset.shift(-i))
		if i == 0:
		    #names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                    names += [(columns[j] + '(t)') for j in range(n_vars)]
		else:
                    names += [(columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def prepare_plot(xlabel, ylabel, min_y, max_y):

    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax = plt.axes(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


    plt.xlabel(xlabel, color = 'silver')
    plt.ylabel(ylabel, color = 'silver')

    # for y in numpy.linspace(min_y, max_y, 9):
    #     plt.plot(range(0, epochs), [y] * len(range(0, epochs)), "--", lw=0.5, color="black", alpha=0.3)


################################################################################
# Data preparation
################################################################################

#--------------------------------------------------------------------------------
# File reading, drop non-relevant columns and save it to a new file
#--------------------------------------------------------------------------------

dataset = pandas.read_csv('Bilbao.txt')
print dataset.head()
dataset.columns = ['year', 'month', 'day', 'hour', 'weekday', 'id', 'station', 'free_bikes', 'free_docks']
# Only maintain rows for the analyzed station
dataset = dataset[dataset['station'].isin([stationToRead])]

print dataset.head()

dataset.drop(dataset.columns[[0,2,5,6,8]], axis = 1, inplace = True)
dataset = dataset.reset_index(drop = True)

#--------------------------------------------------------------------------------
# Data reading
#--------------------------------------------------------------------------------

values  = dataset.values

# 4 columns (month, hour, weekday, free_bikes)
print_smth("Dataset with unwanted columns removed", dataset.head())

#--------------------------------------------------------------------------------
# Data encoding
#--------------------------------------------------------------------------------

encoder     = LabelEncoder()                     # Encode columns that are not numbers
values[:,2] = encoder.fit_transform(values[:,2]) # Encode WEEKDAY as an integer value
values[:,1] = encoder.fit_transform(values[:,1]) # Encode HOUR as int
values      = values.astype('float32')           # Convert al values to floats

scaler = MinMaxScaler(feature_range=(0,1))  # Normalize values
scaled = scaler.fit_transform(values)


#--------------------------------------------------------------------------------
# Generate the columns list for the supervised transformation
#--------------------------------------------------------------------------------

# Columns names for the transformation from time series to supervised learning
columns = ['month', 'hour', 'weekday', 'free bikes']

# outputs 8 columns
# month(t-1) | hour(t-1) | weekday (t-1) | free_bikes(t-1) | month(t) | ...
reframed = series_to_supervised(columns, scaled,1,1)

# Drop columns that I don't want to predict, every (t) column except free_bikes(t)
# month(t-1) | hour(t-1) | weekday (t-1) | free_bikes(t-1) | free_bikes(t) |
#--------------------------------------------------------------------------
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True)

values = reframed.values

print_smth("Reframed dataset without columns that are not going to be predicted", reframed.head())

train_size      = int(len(values) * 0.70) # Divide the set into training and test sets
evaluation_size = int(len(values) * 0.1)
test_size       = int(len(values) * 0.20)

# Divide between train and test sets
train      = values[0:train_size,:]
test       = values[train_size:train_size + test_size, :]
evaluation = values[train_size + test_size:len(values), :]

# train_x: gets  the first four columns month(t-1) | hour(t-1) | weekday (t-1) | free_bikes(t-1)
# train_y: gets the last column free_bikes(t)
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y   = test[:, :-1], test[:, -1]
evaluation_x, evaluation_y = evaluation[:, :-1], evaluation[:, -1]

print_smth("Train X", train_x.shape) # (..., 4)
print_smth("Train Y", train_y.shape) # (..., )
print_smth("Test X", test_x.shape)   # (..., 4)
print_smth("Test X", test_y.shape)   # (..., 4)
print_smth("Evaluation X", evaluation_x.shape)   # (..., 4)
print_smth("Evaluation X", evaluation_y.shape)   # (..., 4)

# reshape input to be [samples, time_steps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1])) # (...,1,4)
test_x  = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))    # (...,1,4)
evaluation_x  = evaluation_x.reshape((evaluation_x.shape[0], 1, evaluation_x.shape[1]))    # (...,1,4)

print train_x.shape, train_y.shape, test_x.shape, test_y.shape, evaluation_x.shape, evaluation_y.shape

################################################################################
# Neural Network
################################################################################

#--------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------
lstm_neurons = 200
batch_size   = 800
epochs       = 15

#--------------------------------------------------------------------------------
# Network definition
#--------------------------------------------------------------------------------
model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_x.shape[1]))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)
model.save('test_1.h5')

print_smth("Model summary", model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model.png')

#--------------------------------------------------------------------------------
# Make a prediction
#--------------------------------------------------------------------------------

min_y = min(history.history['loss'])
max_y = max(history.history['loss'])

yhat   = model.predict(test_x)

print_smth("test_x", test_x)

test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_x[:, :-1], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,3].astype(int) # Cast as int, bikes are integer numbers...
                                     # Only get the last column (predicted bikes)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))

# Real values
inv_y = concatenate((test_x[:, :-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,3].astype(int) # Cast as int, bikes are integer numbers...
                               # Only get the last column (real bikes)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

################################################################################
# Network Evaluation
################################################################################

# Evaluate the network once is trained, do so with new data (evaluation set)

loss, accuracy = model.evaluate(evaluation_x, evaluation_y)

print loss, accuracy


################################################################################
# Plot styling
################################################################################

#--------------------------------------------------------------------------------
# Loss Plot
#--------------------------------------------------------------------------------

prepare_plot('epochs', 'loss', min_y, max_y)

lines  = plt.plot(history.history['loss'], label = 'train', color = '#458DE1')
lines += plt.plot(history.history['val_loss'], label = 'test', color = '#80C797')

plt.setp(lines, linewidth=2)

plt.text((len(history.history['loss']) - 1) * 1.005,
         history.history['loss'][len(history.history['loss']) - 1] + 0.01,
         "Training Loss", color = '#458DE1')

plt.text((len(history.history['val_loss']) - 1) * 1.005,
         history.history['val_loss'][len(history.history['val_loss']) - 1],
         "Validation Loss", color = '#80C797')

texto = "RMSE " +  str('%.3f' % (rmse))  + " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " |  " + str(lstm_neurons) + " LSTM neurons"
plt.title(texto,color="black", alpha=0.3)
plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

plt.savefig("plots/loss.png", bbox_inches="tight")
plt.close()

#--------------------------------------------------------------------------------
# Accuracy Plot
#--------------------------------------------------------------------------------



prepare_plot('epoc', 'accuracy', min_y, max_y)

plt.title(texto,color="black", alpha=0.3)

lines = plt.plot(history.history['acc'], color = '#458DE1')
lines += plt.plot(history.history['val_acc'], color = '#80C797')

plt.setp(lines, linewidth=3)


plt.text((len(history.history['acc']) - 1) * 1.005,
         history.history['acc'][len(history.history['loss']) - 1],
         "Training Accuracy", color = '#458DE1')

plt.text((len(history.history['val_acc']) - 1) * 1.005,
         history.history['val_acc'][len(history.history['val_loss']) - 1],
         "Validation Accuracy", color = '#80C797')

plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')
plt.savefig("plots/acc.png", bbox_inches="tight")
plt.close()

#--------------------------------------------------------------------------------
# Training Plot
#--------------------------------------------------------------------------------

min_y = min(inv_y)
max_y = max(inv_y)

prepare_plot('samples', 'bikes', min_y, max_y)

plt.title(texto,color="black", alpha=0.3)

lines = plt.plot(inv_yhat, color = '#458DE1')
lines += plt.plot(inv_y, label = 'inv_y', color = '#80C797')
plt.setp(lines, linewidth=2)


plt.text((len(inv_yhat) - 1) * 1.005,
         inv_yhat[len(inv_yhat) - 1],
         "Training set", color = '#458DE1')

plt.text((len(inv_y) - 1) * 1.005,
         inv_y[len(inv_y) - 1],
         "Predicted set", color = '#80C797')

plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

plt.savefig("plots/train.png", bbox_inches="tight")
