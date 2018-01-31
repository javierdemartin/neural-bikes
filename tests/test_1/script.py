# Multivariate Time Series Forecasting with LSTMs
#
# Summary
#-------------------------------------------------------------------------------
# Import the data and witouth doing any transformation predict


################################################################################
# Libraries
################################################################################

from math import sqrt
import numpy
import matplotlib.pyplot as plt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame, read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import to_categorical

fileName = 'Zunzunegi'

class col:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_smth(description, x):
    print "", col.WARNING
    print description
    print "----------------------------------------------------------------------------", col.ENDC
    print x
    print col.WARNING, "----------------------------------------------------------------------------", col.ENDC


# convert series to supervised learning
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
        print ">>>>>>>>>>", n_vars
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
        names += [(columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
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


################################################################################
# Data preparation
################################################################################

#--------------------------------------------------------------------------------
# File reading, drop non-relevant columns and save it to a new file
#--------------------------------------------------------------------------------

dataset = read_csv(fileName + '.txt')

# Drop unwanted columns (bike station ID, bike station name...)
dataset.drop(dataset.columns[[0,2,5,6,7]], axis = 1, inplace = True)
dataset.columns = ['month','hour', 'weekday', 'free_bikes']
dataset.to_csv(fileName + '_parsed.txt')

#--------------------------------------------------------------------------------
# Data reading
#--------------------------------------------------------------------------------

dataset = read_csv(fileName + '_parsed.txt', header=0, index_col = 0)
values  = dataset.values

print_smth("VALUES", values)

print_smth("Imported dataset from the file", dataset.head())

print col.WARNING, "> ", values.shape, col.ENDC

#--------------------------------------------------------------------------------
# Data encoding
#--------------------------------------------------------------------------------

encoder     = LabelEncoder()                     # Encode columns that are not numbers
values[:,2] = encoder.fit_transform(values[:,2]) # Encode WEEKDAY as an integer value
values[:,1] = encoder.fit_transform(values[:,1]) # Encode HOUR as int
values      = values.astype('float32')           # Convert al values to floats

print_smth("VALUES", values)

scaler = MinMaxScaler(feature_range=(0,1))  # Normalize values
scaled = scaler.fit_transform(values)


#
# Generate the columns list for the supervised transformation
#

columns = ['month', 'hour', 'weekday', 'free bikes']

############################################################

reframed    = series_to_supervised(columns, scaled,1,1) # 60 columns

# Drop columns that I don't want to predict
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True)

values = reframed.values

print_smth("reframed dataset", reframed.head())

train_size = int(len(values) * 0.67) # Train on 67% test on 33%

print_smth("Training set size (samples)", train_size)

train = values[0:train_size,:]
test  = values[train_size:len(values), :]

# IMPORTANTE
# Quedate con todo excepto la ultima columna que es la que se predice

train_x, train_y = train[:, :-1], train[:, -1]

print_smth("HEHEHEEH", train_x)

test_x, test_y   = test[:, :-1], test[:, -1]


print_smth("Train X", train_x.shape)
print_smth("Train Y", train_y.shape)
print_smth("Test X", test_x.shape)
print_smth("Test X", test_y.shape)

# reshape input to be [samples, time_steps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x  = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print train_x.shape, train_y.shape, test_x.shape, test_y.shape

################################################################################
# Neural Network
################################################################################

#--------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------
lstm_neurons = 50
batch_size   = 90
epochs       = 50

# Training
#--------------------------------------------------------------------------------
model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
print(train_x.shape[1], train_x.shape[2])
model.add(Dense(train_x.shape[1]))

model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print model.summary()

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)
model.save('test_1.h5')

print model.summary()

#--------------------------------------------------------------------------------
# Make a prediction
#--------------------------------------------------------------------------------

min_y = min(history.history['loss'])
max_y = max(history.history['loss'])

# make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_x[:, :-1], yhat), axis=1)

print_smth("inv_yhat concatenated", inv_yhat)

inv_yhat = scaler.inverse_transform(inv_yhat)



inv_yhat = inv_yhat[:,3].astype(int)
print_smth("inv_yhat", inv_yhat)



# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))

print_smth("test_y", test_y)
print_smth("test_x", test_x[:, 1:])

inv_y = concatenate((test_x[:, :-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)



inv_y = inv_y[:,3].astype(int)
print_smth("inv_y", inv_y)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

################################################################################
# Plot styling
################################################################################

ax = plt.subplot(111)

def prepare_plot():
    plt.figure(figsize=(12, 9))
    ax = plt.axes(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

#--------------------------------------------------------------------------------
# Loss
#--------------------------------------------------------------------------------

prepare_plot()

lines  = plt.plot(history.history['loss'], label = 'train', color = 'blue')
lines += plt.plot(history.history['val_loss'], label = 'test', color = 'teal')

plt.setp(lines, linewidth=2)

plt.text((len(history.history['loss']) - 1) * 1.005,
         history.history['loss'][len(history.history['loss']) - 1],
         "loss", color = 'blue')

plt.text((len(history.history['val_loss']) - 1) * 1.005,
         history.history['val_loss'][len(history.history['val_loss']) - 1],
         "val_loss", color = 'teal')

texto = "RMSE " +  str('%.3f' % (rmse))  + " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " | LSTM N " + str(lstm_neurons)
plt.title(texto)

x = history.history['loss']

plt.xticks(numpy.arange(min(x), len(x), 1.0))
plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

for y in numpy.linspace(min_y, max_y, 9):
    plt.plot(range(0, epochs), [y] * len(range(0, epochs)), "--", lw=0.5, color="black", alpha=0.3)

plt.savefig("loss.png", bbox_inches="tight")
plt.close()


#--------------------------------------------------------------------------------
# Accuracy
#--------------------------------------------------------------------------------

prepare_plot()
plt.title(texto)
plt.setp(lines, linewidth=2)
plt.plot(inv_yhat, color = 'teal')
plt.plot(inv_y, label = 'inv_y', color = 'orange')
#plt.plot(list(inv_yhat[:,3]))#, label = 'inv_yhat', color = 'aqua')
plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')



plt.savefig("acc.png", bbox_inches="tight")
