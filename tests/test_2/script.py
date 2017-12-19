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
aux = encoder.fit_transform(values[:,2]) # Encode WEEKDAY as an integer value
values = numpy.delete(values,2,1)
values[:,1] = encoder.fit_transform(values[:,1]) # Encode HOUR

print_smth("Integer encoded values", aux)

encoded_weekday = to_categorical(aux)

print(type(values))

print_smth("ENCODED WEEKDAYS", encoded_weekday)

print_smth("ENCODED WEEKDAYS", encoded_weekday.shape)



print_smth("Antes", values.shape)

values = numpy.append(values, encoded_weekday, axis = 1)


print_smth("Despues", values.shape)

#numpy.set_printoptions(threshold='nan')
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

# nn parameters
lstm_neurons = 30
batch_size   = 90
epochs       = 20


model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_x.shape[1]))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_data = (test_x, test_y), verbose = 2, shuffle = False)
model.save('test_1.h5')

print model.summary()

plt.figure(figsize=(12, 9))
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()

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


texto = "RMSE " +  str(rmse) + " batch size " + str(batch_size) + " epochs " + str(epochs) + " LSTM N " + str(lstm_neurons)
plt.text(0, 0, texto , fontsize=9)

ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

for y in numpy.linspace(min_y, max_y, 9):    
    plt.plot(range(0, epochs), [y] * len(range(0, epochs)), "--", lw=0.5, color="black", alpha=0.3) 
plots = ['train', 'test']

colores = [(31, 119, 180), (174, 199, 232)]

'''
for rank, column in enumerate(plots):
    # Plot each line separately with its own color, using the Tableau 20
    # color set in order.
    plt.plot(gender_degree_data.Year.values,
            gender_degree_data[column.replace("\n", " ")].values,
            lw=2.5, color=colores[rank])

    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    y_pos = gender_degree_data[column.replace("\n", " ")].values[-1] - 0.5
    if column == "train":
        y_pos += 0.5
    elif column == "test":
        y_pos -= 0.5

    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    plt.text(2011.5, y_pos, column, fontsize=14, color=colores[rank])
'''
plt.savefig("loss.png", bbox_inches="tight")

plt.close()


#plt.title("RMSE " +  str(rmse) + " batch size " + str(batch_size) + " epochs " + str(epochs) + " LSTM N " + str(lstm_neurons))
#plt.savefig("loss.png")
#plt.plot(test_y)
#plt.plot(yhat)
#plt.savefig("acc.png")

