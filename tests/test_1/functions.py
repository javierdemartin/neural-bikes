import pandas.core.frame
import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas import concat,DataFrame
from math import sqrt
import numpy
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from keras.utils import plot_model



def readWeatherData(fileName):

	dataset         = pandas.read_csv('data/parsed.txt')
	# dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 

	print "WEATHER"
	print dataset

	return dataset


# Reads txt file containing the data
def readFile(stationToRead):


	substrings = ["00", "10", "20", "30", "40", "50"]

	dataset         = pandas.read_csv('data/Bilbao.txt')
	# dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 

	# Only store the desired columns for the specified station
	dataset         = dataset[dataset['station'].isin([stationToRead])]

	print(dataset)

	dataset.drop(dataset.columns[[2,3,5]], axis = 1, inplace = True) # Remove ID of the sation and free docks

	print(dataset)

	dataset = dataset.reset_index(drop = True)
	values  = dataset.values

	times = [x.split(" ")[1] for x in values[:,0]]

	dataset['datetime'] = [datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timetuple().tm_yday for x in values[:,0]]

	dataset.insert(loc = 1, column = 'time', value = times)


	values  = dataset.values


	dataset = dataset[dataset.time.str[-1:] != "5"]

	dataset = dataset.drop('datetime', axis = 1)

	print "Dropped 5 interval"
	print dataset


	weather = readWeatherData("H")
	weather.drop(weather.columns[[0]], axis = 1, inplace = True) # Remove ID of the sation and free docks	

	print "WEATHER REMOVED"
	print weather

	dataset = dataset[:weather.shape[0]].reset_index(drop = True)

	dataset = pandas.concat([dataset, weather], axis=1)



	print("Dataset with unwanted columns removed", dataset)


	# dataset = dataset[['time', 'weekday', 'precipitation', 'temperature', 'humidity', 'windspeed', 'free_bikes']]

	return dataset

# input dataframe

def formatData(dataset):

	values = dataset.values

	dayEncoder     = LabelEncoder()
	hourEncoder    = LabelEncoder()
	weekdayEncoder = LabelEncoder()

	values[:,0] = dayEncoder.fit_transform(values[:,0])
	values[:,1] = hourEncoder.fit_transform(values[:,1])
	values[:,2] = weekdayEncoder.fit_transform(values[:,2])

	print("JAVI ", values.shape)
	print(values)

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	reframed = series_to_supervised(scaled, 1, 1)

	print("REFRAMED")
	print(reframed)

	# reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True)
	reframed.drop(reframed.columns[[7,8,10,11,12,13]], axis=1, inplace=True)

	print("POST REFRAMED")
	print(reframed)


	return reframed.values, scaler

def split_dataset(dataset):

	testSize = 288
	trainSize = len(dataset) - testSize
	train = dataset[:trainSize, :]
	test = dataset[trainSize:, :]

	print("Training on " + str(trainSize) + "/" + str(len(dataset)) + " samples")

	# split into input and outputs
	train_x, train_y = train[:, :-1], train[:, -1]
	test_x, test_y = test[:, :-1], test[:, -1]

	# reshape input to be 3D [samples, timesteps, features]
	train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
	test_x  = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

	print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

	print "Train X"
	print train_x
	print "Train Y"
	print train_y

	print "Test X"
	print test_x
	print "Test Y"
	print test_y


	return train_x, train_y, test_x, test_y

def create_model(train_x):
	
	# design network
	model = Sequential()
	model.add(LSTM(200, input_shape=(train_x.shape[1], train_x.shape[2])))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam', metrics = ["acc"])

	return model

def train_model(model, train_x, train_y, test_x, test_y):

	history = model.fit(train_x, train_y, epochs=500, batch_size=288, validation_data=(test_x, test_y), verbose=2, shuffle=False)
	# plot history

	plot_model(model, to_file='plots/model.png', show_shapes = True)


	my_plot(history.history['acc'], history.history['val_acc'], "Accuracy")
	my_plot(history.history['loss'], history.history['val_loss'], "Loss")

def predict(model, scaler, average):

	# today = datetime.datetime.now().timetuple().tm_yday # Current day of the year
	hour = 0
	weekday = 1
	bikes = 16

	predicted = []

	for i in range(0,287):

		# og = numpy.array([[today, hour, weekday, bikes]])
		og = numpy.array([[hour, weekday, bikes]])

		datos = scaler.transform(og)

		datos = datos.reshape(1,1,7)


		bikes = model.predict(datos)[0][0] * 20 #int(model.predict(datos)[0][0] * 20)

		print(bikes)

		predicted.append(bikes)

		hour += 1



	my_plot(predicted, average, "Pred")


def evaluate_model(scaler, model, test_x, test_y):

	print(test_x)
	print(test_x.shape)
	print(type(test_x))

	# make a prediction
	yhat = model.predict(test_x)
	test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	# calculate RMSE

	my_plot(inv_y, inv_yhat, 'evaluation')

	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)

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

def my_plot(x1,x2, name):

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	ax = plt.axes(frameon=False)

	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	# plt.xlabel(xlabel, color = 'silver', fontsize = 17)
	# plt.ylabel(ylabel, color = 'silver', fontsize = 17)

	lines  = plt.plot(x1, color = '#458DE1')
	lines  += plt.plot(x2, color = '#f49c0e')

	# plt.xticks(dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24], dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24])

	plt.setp(lines, linewidth=2)


	texto = "Disponibilidad y media en "
	plt.title(texto,color="black", alpha=0.3)
	plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

	# plt.show()
	plt.savefig("plots/" + name + ".png", bbox_inches="tight")

	plt.close()