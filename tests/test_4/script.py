# Multivariate Time Series Forecasting with LSTMs

#
# Summary
#-------------------------------------------------------------------------------
# Import data and categorize the bikes, change LSTM layers' settings
# Doesn't predict well, it learns the values and repeats the previous interval

################################################################################
# Libraries and Imports
################################################################################

# Parameters
# lstm_neurons batch_size epochs n_in

from math import sqrt
import math
import numpy
# import pandas

import matplotlib
matplotlib.use('Agg') # Needed when running on headless server
import sys
import matplotlib.pyplot as plt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime
import datetime
from numpy import argmax
import pandas.core.frame
from sklearn.externals import joblib
import os
import csv

os.system("reset") # Clears the screen

################################################################################
# Global Variables
################################################################################

stationToRead = 'IRALA'
is_in_debug = True
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

list_of_stations = ["PLAZA LEVANTE", "IRUÑA", "AYUNTAMIENTO", "PLAZA ARRIAGA", "SANTIAGO COMPOSTELA", "PLAZA REKALDE", "DR. AREILZA", "ZUNZUNEGI", "ASTILLERO", "EGUILLOR", "S. CORAZON", "PLAZA INDAUTXU", "LEHENDAKARI LEIZAOLA", "CAMPA IBAIZABAL", "POLID. ATXURI", "SAN PEDRO", "KARMELO", "BOLUETA", "OTXARKOAGA", "OLABEAGA", "SARRIKO", "HEROS", "EGAÑA", "P. ETXEBARRIA", "TXOMIN GARAT", "ABANDO", "ESTRADA CALEROS", "EPALZA", "IRALA", "S. ARANA", "C. MARIA"]
print(len(list_of_stations))

lstm_neurons = int(sys.argv[1]) # 50
batch_size   = 1
epochs       = int(sys.argv[3]) # 30
n_in         = int(sys.argv[4]) # 10
n_out        = 1

new_batch_size = int(sys.argv[2]) #1000

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

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

def create_model(batch_size, statefulness):

	model = Sequential()
	model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=statefulness))
	model.add(Dense(n_out, activation = 'relu'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse', 'acc'])

	return model

# Formatted output
def print_smth(description, x):
	
	if is_in_debug == True:
		print("", col.yellow)
		print(description)
		print("----------------------------------------------------------------------------", col.ENDC)
		print(x)
		print(col.yellow, "----------------------------------------------------------------------------", col.ENDC)

# Formatted output
def print_array(description, x):
	
	if is_in_debug == True:
		print("", col.yellow)
		print(description, " ", x.shape)
		print("----------------------------------------------------------------------------", col.ENDC)
		print(x)
		print(col.yellow, "----------------------------------------------------------------------------", col.ENDC)

def prepare_plot(xlabel, ylabel, plot_1, plot_2, name):

	min_y = min(plot_1)
	max_y = max(plot_1)

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

	lines  = plt.plot(plot_1, label = 'train', color = '#458DE1')

	if len(plot_2) > 0:
		lines += plt.plot(plot_2, label = 'test', color = '#80C797')

	plt.setp(lines, linewidth=2)

	plt.text((len(plot_1) - 1) * 1.005,
		 plot_1[len(plot_1) - 1] + 0.01,
		 "Training Loss", color = '#458DE1')

	if len(plot_2) > 0:
		plt.text((len(plot_2) - 1) * 1.005,
		 plot_2[len(plot_2) - 1],
		 "Validation Loss", color = '#80C797')

	# texto = "RMSE " +  str('%.3f' % (rmse))  + " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " |  " + str(lstm_neurons) + " LSTM neurons"
	texto = " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " |  " + str(lstm_neurons) + " LSTM neurons"
	plt.title(texto,color="black", alpha=0.3)
	plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

	plt.savefig("plots/" + name + ".png", bbox_inches="tight")
	plt.close()
	print(col.HEADER + "> Loss plot saved" + col.ENDC)

# Convert series to supervised learning
# Arguments
#  * [Columns]> Array of strings to name the supervised transformation
def series_to_supervised(columns, data, n_in=1, n_out=1, dropnan=True):

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

	print_array("Reframed dataset after converting series to supervised", agg.head())

	return agg

# Calculates the number of incorrect samples
def calculate_no_errors(predicted, real):

	wrong_val = 0

	for i in range(0, len(predicted)):
		if predicted[i] != real[i]:
			wrong_val += 1

	print(wrong_val, "/", len(predicted))

# Returns the array with generated cyclic encoding UNSCALED
# Input data has to be encoded previously

def create_array(doy, hour, weekday, bikes):

	array = numpy.array(numpy.sin(2. * numpy.pi * doy / (max_time + 1)))      # Day of the year SIN
	array = numpy.append(array,numpy.cos(2. * numpy.pi * doy / (max_time + 1)))   # Day of the year COS

	array = numpy.append(array,numpy.sin(2. * numpy.pi * hour / (max_hour + 1)))    # Hour SIN
	array = numpy.append(array,numpy.cos(2. * numpy.pi * hour / (max_hour + 1)))    # Hour COS

	array = numpy.append(array,numpy.sin(2. * numpy.pi * weekday / (max_wday + 1))) # Weekday SIN
	array = numpy.append(array,numpy.cos(2. * numpy.pi * weekday / (max_wday + 1))) # Weekday COS

	array = numpy.append(array,bikes) 

	return array

print(col.HEADER)
print("   __            __     _____")
print("  / /____  _____/ /_   |__  /")
print(" / __/ _ \/ ___/ __/    /_ < ")
print("/ /_/  __(__  ) /_    ___/ / ")
print("\__/\___/____/\__/   /____/  ", col.ENDC)

if os.path.isdir("plots") == False:
	os.system("mkdir plots")

################################################################################
# Data preparation
################################################################################

print(col.HEADER + "Data reading and preparation" + col.ENDC)

#--------------------------------------------------------------------------------
# File reading, drop non-relevant columns like station id, station name...
#--------------------------------------------------------------------------------

dataset         = pandas.read_csv('data/Bilbao.txt')
dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 
dataset         = dataset[dataset['station'].isin([stationToRead])]

print(col.HEADER, "> Data from " + dataset['datetime'].iloc[0], " to ", dataset['datetime'].iloc[len(dataset) - 1], col.ENDC)

print_array("Read dataset", dataset.head())
print_array("Read dataset", dataset)

dataset.drop(dataset.columns[[2,3,5]], axis = 1, inplace = True) # Remove ID of the sation and free docks

print_array("Read dataset", dataset)

dataset = dataset.reset_index(drop = True)
values  = dataset.values

#--------------------------------------------------------------------------------
#-- Data reading ----------------------------------------------------------------
#
# Split datetime column into day of the year and time
#--------------------------------------------------------------------------------

times = [x.split(" ")[1] for x in values[:,0]]

dataset['datetime'] = [datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timetuple().tm_yday for x in values[:,0]]

dataset.insert(loc = 1, column = 'time', value = times)

values  = dataset.values

print_array("Dataset with unwanted columns removed", dataset.head())

#--------------------------------------------------------------------------------
#-- Data encoding ---------------------------------------------------------------
#
# Integer encode day of the year and hour and normalize the columns
#--------------------------------------------------------------------------------

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers

values[:,1] = hour_encoder.fit_transform(values[:,1])    # Encode HOUR as an integer value
values[:,2] = weekday_encoder.fit_transform(values[:,2]) # Encode HOUR as int

values = values.astype('float')

max_time = max(values[:,0])
max_hour = max(values[:,1])
max_wday = max(values[:,2])

new_dataset = DataFrame()
new_dataset['time_sin'] = numpy.sin(2. * numpy.pi * values[:,0].astype('float') / (max_time + 1))
new_dataset['time_cos'] = numpy.cos(2. * numpy.pi * values[:,0].astype('float') / (max_wday + 1))

new_dataset['hour_sin'] = numpy.sin(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))
new_dataset['hour_cos'] = numpy.cos(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))

new_dataset['wday_sin'] = numpy.sin(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))
new_dataset['wday_cos'] = numpy.cos(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))

new_dataset['bikes'] = values[:,3]

plt.plot(numpy.arange(0, len(new_dataset['hour_sin'].values[0:300]), 1), new_dataset['hour_sin'].values[0:300])
plt.axis('off')

plt.savefig("plots/cyclic_encoding_sin.png", bbox_inches="tight")
plt.close()

plt.plot(numpy.arange(0, len(new_dataset['hour_cos'].values[0:300]), 1), new_dataset['hour_cos'].values[0:300])
plt.axis('off')
plt.savefig("plots/cyclic_encoding_cos.png", bbox_inches="tight")
plt.close()

plt.scatter(new_dataset['hour_sin'].values[0:300], new_dataset['hour_cos'].values[0:300], alpha = 0.5)
plt.axis('off')
plt.savefig("plots/cyclic_encoding.png", bbox_inches="tight")
plt.close()

values = new_dataset.values

print_array("Prescaled values NEW NEW", new_dataset.head(60500))

max_bikes = max(values[:,6]) # Maximum number of bikes a station holds
max_cases = max_bikes + 1

values    = values.astype('float32') # Convert al values to floats

print_array("Prescaled values", values)
# print_array("Prescaled values NEW NEW", dataset)

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
scaled = scaler.fit_transform(values)


print_array("Dataset with normalized values", scaled)

#--------------------------------------------------------------------------------
# Generate the columns list for the supervised transformation
#--------------------------------------------------------------------------------

columns = ['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos', 'bikes']

print("COLUMNS", columns)

reframed = series_to_supervised(columns, scaled, n_in, n_out)

to_drop = range(n_in * len(columns) , (1 + n_in) * len(columns) - 1)

print_smth("TODROP", to_drop)

reframed.drop(reframed.columns[to_drop], axis=1, inplace=True)

values = reframed.values

print_array("Reframed dataset without columns that are not going to be predicted", reframed.head())

train_size, test_size, prediction_size = int(len(values) * 0.75) , int(len(values) * 0.2), int(len(values) * 0.05)

train_size      = int(int(train_size / new_batch_size) * new_batch_size) 
test_size       = int(int(test_size / new_batch_size) * new_batch_size) 
prediction_size = int(int(prediction_size / new_batch_size) * new_batch_size) 

# Divide between train and test sets
train, test, prediction = values[0:train_size,:], values[train_size:train_size + test_size, :], values[train_size + test_size:train_size + test_size + prediction_size, :]

output_vals = range(4,7)
print(output_vals)

print_array("TRAIN", train)

train_x, train_y           = train[:,range(0,len(columns) * n_in)], train[:,-1]
test_x, test_y             = test[:,range(0,len(columns) * n_in)], test[:,-1]
prediction_x, prediction_y = prediction[:,range(0,len(columns) * n_in)], prediction[:,-1]

print_array("TRAIN_X ", train_x)
print_array("TRAIN_Y ", train_y)
print_array("TEST_X ", test_x)
print_array("TEST_Y ", test_y)
print_array("PREDICTION_X ", prediction_x)
print_array("PREDICTION_Y ", prediction_y)

# reshape input to be [samples, time_steps, features]
train_x       = train_x.reshape((train_x.shape[0], n_in, len(columns))) # (...,n_in,4)
test_x        = test_x.reshape((test_x.shape[0], n_in, len(columns)))    # (...,n_in,4)
prediction_x  = prediction_x.reshape((prediction_x.shape[0], n_in, len(columns)))    # (...,n_in,4)

print_array("TRAIN_X 2 ", train_x)
print_array("TRAIN_Y 2 ", train_y)
print_array("TEST_X 2 ", test_x)
print_array("TEST_Y 2 ", test_y)
print_array("PREDICTION_X 2 ", prediction_x)
print_array("PREDICTION_Y 2 ", prediction_y)

print(col.blue, "[Dimensions]> ", "Train X ", train_x.shape, "Train Y ", train_y.shape, "Test X ", test_x.shape, "Test Y ", test_y.shape, "Prediction X", prediction_x.shape, "Prediction Y", prediction_y.shape, col.ENDC)

################################################################################
# Neural Network
################################################################################

print("Checking if exists")

file_name = str(lstm_neurons) + "_" + str(new_batch_size) + "_" + str(epochs) + "_" + str(n_in)


# If the model is already trained don't do it again
if os.path.isfile("model/model.h5") == False:

	print("EXISTS")

	if os.path.isdir("/model") == False:
		os.system("mkdir model")
		os.system("chmod 775 model")

	print(col.HEADER + "Neural Network definition" + col.ENDC)

	#--------------------------------------------------------------------------------
	# Network definition
	#--------------------------------------------------------------------------------

	model = create_model(new_batch_size, False)

	print("DIMENSIONS LOCO")
	print(train_x.shape[1], train_x.shape[2])

	list_acc  = []
	list_loss = []
	list_mse  = []

	for i in range(epochs):

		print("Epoch " + str(i+1) + "/" + str(epochs))
		history = model.fit(train_x, train_y, epochs=1, batch_size=new_batch_size, verbose=2, shuffle=False, validation_data = (test_x, test_y))

		list_acc.append(float(history.history['acc'][0]))
		list_loss.append(float(history.history['loss'][0]))
		list_mse.append(float(history.history['mean_squared_error'][0]))

		model.reset_states()

		if i % 10 == 0:
			model.save_weights("model/model.h5")

	print(list_acc)


	if os.path.isdir("/data_gen") == False:
		os.system("mkdir data_gen")
		os.system("chmod 775 data_gen")

	if os.path.isdir("/data_gen/acc") == False:
		os.system("mkdir data_gen/acc")
		os.system("chmod 775 data_gen/acc")

	if os.path.isdir("/data_gen/loss") == False:
		os.system("mkdir data_gen/loss")
		os.system("chmod 775 data_gen/loss")

	if os.path.isdir("/data_gen/mse") == False:
		os.system("mkdir data_gen/mse")
		os.system("chmod 775 data_gen/mse")


	with open('data_gen/acc/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(list_acc)

	with open('data_gen/loss/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(list_loss)
	
	with open('data_gen/mse/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(list_mse)

	min_y = min(history.history['loss'])
	max_y = max(history.history['loss'])

	

	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model/model.h5")

	#plot_model(model, to_file='model/model.png', show_shapes = True)

	w = model.get_weights()

	print("Saved model to disk")


print("Loading model from disk")

# Load trained model from disk
model = create_model(batch_size, True)
model.load_weights("model/model.h5")

#plot_model(model, to_file='model/new_model.png', show_shapes = True)

print_array("TEST_X", prediction_x)

if os.path.isdir("/data_gen/prediction") == False:
		os.system("mkdir data_gen/prediction")
		os.system("chmod 775 data_gen/prediction")


predicted = []

for i in range(len(prediction_x)):

	aux = prediction_x[i]
	aux = aux.reshape(1, prediction_x[i].shape[0], prediction_x[i].shape[1])

	# print_array("Feeding", aux)

	yhat = model.predict(aux)

	predicted.append(yhat[0][0] * max_bikes)


with open('data_gen/prediction/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(predicted)

#prepare_plot('samples', 'bikes', predicted, [], 'YOYOYO')

################################################################################
# Predictron
################################################################################

# Makes future predictions by doing iterations, takes some real initial samples
# makes a prediction and then uses the prediction to predict

print(col.BOLD, "\n\n------------------------------------------------------------------------")
print("Predicting a whole day of availability")
print("------------------------------------------------------------------------\n\n", col.ENDC)

inital_bikes = 9
today        = datetime.datetime.now().timetuple().tm_yday # Current day of the year
weekday      = weekdays[datetime.datetime.today().weekday()]
hour         = "10:00"

hour = hour_encoder.transform([hour])[0]
weekday = weekday_encoder.transform([weekday])[0]

print("Starting prediction at " + str(hour) + " of the day " + str(today) + " which is " + str(weekday) + " with " + str(inital_bikes) + " initial bikes")


# Create the initial array of information with only one time-step and then adding the remaining ones
# --------------------------------------------------------------------------------------------------- 

d = create_array(today, hour, weekday, inital_bikes)

d = scaler.transform([d]) # Scale the values using the same scaler as before

print_smth("Scaled values", d)

# Create the remaining timesteps to later feed the network

if n_in > 1:

	for ts in range(1,n_in):

		aux = numpy.array(numpy.sin(2. * numpy.pi * today / (max_time + 1)))           # Day of the year SIN
		aux = numpy.append(aux,numpy.cos(2. * numpy.pi * today / (max_time + 1)))      # Day of the year COS

		aux = numpy.append(aux,numpy.sin(2. * numpy.pi * (hour+ts) / (max_hour + 1)))  # Hour SIN
		aux = numpy.append(aux,numpy.cos(2. * numpy.pi * (hour+ts) / (max_hour + 1)))  # Hour COS

		aux = numpy.append(aux,numpy.sin(2. * numpy.pi * weekday / (max_wday + 1)))    # Weekday SIN
		aux = numpy.append(aux,numpy.cos(2. * numpy.pi * weekday / (max_wday + 1)))    # Weekday COS
		aux = numpy.append(aux,inital_bikes)

		aux = scaler.transform([aux])

		d = numpy.append(d, aux)

	print_array("Resulting array with all the time-steps", d)

d = d.reshape(1, n_in, len(columns))

print_array("Reshaped array", d)

predicted_bikes = -1

pred = []

def return_og_hour(lista):

# 	print(hour_encoder.inverse_transform([[int(reverse_hour_sin(lista))]])[0][0])

	return hour_encoder.inverse_transform([[int(reverse_hour_sin(lista))]])[0][0]

def reverse_hour_sin(lista):

# 	print("> " + str(numpy.arcsin(lista[2]) * (max_hour+1) / (2 * numpy.pi)))
# 	print("> " + str(math.ceil(numpy.arcsin(lista[2]) * (max_hour+1) / (2 * numpy.pi))))

	return math.ceil(numpy.arcsin(lista[2]) * (max_hour+1) / (2 * numpy.pi))

def reverse_hour_cos(lista):

	# print("> " + str(numpy.arcsin(lista[2]) * (max_hour+1) / (2 * numpy.pi)))

	return math.ceil(numpy.arccos(lista[3]) * (max_hour+1) / (2 * numpy.pi))

def return_og_day(lista):
	return hour_encoder.inverse_transform([[int(reverse_day_sin(lista))]])[0][0]

def reverse_day_sin(lista):
	return math.ceil(numpy.arcsin(lista[0]) * (max_time+1) / (2 * numpy.pi))
	
def og_list(lista):
	
	print(col.green + "Hora " + return_og_hour(lista) + " del dia " + str(reverse_day_sin(lista)) + col.ENDC)

# def unencode_array(array, time_steps):


for i in range(0,5):

	print(col.FAIL + "\n###############################################################################\n" + col.ENDC)

	print_array("Data fed to the model", d)


	predicted_bikes = model.predict(d)[0][0]
	pred.append(predicted_bikes)

	print("Predicted " + str(predicted_bikes))


	print("\t> " + str(predicted_bikes * max_bikes))

	d = d.reshape(n_in * len(columns),) # Flattens the array to the shape  (n_in * 7,) :: (n_in * len(columns),)
	
	print_array("Flattened Data", d)

	print(range((n_in - 1) * len(columns), n_in * len(columns)))

	# Get the newest sample
	newest = d[range((n_in - 1) * len(columns), n_in * len(columns))] # (7,)
	
	print_array("Got the newest sample of the previous data", newest)
	
	newest = scaler.inverse_transform([newest])[0] # Back to the original scale (7,0)

	print_array("Scaled back to the original size the newest sample", newest)
	
	og_list(newest)

	# A partir de la muestra más reciente crear la nueva con un timestep mas de hora
	new_sample = numpy.array(numpy.sin(2. * numpy.pi * today / (max_time + 1)))
	new_sample = numpy.append(new_sample,numpy.cos(2. * numpy.pi * today / (max_time + 1)))

	new_sample = numpy.append(new_sample, numpy.sin((reverse_hour_sin(newest) + 1) * 2 * numpy.pi / (max_hour + 1)))
	new_sample = numpy.append(new_sample, numpy.cos((reverse_hour_cos(newest) + 1) * 2 * numpy.pi / (max_hour + 1)))

	# reverse_hour(new_sample)

	new_sample = numpy.append(new_sample,numpy.sin(2. * numpy.pi * weekday / (max_wday + 1)))
	new_sample = numpy.append(new_sample,numpy.cos(2. * numpy.pi * weekday / (max_wday + 1)))

	new_sample = numpy.append(new_sample,predicted_bikes)

	new_sample = scaler.transform([new_sample])



	# print_array("Newest", newest)
	# print_array("new_sample", new_sample)
	# print_array("Total data", d)

	d = numpy.append(d, new_sample) # Now the array has one more sample at the end

	# print_array("Total data", d)	

	# print(range(len(columns), (n_in+1) * len(columns)))

	d = d[range(len(columns), (n_in+1) * len(columns))] # Remove the oldest sample, the one that is at the beginning
	d = d.reshape(1, n_in, len(columns))  # (1, n_in, 7)
	
	print_array("FInal", d)


print(pred)


if os.path.isdir("/data_gen/predictron") == False:
		os.system("mkdir data_gen/predictron")
		os.system("chmod 775 data_gen/predictron")


with open('data_gen/predictron/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(pred)

prepare_plot('Time', 'Bikes', pred, [], 'predictron')
