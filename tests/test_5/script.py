###############################################################################################################
#  _                  _       _  _   
# | |                | |     | || |  
# | |_    ___   ___  | |_    | || |_ 
# | __|  / _ \ / __| | __|   |__   _|
# | |_  |  __/ \__ \ | |_       | |   
#  \__|  \___| |___/  \__|      |_|  
#
#                                    
##############################################################################################################          

# One hot de nuevo

#
# Summary
#-------------------------------------------------------------------------------
# Import data and categorize the bikes, change LSTM layers' settings
# Doesn't predict well, it learns the values and repeats the previous interval

################################################################################
# Libraries and Imports
################################################################################

# -- Input Parameters
# --------------------------------------------------------------------------------
# python3 script.py [lstm_neurons] [batch_size] [epochs] [n_in]

from math import sqrt
import math
import numpy

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

save_model_img = False
stationToRead = 'IRALA'
is_in_debug = True
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

# If no inputs are given train with these parameters
if len(sys.argv) > 0:
	lstm_neurons = 200
	batch_size   = 1
	epochs       = 10
	n_in         = 10
	n_out        = 1
	new_batch_size = 5000
else:
	lstm_neurons = int(sys.argv[1]) # 50
	batch_size   = 1
	epochs       = int(sys.argv[3]) # 30
	n_in         = int(sys.argv[4]) # 10
	n_out        = 1
	new_batch_size = int(sys.argv[2]) #1000

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

# Formatted output
def print_smth(description, x):
	
	if is_in_debug == True:
		print("", col.yellow)
		print(description)
		print("----------------------------------------------------------------------------", col.ENDC)
		print(x)
		print(col.yellow, "----------------------------------------------------------------------------", col.ENDC)

# Print an array with a description and its size
def print_array(description, array):
	
	if is_in_debug == True:
		print("", col.yellow)
		print(description, " ", array.shape)
		print("----------------------------------------------------------------------------", col.ENDC)
		print(array)
		print(col.yellow, "----------------------------------------------------------------------------", col.ENDC)

# Stilish the plot without plot ticks, 
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

	plt.xlabel(xlabel, color = 'silver', fontsize = 17)
	plt.ylabel(ylabel, color = 'silver', fontsize = 17)

	lines  = plt.plot(plot_1, label = 'train', color = '#458DE1')

	if len(plot_2) > 0:
		lines += plt.plot(plot_2, label = 'test', color = '#80C797')

	plt.setp(lines, linewidth=2)

	plt.text((len(plot_1) - 1) * 1.005,
		 plot_1[len(plot_1) - 1] + 0.01,
		 "Predicted", color = '#458DE1')

	if len(plot_2) > 0:
		plt.text((len(plot_2) - 1) * 1.005,
		 plot_2[len(plot_2) - 1],
		 "Real", color = '#80C797')

	texto = " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " |  " + str(lstm_neurons) + " LSTM neurons"
	plt.title(texto,color="black", alpha=0.3)
	plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

	plt.savefig("plots/" + name + ".png", bbox_inches="tight")
	plt.close()
	print(col.HEADER + "> Loss plot saved" + col.ENDC)

# Check if the current directory exists, if not create it.
def check_directory(path):
	if os.path.isdir(path) == False:
		os.system("mkdir " + path)
		os.system("chmod 775 " + path)

def save_file_to(directory, fileName, data):

	with open(directory + fileName, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(data)

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
				
################################################################################
# Data preparation
################################################################################

check_directory("plots")
check_directory("data_gen")
check_directory("data_gen/acc")
check_directory("data_gen/loss")
check_directory("data_gen/mse")
check_directory("data_gen/prediction")
check_directory("encoders")

print(col.HEADER + "Data reading and preparation" + col.ENDC)

#--------------------------------------------------------------------------------
# File reading, drop non-relevant columns like station id, station name...
#--------------------------------------------------------------------------------

dataset         = pandas.read_csv('data/Bilbao.txt')
dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 
dataset         = dataset[dataset['station'].isin([stationToRead])]

# dataset         = dataset[dataset['station'].isin(hours_encoder)]

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

misDatos =  dataset[dataset['datetime'].isin([datetime.datetime.now().timetuple().tm_yday])].values

# Generate same number of columns for the categorization problem as bikes are in this station
def generate_column_array(columns, max_bikes) :

	columns = columns

	for i in range(0, max_bikes + 1):
		columns.append(str(i) + '_free_bikes')

	return columns

#--------------------------------------------------------------------------------
#-- Data encoding ---------------------------------------------------------------
#
# First steps:
#  * Hour Integer Encode
#  * Weekday
#--------------------------------------------------------------------------------

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers

values[:,1] = hour_encoder.fit_transform(values[:,1])    # Encode HOUR as an integer value
values[:,2] = weekday_encoder.fit_transform(values[:,2]) # Encode HOUR as int

print_smth("JAVIER", hour_encoder.classes_)
print(type(hour_encoder.classes_))

weekday_encoder.classes_ = numpy.array(weekdays)

hour_encoder.classes_.tofile('encoders/hour_encoder.txt', sep='\n')
weekday_encoder.classes_.tofile('encoders/weekday_encoder.txt', sep='\n')

values = values.astype('float')

oneHot = to_categorical(values[:,3])

print_array("ONE HOT", oneHot)

max_day  = max(values[:,0])
max_hour = max(values[:,1])
max_wday = max(values[:,2])

max_bikes = int(max(values[:,3])) # Maximum number of bikes a station holds


# Create the model, used two times
#  (1) Batch training the model (batch size = specified as input)
#  (2) Making online predictions (batch size = 1)
def create_model(batch_size, statefulness):



	model = Sequential()
	model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=False))
	model.add(Dense(max_bikes + 1, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'mse', 'mae'])
	# model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=statefulness, activation = 'relu'))
	# model.add(Dense(max_bikes + 1, activation = 'sigmoid'))
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['mse', 'acc'])

	return model

new_dataset = DataFrame()
new_dataset['time_sin'] = numpy.sin(2. * numpy.pi * values[:,0].astype('float') / (max_day + 1))
new_dataset['time_cos'] = numpy.cos(2. * numpy.pi * values[:,0].astype('float') / (max_day + 1))
new_dataset['hour_sin'] = numpy.sin(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))
new_dataset['hour_cos'] = numpy.cos(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))
new_dataset['wday_sin'] = numpy.sin(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))
new_dataset['wday_cos'] = numpy.cos(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))
# new_dataset['bikes']    = oneHot #values[:,3]

bikes_data = DataFrame(oneHot)

new_dataset = new_dataset.join(bikes_data)

print_array("FINAL DATASET", new_dataset)

# Plot the columns representing the sine and cosine

plt.plot(numpy.arange(0, len(new_dataset['hour_sin'].values[0:2000]), 1), new_dataset['hour_sin'].values[0:2000])
plt.axis('off')
plt.savefig("plots/cyclic_encoding_sin.png", bbox_inches="tight")
plt.close()

plt.plot(numpy.arange(0, len(new_dataset['hour_cos'].values[0:2000]), 1), new_dataset['hour_cos'].values[0:2000])
plt.axis('off')
plt.savefig("plots/cyclic_encoding_cos.png", bbox_inches="tight")
plt.close()

plt.scatter(new_dataset['hour_sin'].values[0:2000], new_dataset['hour_cos'].values[0:2000], alpha = 0.5)
plt.axes().set_aspect('equal')
plt.axis('off')
plt.savefig("plots/cyclic_encoding.png", bbox_inches="tight")
plt.close()

values = new_dataset.values


values    = values.astype('float32') # Convert al values to floats

print_array("Prescaled values", values)

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
scaled = scaler.fit_transform(values)

print_array("Dataset with normalized values", scaled)

#--------------------------------------------------------------------------------
# Generate the columns list for the supervised transformation
#--------------------------------------------------------------------------------

columns = generate_column_array(['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos'], int(max_bikes))

# columns = ['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos', 'bikes']

print("COLUMNS", columns)

# Transform a time series into a supervised learning problem
reframed = series_to_supervised(columns, scaled, n_in, n_out)

# Select the columns
to_drop = range(n_in * len(columns) , (1 + n_in) * len(columns) - (max_bikes + 1))

print_smth("TODROP", to_drop)

reframed.drop(reframed.columns[to_drop], axis=1, inplace=True)

values = reframed.values

print_array("Reframed dataset without columns that are not going to be predicted", reframed.head())
print(reframed.columns)

# --------------------------------------------------------------------------------------------------------------
# -- Calculate the number of samples for each set
# --------------------------------------------------------------------------------------------------------------

train_size, test_size, prediction_size = int(len(values) * 0.6) , int(len(values) * 0.2), int(len(values) * 0.15)

train_size      = int(int(train_size / new_batch_size) * new_batch_size) 
test_size       = int(int(test_size / new_batch_size) * new_batch_size) 
prediction_size = int(int(prediction_size / new_batch_size) * new_batch_size) 

prediction_size = 1500

print("Train size " + str(train_size) + " Prediction size " + str(prediction_size))

# Divide between train and test sets
train, test, prediction = values[0:train_size,:], values[train_size:train_size + test_size, :], values[train_size + test_size:train_size + test_size + prediction_size, :]

output_vals = range(len(columns) * (n_in), len(reframed.columns))
print("OUTPUT VALS " + str(output_vals))

print_array("TRAIN", train)

train_x, train_y           = train[:,range(0,len(columns) * n_in)], train[:,output_vals]
test_x, test_y             = test[:,range(0,len(columns) * n_in)], test[:,output_vals]
prediction_x, prediction_y = prediction[:,range(0,len(columns) * n_in)], prediction[:,output_vals]

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

# -------------------------------------------------------------------------------
# -- Neural Network--------------------------------------------------------------
#
# Model definition
#--------------------------------------------------------------------------------

print("Checking if exists")

file_name = str(lstm_neurons) + "_" + str(new_batch_size) + "_" + str(epochs) + "_" + str(n_in)

# If the model is already trained don't do it again
if os.path.isfile("model/model.h5") == False:

	# As the model doesn't exists create the folder to save it there
	check_directory("model")

	model = create_model(new_batch_size, True)

	list_acc  = []
	list_loss = []
	list_mse  = []

	for i in range(epochs):

		print("Epoch " + str(i+1) + "/" + str(epochs))
		history = model.fit(train_x, train_y, epochs=1, batch_size=new_batch_size, verbose=2, 
			shuffle=False, validation_data = (test_x, test_y))

		list_acc.append(float(history.history['acc'][0]))
		list_loss.append(float(history.history['loss'][0]))

		# Save every N epochs the model, in case it gets interrupted
		if i % 10 == 0:
			model.save_weights("model/model.h5")

		model.reset_states()

	save_file_to("data_gen/acc/", file_name, list_acc)
	save_file_to("data_gen/loss/", file_name, list_loss)
	save_file_to("data_gen/mse/", file_name, list_mse)

	min_y = min(history.history['loss'])
	max_y = max(history.history['loss'])


	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model/model.h5")

	if save_model_img:
		plot_model(model, to_file='model/model.png', show_shapes = True)

	w = model.get_weights()

	print("Saved model to disk")

print("Loading model from disk")

# Load trained model from disk
model = create_model(batch_size, False)
model.load_weights("model/model.h5")

if save_model_img:
	plot_model(model, to_file='model/new_model.png', show_shapes = True)

check_directory("/data_gen/prediction")

predicted = []

for i in range(len(prediction_x)):

	sample = prediction_x[i]
	sample = sample.reshape(1, prediction_x[i].shape[0], prediction_x[i].shape[1])

	yhat = model.predict(sample)

	print_array("PREPREDICTED", argmax(yhat[0]))

	predicted.append(argmax(yhat[0])) # Rescale back the predictions



prediction_y = [argmax(x) for x in prediction_y] # Rescale back the real data

print_smth("PREDICTION_Y JAVI", prediction_y)
print_smth("PREDICTED JAVI", predicted)

save_file_to('data_gen/prediction/', file_name, predicted)


prepare_plot('samples', 'bikes', predicted, prediction_y, 'prediction')

################################################################################################################################################################
# 
#                            _   _          _                          
#                           | | (_)        | |                         
#  _ __    _ __    ___    __| |  _    ___  | |_   _ __    ___    _ __  
# | '_ \  | '__|  / _ \  / _` | | |  / __| | __| | '__|  / _ \  | '_ \ 
# | |_) | | |    |  __/ | (_| | | | | (__  | |_  | |    | (_) | | | | |
# | .__/  |_|     \___|  \__,_| |_|  \___|  \__| |_|     \___/  |_| |_|
# | |                                                                  
# |_|                                                                  
#
#
################################################################################################################################################################

print("Predictions using made up data")

inital_bikes = 11
today        = datetime.datetime.now().timetuple().tm_yday # Current day of the year
weekday      = weekdays[datetime.datetime.today().weekday()]
hour         = "00:30"

inital_bikes = to_categorical(inital_bikes, max_bikes + 1)

hour = hour_encoder.transform([hour])[0]
weekday = weekday_encoder.transform([weekday])[0]

print("Starting prediction at " + str(hour) + " of the day " + str(today) + " which is " + str(weekday) + " with " + str(inital_bikes) + " initial bikes")


# Every column is split into two: the sine and cosine value.

#        + | +             - | +
# Seno  -------   Coseno  -------
#        - | -		       - | + 

def detectar_cuadrante(seno, coseno):

	cuadrante = -1

	if seno >= 0 and coseno >= 0:
		# print("Primer cuadrante")
		cuadrante = 1
	elif seno >= 0 and coseno <= 0:
		# print("Segundo cuadrante")
		cuadrante = 2
	elif seno < 0 and coseno < 0:
		# print("Tercer cuadrante")
		cuadrante = 3
	elif seno < 0 and coseno >= 0:
		# print("Cuarto cuadrante")
		cuadrante = 4


	return cuadrante

# Create the initial array of information with only one time-step and then adding the remaining ones
# values are returned scaled and reshaped
# --------------------------------------------------------------------------------------------------- 
def create_array(doy, hour, weekday, bikes):
	
	array = numpy.empty(0)

	for ts in range(0,n_in):

		print(">> Day " + str(today) + " MAX DAY " + str(max_day) + " = " + str(numpy.sin(2 * numpy.pi * float(today) / (max_day + 1))))

		aux = numpy.array(numpy.sin(2 * numpy.pi * float(today) / (max_day + 1)))             # Day of the year SIN
		aux = numpy.append(aux, numpy.cos(2 * numpy.pi * float(today) / (max_day + 1)))       # Day of the year COS
		aux = numpy.append(aux, numpy.sin(2 * numpy.pi * float(hour + ts) / (max_hour + 1)))  # Hour SIN
		aux = numpy.append(aux, numpy.cos(2 * numpy.pi * float(hour + ts) / (max_hour + 1)))  # Hour SIN
		aux = numpy.append(aux, numpy.sin(2 * numpy.pi * float(weekday) / (max_wday + 1)))    # Hour SIN
		aux = numpy.append(aux, numpy.cos(2 * numpy.pi * float(weekday) / (max_wday + 1)))    # Hour SIN
		aux = numpy.append(aux,inital_bikes)

		aux = scaler.transform([aux])

		array = numpy.append(array, aux)

		print_array("FINAL ARR", array)

	print_array("Resulting array with all the time-steps", array)


	print_array("Scaled array", array)

	array = array.reshape(1, n_in, len(columns))

	print_array("reshaped array", array)

	return array

d = create_array(today, hour, weekday, inital_bikes)

predicted_bikes = -1

pred = []

#################################################################
#
# PREDICT
#
#################################################################

def get_og_day(list):

	inverso_c = ((max_day + 1) / (2 * numpy.pi)) * numpy.arccos(aux[1])
	inverso_c_360 = ((max_day + 1) / (2 * numpy.pi)) * (2 * numpy.pi - numpy.arccos(aux[1]))


	cuadrante = detectar_cuadrante(aux[0], aux[1])

	correct_day = -1

	if cuadrante == 1 or cuadrante == 2:
		correct_day = inverso_c
	else:
		correct_day = inverso_c_360

	return correct_day

def get_og_hour(list):

	# inverso_s = ((max_hour + 1) / (2 * numpy.pi)) * numpy.arcsin(aux[2])
	inverso_c = ((max_hour + 1) / (2 * numpy.pi)) * numpy.arccos(aux[3])
	inverso_c_360 = ((max_hour + 1) / (2 * numpy.pi)) * (2 * numpy.pi - numpy.arccos(aux[3]))

	cuadrante = detectar_cuadrante(aux[2], aux[3])

	correct_hour = -1

	if cuadrante == 1 or cuadrante == 2:
		correct_hour = inverso_c
	else:
		correct_hour = inverso_c_360

	correct_hour = math.ceil(correct_hour)

	return correct_hour

for time_step in range(0,230):

	predd = model.predict(d)[0]

	print_smth("JAVO", predd)

	predicted_bikes = argmax(predd)

	print_smth("JAVO", predicted_bikes)
	pred.append(predicted_bikes)

	ma_bikes = to_categorical(predicted_bikes, max_bikes + 1)

	

	d = d.reshape(n_in * len(columns),) # Flattens the array to the shape  (n_in * 7,) :: (n_in * len(columns),)
	
	# Get the newest sample
	newest = d[range((n_in - 1) * len(columns), n_in * len(columns))] # (7,)

	########################## DECODE ##########################	

	aux = scaler.inverse_transform([newest])[0]

	print_array("Inverted scaling on new_sample", aux)

	# --------------- Detectar Cuadrante Dia Año ---------------

	correct_day = get_og_day(aux)

	# ----------------------------------------------------------	

	# --------------- Detectar Cuadrante Hora ---------------

	correct_hour = math.ceil(get_og_hour(aux))

	# print("Inverso SIN=" + str(numpy.degrees(numpy.arcsin(aux[2]))) + " inverso COS=" + str(numpy.degrees(numpy.arccos(aux[3]))) )
	# print("Inverso " + str(math.ceil(correct_hour)))		

	print(col.HEADER +  "Predichas " + str(predicted_bikes) + " bicis a las " + str(hour_encoder.inverse_transform([math.ceil(correct_hour)])[0]) + " del dia " + str(int(correct_day)) + col.ENDC)

	# ----------------------------------------------------------	

	########################## ENCODE ##########################

	# A partir de la muestra más reciente crear la nueva con un timestep mas de hora
	new_sample = numpy.array(numpy.sin(2 * numpy.pi * correct_day / (max_day + 1)))                     # SIN Day
	new_sample = numpy.append(new_sample, numpy.cos(2 * numpy.pi * correct_day / (max_day + 1)))        # COS Day
	new_sample = numpy.append(new_sample, numpy.sin(2 * numpy.pi * float(correct_hour + 1) / (max_hour + 1)))   # SIN Time
	new_sample = numpy.append(new_sample, numpy.cos(2 * numpy.pi * float(correct_hour + 1) / (max_hour + 1)))   # COS Time
	new_sample = numpy.append(new_sample,numpy.sin(2 * numpy.pi * float(((max_day + 1) / (2 * numpy.pi)) * numpy.arcsin(aux[4])) / (max_wday + 1)))        # SIN WEEKDAY
	new_sample = numpy.append(new_sample,numpy.cos(2 * numpy.pi * float(((max_day + 1) / (2 * numpy.pi)) * numpy.arcsin(aux[5])) / (max_wday + 1)))        # COS Weekday
	new_sample = numpy.append(new_sample, ma_bikes)

	new_sample = scaler.transform([new_sample])

	d = numpy.append(d, new_sample) # Now the array has one more sample at the end

	if is_in_debug:
		print_array("Total data", d)	

		

	d = d[range(len(columns), (n_in+1) * len(columns))] # Remove the oldest sample, the one that is at the beginning
	d = d.reshape(1, n_in, len(columns))  # (1, n_in, 7)
	
	if is_in_debug:
		print_array("Final", d)


print(pred)


if os.path.isdir("/data_gen/predictron") == False:
		os.system("mkdir data_gen/predictron")
		os.system("chmod 775 data_gen/predictron")


with open('data_gen/predictron/' + file_name, 'w') as file:
		wr = csv.writer(file, delimiter = '\n')
		wr.writerow(pred)

prepare_plot('Time', 'Bikes', pred, misDatos[:,3], 'predictron')
