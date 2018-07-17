# ------------------------------------------------------------------------------
# test_1
# ------------------------------------------------------------------------------
# -- Summary -------------------------------------------------------------------
# -
# - (1) Read datafile into a DataFrame
# - (2) Select data only for the desired station
# - (3) Drop unused columns and only have left the wanted columns
# -	   (3.1) Dropped columns
# -            · Station ID
# -            · Free Docks
# -     (3.2) Remaining columns
# -			  · datetime
# -            · weekday
# -            · free bikes
# - (4) Split datetime into two columns
# -     · day of the year
# -     · time
# - (5) Encode data
# -     (5.1) Integer encode the values (doy, time, weekday)
# -	    (5.2) Cyclic encode the values (doy, time, weekday)
# - (6) Fit the model
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Input parameters ----------------------------------------------------------
# python3 script.py [lstm_neurons] [batch_size] [epochs] [n_in]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Libraries & Imports -------------------------------------------------------

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
from keras.layers import Dense, LSTM, Dropout, Activation
from datetime import datetime
import datetime
from numpy import argmax
import pandas.core.frame
from sklearn.externals import joblib
import os
import csv

os.system("reset") # Clears the screen

# os.system("rm -rf model/")

# ------------------------------------------------------------------------------
# -- Global Parameters & Configuration -----------------------------------------

save_model_img = False
# Prints the array
is_in_debug    = True
# Station to analyze and read the data from
stationToRead  = 'IRALA'
weekdays       = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
list_hours = ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35', '00:40',
 '00:45', '00:50', '00:55', '01:00', '01:05', '01:10', '01:15',
 '01:20', '01:25', '01:30', '01:35', '01:40', '01:45', 
 '01:50', '01:55', '02:00', '02:05', '02:10', '02:15', '02:20',
 '02:25', '02:30', '02:35', '02:40', '02:45', '02:50', '02:55', '03:00', '03:05',
 '03:10', '03:15', '03:20', '03:25', '03:30', '03:35', '03:40', '03:45', '03:50',
 '03:55', '04:00', '04:05', '04:10', '04:15', '04:20', '04:25', '04:30', '04:35',
 '04:40', '04:45', '04:50', '04:55', '05:00', '05:05', '05:10', '05:15', '05:20',
 '05:25', '05:30', '05:35', '05:40', '05:45', '05:50', '05:55', '06:00', '06:05',
 '06:10', '06:15', '06:20', '06:25', '06:30', '06:35', '06:40', '06:45', '06:50',
 '06:55', '07:00', '07:05', '07:10', '07:15', '07:20', '07:25', '07:30',
 '07:35', '07:40', '07:45', '07:50', '07:55', '08:00', '08:05', '08:10', '08:15',
 '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55', '09:00',
 '09:05', '09:10', '09:15', '09:20', '09:25', '09:30', '09:35', '09:40',
 '09:45', '09:50', '09:55', '10:00', '10:05', '10:10', '10:15', '10:20', '10:25',
 '10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00',
 '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35',
 '11:40', '11:45', '11:50', '11:55', '12:00', '12:05', '12:10', '12:15',
 '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55', '13:00',
 '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40',
 '13:45', '13:50', '13:55', '14:00', '14:05', '14:10', '14:15',
 '14:20', '14:25', '14:30', '14:35', '14:40',
 '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15',
 '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55', '16:00',
 '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40',
 '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20',
 '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55',
 '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30',
 '18:35', '18:40', '18:45', '18:50', '18:55', '19:00', '19:05', '19:10',
 '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50',
 '19:55', '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30', '20:35',
 '20:40', '20:45', '20:50', '20:55', '21:00', '21:05', '21:10', '21:15',
 '21:20', '21:25', '21:30', '21:35', '21:40', '21:45', '21:50', '21:55', '22:00',
 '22:05', '22:10', '22:15', '22:20', '22:25', '22:30', '22:35', '22:40', '22:45',
 '22:50', '22:55', '23:00', '23:05', '23:10', '23:15', '23:20', '23:25',
 '23:30', '23:35', '23:40', '23:45', '23:50', '23:55']


lstm_neurons   = int(sys.argv[1]) # 50
batch_size     = 1
epochs         = int(sys.argv[3]) # 30
n_in           = int(sys.argv[4]) # 10
n_out          = int(sys.argv[5]) # 10
new_batch_size = int(sys.argv[2]) * n_in #1000

file_name = str(lstm_neurons) + "_" + str(new_batch_size) + "_" + str(epochs) + "_" + str(n_in)

model_name = "model_" + str(lstm_neurons) + "_neurons_" + str(new_batch_size) + "_batch_" + str(epochs) + "_epochs_" + str(n_in) + "_n_in_" + str(n_out) + "_n_out"

# lstm_neurons = 200
# batch_size   = 1
# epochs       = 50
# n_in         = 50
# n_out        = 1
# new_batch_size = 45000

print(str(lstm_neurons) + " neurons in the LSTM shape, " + str(epochs) + " epochs" + str(n_in) + " previous time-steps and " + str(new_batch_size) + " batch size and" + str(n_out) + " n_out")

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


def one_plot(xlabel, ylabel, plot_1, plot_2, name, dia):

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

	lines  = plt.plot(plot_1, plot_2, label = 'train', color = '#458DE1')

	plt.xticks(dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24], dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24])

	# if len(plot_2) > 0:
	# 	lines += plt.plot(plot_2, label = 'test', color = '#80C797')

	plt.setp(lines, linewidth=2)

	# plt.text((len(plot_2) - 1) * 1.005,
	# 	 plot_1[len(plot_2) - 1] + 0.01,
	# 	 "Predicted", color = '#458DE1')

	# if len(plot_2) > 0:
	# 	plt.text((len(plot_2) - 1) * 1.005,
	# 	 plot_2[len(plot_2) - 1],
	# 	 "Real", color = '#80C797')

	texto = " | Batch size " + str(batch_size) + " | Epochs " + str(epochs) + " |  " + str(lstm_neurons) + " LSTM neurons"
	plt.title(texto,color="black", alpha=0.3)
	plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

	# plt.show()
	plt.savefig("plots/" + name + ".png", bbox_inches="tight")

	plt.close()
	print(col.HEADER + "> Loss plot saved" + col.ENDC)

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

	# plt.show()
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
	

# Create the model, used two times
#  (1) Batch training the model (batch size = specified as input)
#  (2) Making online predictions (batch size = 1)
def create_model(batch_sizee, statefulness):

	model = Sequential()
	# model.add(LSTM(lstm_neurons, batch_input_shape=(batch_sizee, train_x.shape[1], train_x.shape[2]), stateful=statefulness, return_sequences=True))
	model.add(LSTM(lstm_neurons, input_shape=(train_x.shape[1], train_x.shape[2]), stateful=statefulness, return_sequences=True))
	model.add(LSTM(lstm_neurons, return_sequences = True))
	model.add(LSTM(lstm_neurons))
	model.add(Dense(21 * n_out))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['mse', 'acc'])
	

	# model = Sequential()
	# model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=statefulness, return_sequences=True))
	# model.add(LSTM(lstm_neurons))
	# model.add(Dense(max_bikes + 1))
	# model.add(Activation('softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy', 'mse', 'mae'])

	return model

def prepare_sequences(x_train, y_train, window_length):
	windows = []
	windows_y = []
	for i, sequence in enumerate(x_train):
		len_seq = len(sequence)
		for window_start in range(0, len_seq - window_length + 1):
			window_end = window_start + window_length
			window = sequence[window_start:window_end]
			windows.append(window)
			windows_y.append(y_train[i])
	return numpy.array(windows), numpy.array(windows_y)


# Generate same number of columns for the categorization problem as bikes are in this station
def generate_column_array(columns, max_bikes) :

	columns = columns

	for i in range(0, max_bikes + 1):
		columns.append(str(i) + '_free_bikes')

	return columns
				
################################################################################
# Data preparation
################################################################################

check_directory("plots")
check_directory("data_gen")
check_directory("data_gen/acc")
check_directory("data_gen/loss")
check_directory("data_gen/mean_squared_error")
check_directory("data_gen/prediction")
check_directory("encoders")

print(col.HEADER + "Data reading and preparation" + col.ENDC)

#--------------------------------------------------------------------------------
# File reading, drop non-relevant columns like station id, station name...
#--------------------------------------------------------------------------------

dataset         = pandas.read_csv('data/Bilbao.txt')
dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 


print_array("Read dataset", dataset)

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



print_array("Dataset with unwanted columns removed", dataset.head(15))

# df[~df['title'].isin(to_drop)]

dataset = dataset[dataset['time'].isin(list_hours)]
og_dataset = dataset

print_array("Dataset with unwanted columns removed 2", dataset.head(15))

values  = dataset.values


misDatos =  dataset[dataset['datetime'].isin([datetime.datetime.now().timetuple().tm_yday])].values

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


print_smth("HOUR ENCODER", hour_encoder.classes_)
print(len(hour_encoder.classes_))

print_array("JAVO MIRA ESTO", values)




for i in range(0,30):
	print(values[i])


day = -1
hora = -1
counter = 0


for i in range(0, len(values)):

	if values[i][1] == 0:
		day = values[i][0]
		hora = values[i][1]
	else:
		# print("[" + str(day) + "] (1) " + str(hora) + " (2) " + str(values[i][1]))

		if day == values[i][0]:
			
			if values[i][1] != (hora + 1):
				

				element_aux = values[i]
				element_aux[1] = element_aux[1] - 1

				# print("\tInsertado>>>> " + str(element_aux))

				hora = element_aux[1] + 1

				numpy.insert(values, counter, numpy.array(element_aux), 0)

			else:
				hora = values[i][1]
			# print("SET HORA " + str(hora))

# print_smth("HEY", dataset.loc [ dataset [ 'datetime' ] == 273 ].values[:,3])
# one_plot('time', 'bikes', dataset.loc [ dataset [ 'datetime' ] == 273 ].values[:,1], dataset.loc [ dataset [ 'datetime' ] == 273 ].values[:,3], 'usage_septiembre', 273)
# one_plot('time', 'bikes', dataset.loc [ dataset [ 'datetime' ] == 20 ].values[:,1], dataset.loc [ dataset [ 'datetime' ] == 20 ].values[:,3], 'usage_enero', 20)



weekday_encoder.classes_ = numpy.array(weekdays)

hour_encoder.classes_.tofile('encoders/hour_encoder.txt', sep='\n')
weekday_encoder.classes_.tofile('encoders/weekday_encoder.txt', sep='\n')

values = values.astype('float')

oneHot = to_categorical(values[:,3])

max_day  = max(values[:,0])
max_hour = max(values[:,1])
max_wday = max(values[:,2])

max_bikes = int(max(values[:,3])) # Maximum number of bikes a station holds

# +---------+---------+----------+----------+-------------+-------------+-------+
# |         |         |          |          |             |             |       |
# | day_sin | day_cos | time_sin | time_cos | weekday_sin | weekday_cos | bikes |
# |         |         |          |          |             |             |       |
# +---------+---------+----------+----------+-------------+-------------+-------+

new_dataset = DataFrame()
# new_dataset['time_sin'] = numpy.sin(2. * numpy.pi * values[:,0].astype('float') / (max_day + 1))
# new_dataset['time_cos'] = numpy.cos(2. * numpy.pi * values[:,0].astype('float') / (max_day + 1))
new_dataset['hour_sin'] = numpy.sin(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))
new_dataset['hour_cos'] = numpy.cos(2. * numpy.pi * values[:,1].astype('float') / (max_hour + 1))
new_dataset['wday_sin'] = numpy.sin(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))
new_dataset['wday_cos'] = numpy.cos(2. * numpy.pi * values[:,2].astype('float') / (max_wday + 1))
# new_dataset['bikes']    = oneHot # values[:,3]

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

# columns = generate_column_array(['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos'], int(max_bikes))
columns = generate_column_array(['hour_sin', 'hour_cos', 'wday_sin', 'wday_cos'], int(max_bikes))

# columns = ['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos', 'bikes']

print("COLUMNS", columns)

# Transform a time series into a supervised learning problem
reframed = series_to_supervised(columns, scaled, n_in, n_out)


final_drop = []

for i in range(0,n_out):

	position = len(columns) * (n_in + i)
	to_drop = range(position, position + 4)
	print(str(i) + " RANGE " + str(to_drop))
	final_drop.append(reframed.columns[to_drop].tolist())


final_drop = [val for sublist in final_drop for val in sublist]

print_smth("Lista columnas a eliminar", final_drop)

reframed.drop(final_drop, axis=1, inplace=True)

values = reframed.values

print_array("Reframed dataset without columns that are not going to be predicted", reframed.head())
print(reframed.columns)

print("AFTER DROP")
for i in range(0,len(reframed.columns)):
	print(str(i) + " " + reframed.columns[i])

# --------------------------------------------------------------------------------------------------------------
# -- Calculate the number of samples for each set
# --------------------------------------------------------------------------------------------------------------

train_size, test_size, prediction_size = int(len(values) * 0.6) , int(len(values) * 0.2), int(len(values) * 0.15)

train_size      = int(int(train_size / new_batch_size) * new_batch_size) 
test_size       = int(int(test_size / new_batch_size) * new_batch_size) 
prediction_size = int(int(prediction_size / new_batch_size) * new_batch_size) 

# prediction_size = 1500

print("Train size " + str(train_size) + " Prediction size " + str(prediction_size))

# Divide between train and test sets
train, test, prediction = values[0:train_size,:], values[train_size:train_size + test_size, :], values[train_size + test_size:train_size + test_size + prediction_size, :]

output_vals = range(len(columns) * (n_in), values.shape[1])
# print("OUTPUT VALS " + str(output_vals))


train_x, train_y           = train[:,range(0,len(columns) * n_in)], train[:,output_vals]
test_x, test_y             = test[:,range(0,len(columns) * n_in)], test[:,output_vals]
prediction_x, prediction_y = prediction[:,range(0,len(columns) * n_in)], prediction[:,output_vals]

print_array(">>> TRAIN_X 2 ", train_x)
print_array(">>> TRAIN_Y 2 ", train_y)
print_array(">>> TEST_X 2 ", test_x)
print_array(">>> TEST_Y 2 ", test_y)
print_array(">>> PREDICTION_X 2 ", prediction_x)
print_array(">>> PREDICTION_Y 2 ", prediction_y)

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


train_x_aux = []
train_y_aux = []

for i in range(0,int(len(train_x) / n_in)):

	# print("DROP " + str(i * n_in) + "(" + str(range(i*n_in, i*n_in +3)) + ")")

	# print_array("Appending", train_x[i])

	# numpy.append(train_x_aux, train_x[i], axis = 0)
	train_x_aux.append(train_x[i])
	train_y_aux.append(train_y[i])

train_x = train_x_aux
train_y = train_y_aux

test_x_aux = []
test_y_aux = []


for i in range(0,int(len(test_x) / n_in)):

	test_x_aux.append(test_x[i])
	test_y_aux.append(test_y[i])

test_x = test_x_aux
test_y = test_y_aux

prediction_x_aux = []
prediction_y_aux = []

for i in range(0,int(len(prediction_x) / n_in)):

	prediction_x_aux.append(prediction_x[i])
	prediction_y_aux.append(prediction_y[i])

prediction_x = prediction_x_aux
prediction_y = prediction_y_aux

for i in range(0,10):

	print_array(str(i), train_x[i])

train_x = numpy.asarray(train_x)
train_y = numpy.asarray(train_y)

test_x = numpy.asarray(test_x)
test_y = numpy.asarray(test_y)

prediction_x = numpy.asarray(prediction_x)
prediction_y = numpy.asarray(prediction_y)


print(col.blue, "[Dimensions]> ", "Train X ", train_x.shape, "Train Y ", train_y.shape, "Test X ", test_x.shape, "Test Y ", test_y.shape, "Prediction X", prediction_x.shape, "Prediction Y", prediction_y.shape, col.ENDC)

print_array("TRAIN_X", train_x)

print_array("TRAIN_Y", train_y)

# -------------------------------------------------------------------------------
# -- Neural Network--------------------------------------------------------------
#
# Model definition
#--------------------------------------------------------------------------------

print("Checking if exists")



# If the model is already trained don't do it again
if os.path.isfile("model/" + model_name + ".h5") == False:

	# As the model doesn't exists create the folder to save it there
	check_directory("model")

	model = create_model(int(new_batch_size/n_in), False)

	list_acc  = []
	list_loss = []
	list_mse  = []

	max_len = 20

	mean_tr_acc = []
	mean_tr_loss = []


	model.fit(train_x, train_y, batch_size=int(new_batch_size/n_in), epochs=epochs, validation_data=(test_x, test_y), verbose=1)

	# for i in range(epochs):



	# 	print("Epoch " + str(i+1) + "/" + str(epochs))

	# 	history = model.fit(train_x, train_y, epochs=1, batch_size=int(new_batch_size/n_in), verbose=1, validation_data=(test_x, test_y) , shuffle=False)

	# 	print(history.history)
            
	# 	list_acc.append(float(history.history['acc'][0]))
	# 	list_loss.append(float(history.history['loss'][0]))
	# 	list_mse.append(float(history.history['mean_squared_error'][0]))

	# 	# Save every N epochs the model, in case it gets interrupted
	# 	if i % 10 == 0:
	# 		model.save_weights("model/" + model_name + ".h5")
	# 		prepare_plot('epochs', 'accuracy',  list_acc, [], 'provisional_accuracy')

	# 	model.reset_states()

	# save_file_to("data_gen/acc/", file_name, list_acc)
	# save_file_to("data_gen/loss/", file_name, list_loss)
	# save_file_to("data_gen/mean_squared_error/", file_name, list_mse)

	# min_y = min(history.history['loss'])
	# max_y = max(history.history['loss'])



	model_json = model.to_json()
	with open("model/model" + model_name +  ".json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model/" + model_name + ".h5")

	if save_model_img:
		plot_model(model, to_file='model/' + model_name + '.png', show_shapes = True)

	w = model.get_weights()

	print("Saved model to disk")

print("Loading model from disk")

# Load trained model from disk
model = create_model(batch_size, False)
model.load_weights("model/" + model_name + ".h5")

if save_model_img:
	plot_model(model, to_file='model/new_' + model_name + '.png', show_shapes = True)

check_directory("/data_gen/prediction")


prediction_x = prediction_x[2:int(n_in/n_in)+2,:]
prediction_y = prediction_y[2:int(n_in/n_in)+2,:]

print("LONGITU " + str(len(prediction_x)))

mean_te_acc = []

prediction_y = prediction_y.reshape(n_out, 21)

print_array("PREDICTION_X", prediction_x)
print_array("PREDICTION_Y", prediction_y)

prediction_x = prediction_x[0].reshape(1,n_in, 25)

print_array("INPUT TO PREDICT", prediction_x)

predicted = model.predict(prediction_x)

print_array("PREDICTED FINAL 1", predicted)

predicted = predicted.reshape(predicted.shape[1])

predicted = predicted.reshape(n_out, 21)

print_array("PREDICTED FINAL 2", predicted)

predicted = [argmax(x) for x in predicted] # Rescale back the real data

print_smth("PREDICTED FINAL 3", predicted)

valos = [argmax(x) for x in prediction_y] # Rescale back the real data

print_smth("VALOS", valos)

print_smth("LEN VALOS", len(valos))

print_smth("LEN PREDICTED", len(predicted))

prepare_plot('samples', 'bikes', predicted , valos, 'A VER SI VA')

save_file_to("data_gen/prediction/", file_name, predicted)

for i in range(len(prediction_x)):

	# print_array("INPUT X", prediction_x[i])
	# print_array("INPUT X", prediction_x[i].reshape(1,n_in,25))

	preddd = model.predict_on_batch(prediction_x)

	print_smth("PREDICTED", preddd)

	mean_te_acc.append(argmax(preddd[0]))

	print(">> " + str(argmax(preddd[0])))
	model.reset_states()


prepare_plot('samples', 'bikes', mean_te_acc ,valos , 'prediction_using_sample_data')

#################################################################################################################################
#                            _   _          _                          
#                           | | (_)        | |                         
#  _ __    _ __    ___    __| |  _    ___  | |_   _ __    ___    _ __  
# | '_ \  | '__|  / _ \  / _` | | |  / __| | __| | '__|  / _ \  | '_ \ 
# | |_) | | |    |  __/ | (_| | | | | (__  | |_  | |    | (_) | | | | |
# | .__/  |_|     \___|  \__,_| |_|  \___|  \__| |_|     \___/  |_| |_|
# | |                                                                  
# |_|                                                                  
#
#################################################################################################################################

# Get the last n_in samples to predict
today        = datetime.datetime.now().timetuple().tm_yday # Current day of the year
oldest_day_to_search = today - int(n_in/288)

list_days_to_save = []

for i in range(int(n_in/288)):
	list_days_to_save.append(oldest_day_to_search + i)

print_smth("A G UARDAR", list_days_to_save)

# TODO: Tener en cuenta más días
datos = og_dataset[og_dataset['datetime'].isin([list_days_to_save])]









