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

# Generate same number of columns for the categorization problem as bikes are in this station
def generate_column_array(columns, max_cases) :

	columns = columns

	for i in range(0, max_bikes + 1):
		columns.append(str(i) + '_free_bikes')

	return columns


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
# dataset         = dataset[dataset['station'].isin([stationToRead])]

print(col.HEADER, "> Data from " + dataset['datetime'].iloc[0], " to ", dataset['datetime'].iloc[len(dataset) - 1], col.ENDC)

print_array("Read dataset", dataset.head())
print_array("Read dataset", dataset)

dataset.drop(dataset.columns[[2,5]], axis = 1, inplace = True) # Remove ID of the sation and free docks

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
station_encoder = LabelEncoder() 

values[:,1] = hour_encoder.fit_transform(values[:,1])    # Encode HOUR as an integer value
values[:,2] = weekday_encoder.fit_transform(values[:,2]) # Encode HOUR as int
values[:,3] = station_encoder.fit_transform(values[:,3]) # Encode HOUR as int

max_bikes = max(values[:,3]) # Maximum number of bikes a station holds
max_cases = max_bikes + 1

values    = values.astype('float32') # Convert al values to floats

print_array("Prescaled values", values)

oneHot = to_categorical(values[:,3])

print_array("HEEEEEEEEEEY", values)

values = values[:,:-1]

print_array("HEEEEEEEEEEY", values)

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
scaled = scaler.fit_transform(values)

print_array("Deleted column", scaled)

scaled = numpy.append(scaled, oneHot, axis = 1)

print_array("Dataset with normalized values", scaled)

#--------------------------------------------------------------------------------
# Generate the columns list for the supervised transformation
#--------------------------------------------------------------------------------

columns = generate_column_array(['doy', 'time', 'weekday', 'station'], max_cases)

print("COLUMNS", columns)

reframed = series_to_supervised(columns, scaled, n_in, n_out)

# Drop columns that I don't want to predict, every (t) column except free_bikes(t)
to_drop = range((max_cases + 4) * n_in, (max_cases + 4) * n_in + 4)

reframed.drop(reframed.columns[to_drop], axis=1, inplace=True)

print_smth("PUTO", reframed.columns)

values = reframed.values

print_array("Reframed dataset without columns that are not going to be predicted", reframed.head())

train_size, test_size, prediction_size = int(len(values) * 0.65) , int(len(values) * 0.3), int(len(values) * 0.05)

train_size      = int(int(train_size / new_batch_size) * new_batch_size) 
test_size       = int(int(test_size / new_batch_size) * new_batch_size) 
prediction_size = int(int(prediction_size / new_batch_size) * new_batch_size) 

# Divide between train and test sets
train, test, prediction = values[0:train_size,:], values[train_size:train_size + test_size, :], values[train_size + test_size:train_size + test_size + prediction_size, :]

output_vals = range(n_in * (max_cases + 4),n_in * (max_cases + 4) + n_out * max_cases)
print(output_vals)

train_x, train_y           = train[:,range(0,(max_cases+4) * n_in)], train[:,output_vals]
test_x, test_y             = test[:,range(0,(max_cases+4) * n_in)], test[:,output_vals]
prediction_x, prediction_y = prediction[:,range(0,(max_cases+4) * n_in)], prediction[:,output_vals]

print_array("TRAIN_X ", train_x)
print_array("TRAIN_Y ", train_y)
print_array("TEST_X ", test_x)
print_array("TEST_Y ", test_y)
print_array("PREDICTION_X ", prediction_x)
print_array("PREDICTION_Y ", prediction_y)

# reshape input to be [samples, time_steps, features]
train_x       = train_x.reshape((train_x.shape[0], n_in, max_cases +4)) # (...,n_in,4)
test_x        = test_x.reshape((test_x.shape[0], n_in, max_cases +4))    # (...,n_in,4)
prediction_x  = prediction_x.reshape((prediction_x.shape[0], n_in, max_cases +4))    # (...,n_in,4)

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


# If the model is already trained don't do it again
if os.path.isfile("model/model.h5") == False:



	print(col.HEADER + "Neural Network definition" + col.ENDC)

	#--------------------------------------------------------------------------------
	# Network definition
	#--------------------------------------------------------------------------------

	# lstm_neurons = max_cases * n_out + int(max_cases/2)

	model = Sequential()
	model.add(LSTM(lstm_neurons, batch_input_shape=(new_batch_size, train_x.shape[1], train_x.shape[2]), stateful=False))
	model.add(Dense(max_cases * n_out, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'mse', 'mae'])
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

	# history = model.fit(train_x, train_y, 
	#     epochs = epochs, 
	#     batch_size = batch_size, 
	#     validation_data = (test_x, test_y), 
	#     verbose = 1, 
	#     shuffle = False)

	list_acc  = []
	list_loss = []
	list_mse  = []

	for i in range(epochs):

		print("Epoch " + str(i+1) + "/" + str(epochs))
		history = model.fit(train_x, train_y, epochs=1, batch_size=new_batch_size, verbose=2, shuffle=False)

		list_acc.append(float(history.history['acc'][0]))
		list_loss.append(float(history.history['loss'][0]))
		list_loss.append(float(history.history['mean_squared_error'][0]))

		model.reset_states()

	print(list_acc)

	file_name = str(lstm_neurons) + "_" + str(new_batch_size) + "_" + str(epochs) + "_" + str(n_in)

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

	if os.path.isdir("/model") == False:
		os.system("mkdir model")
		os.system("chmod 775 model")

	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model/model.h5")

	plot_model(model, to_file='model/model.png', show_shapes = True)

	print("Saved model to disk")


# Load trained model from disk
model = Sequential()
model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=True)) # Changed batch size
model.add(Dense(max_cases * n_out, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'mse', 'mae'])

model.load_weights('model/model.h5')

# Redefinition of the neural to change the input shape

# --------------------------------------------------------------------------------
# Predicted data (yhat)
# --------------------------------------------------------------------------------
for i in range(0, len(test_x)):

	auxx = test_x[i].reshape((batch_size, test_x[i].shape[0], test_x[i].shape[1])) # (...,n_in,4)

	yhat = model.predict(auxx, batch_size = batch_size)

	# print("[" + str(i+1) +  "/" + str(len(train_x)) + ']> Expected=' + str(argmax(auxx)) + " Predicted=" + str(argmax(yhat)))

yhat = model.predict(test_x, batch_size = batch_size)

# revert to_categorical
yhat = argmax(yhat, axis = 1)
yhat = yhat.reshape(len(yhat),1)

print_smth("yhat", yhat)
print(yhat.shape)

test_x = test_x.reshape((test_x.shape[0], test_x.shape[2] * n_in))
test_x = test_x[:,[0,1,2]]

inv_yhat = scaler.inverse_transform(test_x)
inv_yhat = concatenate((inv_yhat, yhat), axis=1)

print_smth("inv_yhat before cast", inv_yhat)

#--------------------------------------------------------------------------------
# Real data (inv_yhat)
#--------------------------------------------------------------------------------

yhat = argmax(test_y, axis = 1)
yhat = yhat.reshape(len(yhat),1)

# Real values
inv_y = scaler.inverse_transform(test_x)
inv_y = concatenate((inv_y, yhat), axis=1)

print_smth("inv_y before cast", inv_y)

# Cast the bikes as ints
inv_yhat = inv_yhat[:,3].astype(int)
inv_y    = inv_y[:,3].astype(int) 

print_smth("inv_y after casting to int the bikes", inv_y)
print_smth("inv_yhat after casting to int the bikes", inv_yhat)

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(col.HEADER + '> Test RMSE: %.3f' % rmse + col.ENDC)

calculate_no_errors(inv_y, inv_yhat)

################################################################################
# Plot styling
################################################################################

# prepare_plot('epochs', 'loss', history.history['loss'], history.history['val_loss'], 'loss')
# prepare_plot('epoch', 'accuracy', history.history['acc'], history.history['val_acc'], 'acc')
prepare_plot('samples', 'bikes', inv_yhat, inv_y, 'train')
prepare_plot('samples', 'bikes', inv_y[range(3100,3100 + 500)], inv_yhat[range(3100,3100 + 500)], 'train_zoomed')

################################################################################
# Value predictions
################################################################################

print_array("Prediction IN", prediction_x)

yhat   = model.predict(prediction_x, batch_size = batch_size)

print_array("HEY MIRA ESTO", yhat)

# invert to_categorical
yhat = argmax(yhat, axis = 1)
yhat = yhat.reshape(len(yhat),1)

prediction_x = prediction_x.reshape((prediction_x.shape[0], prediction_x.shape[2] * n_in))
prediction_x = prediction_x[:,[0,1,2]]

inv_yhat = scaler.inverse_transform(prediction_x)
inv_yhat = concatenate((inv_yhat, yhat), axis=1)

print_smth("inv_yhat before cast", inv_yhat)

#--------------------------------------------------------------------------------
# Real data (inv_yhat)
#--------------------------------------------------------------------------------

yhat = argmax(prediction_y, axis = 1)
yhat = yhat.reshape(len(yhat),1)

# Real values
inv_y = scaler.inverse_transform(prediction_x)
inv_y = concatenate((inv_y, yhat), axis=1)

print_smth("inv_y before cast", inv_y)

# Cast the bikes as ints
inv_yhat = inv_yhat[:,3].astype(int)
inv_y    = inv_y[:,3].astype(int) 

print_smth("inv_y after casting to int the bikes", inv_y)
print_smth("inv_yhat after casting to int the bikes", inv_yhat)

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(col.HEADER + '> Test RMSE: %.3f' % rmse + col.ENDC)

calculate_no_errors(inv_y, inv_yhat)

prepare_plot('asa', 'ylabel', inv_y, inv_yhat, 'name')

################################################################################
# Predictron
################################################################################


# Makes future predictions by doing iterations, takes some real initial samples
# makes a prediction and then uses the prediction to predict

print(col.BOLD, "\n\n------------------------------------------------------------------------")
print("Predicting a whole day of availability")
print("------------------------------------------------------------------------\n\n", col.ENDC)

inital_bikes = 17
today   = datetime.datetime.now().timetuple().tm_yday # Current day of the year
weekday = weekdays[datetime.datetime.today().weekday()]
hour    = "00:30"
station_to_predict = "IRALA"


data_predicted = []

data_predicted.append(inital_bikes)


hour = hour_encoder.transform([hour])[0]
weekday = weekday_encoder.transform([weekday])[0]
station_to = station_encoder.transform([weekday])[0]

inital_bikes = to_categorical(inital_bikes, max_cases)

three_main = numpy.array(today)
three_main = numpy.append(three_main, hour)
three_main = numpy.append(three_main, weekday)
three_main = numpy.append(three_main, station_to)

data_in = scaler.transform([three_main])

data_in = numpy.append(data_in, inital_bikes)

data_to_feed = numpy.array([data_in])



# data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)

# print(data_to_feed, data_to_feed.shape

# If there are more than one time-steps create the initial array
if n_in > 1:
    for ts in range(1,n_in):
        three_main[1] += 1
        data_in = scaler.transform([three_main])
        data_in = numpy.append(data_in, inital_bikes)

        data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)


data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +4)) # (...,1,4)

print_array("data_to_feed",  data_to_feed)

auxxx = data_to_feed

for i in range(batch_size - 1):
    # print(i
    data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

data_to_feed = data_to_feed.reshape((batch_size, n_in, max_cases +4)) # (...,1,4)
# print_array("data_to_feed",  data_to_feed)

print(data_to_feed, data_to_feed.shape)


# Generate predictions for 24 hours, as every interval is 5' a whole day it's 288 predictions
# n_horas = 12 * n_in * 11 #+ 1



for i in range(0,290):



    # undo the transformation of the input that is in the shape of [batch_size, n_in, 24], 
    # and get the first 3 columns
    # and inverse transform of the first three columns [doy, time, dow]
    datoa = data_to_feed[0][:,range(0, 4)][n_in - 1]

    data_rescaled = scaler.inverse_transform([datoa]).astype(int)
	
    predicted_bikes =  model.predict(data_to_feed, batch_size = batch_size)
    predicted_bikes = predicted_bikes[0]
	
    print(col.blue, "[" + str(i + 1) + "] Predichas ", argmax(predicted_bikes), " bicis a las ", hour_encoder.inverse_transform(data_rescaled[:,1].astype(int))[0], col.ENDC)

    data_predicted.append(argmax(predicted_bikes))

    data_rescaled[0][1] += 1 # increase hour interval (+5')

    data_in = scaler.transform(data_rescaled)

    data_to_feed = data_to_feed[0]
    data_to_feed = data_to_feed.reshape((1, n_in * (max_cases +4))) # (...,1,4)

    # discard the oldest sample to shift the data
    data_to_feed = data_to_feed[:,range(max_cases+4, n_in * (max_cases+4))]

    bikes = to_categorical(argmax(predicted_bikes), max_cases)

    data_in = numpy.append(data_in, bikes)
    data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)
    data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +4)) # (...,1,4)

    print_array("HEEYEYYEE", data_to_feed)

    for j in range(batch_size - 1):
        print_array("DATA TO FEED", data_to_feed)

        data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

	

print(data_predicted)

prepare_plot('Time', 'Bikes', data_predicted, [], 'predictron')

os.system("touch finished")



