# Multivariate Time Series Forecasting with LSTMs
#
# Summary
#-------------------------------------------------------------------------------
# Import data and categorize the bikes, change LSTM layers' settings
# Doesn't predict well, it learns the values and repeats the previous interval

################################################################################
# Libraries and Imports
################################################################################

from math import sqrt
import numpy
# import pandas

import matplotlib
matplotlib.use('Agg')
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


################################################################################
# Global Variables
################################################################################

stationToRead = 'IRALA'
is_in_debug = True
weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

lstm_neurons = 50
batch_size   = 1
epochs       = 5
n_in         = 10
n_out        = 1

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

dataset         = pandas.read_csv('Bilbao.txt')
dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] 
dataset         = dataset[dataset['station'].isin([stationToRead])]

print(col.HEADER, "> Data from " + dataset['datetime'].iloc[0], " to ", dataset['datetime'].iloc[len(dataset) - 1], col.ENDC)

print_array("Read dataset", dataset.head())

dataset.drop(dataset.columns[[2,3,5]], axis = 1, inplace = True)

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

numpy.save('hour_encoder.npy', hour_encoder.classes_)
numpy.save('weekday_encoder.npy', weekday_encoder.classes_)

max_bikes = max(values[:,3]) # Maximum number of bikes a station holds
max_cases = max_bikes + 1

values    = values.astype('float32') # Convert al values to floats

print_array("Prescaled values", values)

oneHot = to_categorical(values[:,3])

values = values[:,:-1]

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
scaled = scaler.fit_transform(values)

scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 

print_array("Deleted column", scaled)

scaled = numpy.append(scaled, oneHot, axis = 1)

print_array("Dataset with normalized values", scaled)

#--------------------------------------------------------------------------------
# Generate the columns list for the supervised transformation
#--------------------------------------------------------------------------------

columns = generate_column_array(['doy', 'time', 'weekday'], max_cases)

print("COLUMNS", columns)

reframed = series_to_supervised(columns, scaled, n_in, n_out)

# Drop columns that I don't want to predict, every (t) column except free_bikes(t)
to_drop = range((max_cases + 3) * n_in, (max_cases + 3) * n_in + 3)

reframed.drop(reframed.columns[to_drop], axis=1, inplace=True)

values = reframed.values

print_array("Reframed dataset without columns that are not going to be predicted", reframed.head())

train_size, test_size, prediction_size = int(len(values) * 0.65) , int(len(values) * 0.3), int(len(values) * 0.005)

train_size      = int(int(train_size / batch_size) * batch_size) 
test_size       = int(int(test_size / batch_size) * batch_size) 
prediction_size = int(int(prediction_size / batch_size) * batch_size) 

# Divide between train and test sets
train, test, prediction = values[0:train_size,:], values[train_size:train_size + test_size, :], values[train_size + test_size:train_size + test_size + prediction_size, :]


output_vals = range(n_in * (max_cases + 3),n_in * (max_cases + 3) + n_out * max_cases)
print(output_vals)

train_x, train_y           = train[:,range(0,(max_cases+3) * n_in)], train[:,output_vals]
test_x, test_y             = test[:,range(0,(max_cases+3) * n_in)], test[:,output_vals]

prediction_x, prediction_y = prediction[:,range(0,(max_cases+3) * n_in)], prediction[:,output_vals]

print_array("TRAIN_X ", train_x)
print_array("TRAIN_Y ", train_y)
print_array("TEST_X ", test_x)
print_array("TEST_Y ", test_y)
print_array("PREDICTION_X ", prediction_x)
print_array("PREDICTION_Y ", prediction_y)

# reshape input to be [samples, time_steps, features]
train_x       = train_x.reshape((train_x.shape[0], n_in, max_cases +3)) # (...,n_in,4)
test_x        = test_x.reshape((test_x.shape[0], n_in, max_cases +3))    # (...,n_in,4)
prediction_x  = prediction_x.reshape((prediction_x.shape[0], n_in, max_cases +3))    # (...,n_in,4)

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

print(col.HEADER + "Neural Network definition" + col.ENDC)

#--------------------------------------------------------------------------------
# Network definition
#--------------------------------------------------------------------------------

lstm_neurons = max_cases * n_out + int(max_cases/2)

model = Sequential()
# model.add(Dropout(0.3, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))
model.add(LSTM(lstm_neurons, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2]), stateful=True))

# model.add(Dense(max_cases * n_out + int(max_cases/2), activation='relu'))
model.add(Dense(max_cases * n_out, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'mse', 'mae'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# history = model.fit(train_x, train_y, 
#     epochs = epochs, 
#     batch_size = batch_size, 
#     validation_data = (test_x, test_y), 
#     verbose = 1, 
#     shuffle = False)


# If the model is already trained don't do it again
if os.path.isfile("/home/aholab/javier/model.h5") == False:
    for i in range(epochs):

        print("Epoch " + str(i+1) + "/" + str(epochs))
        history = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()

    min_y = min(history.history['loss'])
    max_y = max(history.history['loss'])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    #plot_model(model, to_file='plots/model.png', show_shapes = True)

    #--------------------------------------------------------------------------------
    # Predicted data (yhat)
    #--------------------------------------------------------------------------------

    for i in range(0, len(test_x)):

    	auxx = test_x[i].reshape((batch_size, test_x[i].shape[0], test_x[i].shape[1])) # (...,n_in,4)

    	yhat = model.predict(auxx, batch_size = batch_size)

    	print("[" + str(i+1) +  "/" + str(len(train_x)) + ']>Expected=' + str(argmax(auxx)) + " Predicted=" + str(argmax(yhat)))

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

    numpy.save("/Users/javierdemartin/Documents/GitHub/neural-bikes/tests/test_3/prediction_x", prediction_x)
    numpy.save("/Users/javierdemartin/Documents/GitHub/neural-bikes/tests/test_3/prediction_y", prediction_y)

    yhat   = model.predict(prediction_x, batch_size = batch_size)


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
hour    = "00:00"


data_predicted = []

data_predicted.append(inital_bikes)


hour = hour_encoder.transform([hour])[0]
weekday = weekday_encoder.transform([weekday])[0]
inital_bikes = to_categorical(inital_bikes, max_cases)

three_main = numpy.array(today)
three_main = numpy.append(three_main, hour)
three_main = numpy.append(three_main, weekday)

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


data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +3)) # (...,1,4)

print_array("data_to_feed",  data_to_feed)

auxxx = data_to_feed

for i in range(batch_size - 1):
    # print(i
    data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

data_to_feed = data_to_feed.reshape((batch_size, n_in, max_cases +3)) # (...,1,4)
# print_array("data_to_feed",  data_to_feed)

print(data_to_feed, data_to_feed.shape)


# Generate predictions for 24 hours, as every interval is 5' a whole day it's 288 predictions
# n_horas = 12 * n_in * 11 #+ 1



for i in range(0,290):


    # undo the transformation of the input that is in the shape of [batch_size, n_in, 24], 
    # and get the first 3 columns
    # and inverse transform of the first three columns [doy, time, dow]
    datoa = data_to_feed[0][:,range(0, 3)][n_in - 1]

    data_rescaled = scaler.inverse_transform([datoa]).astype(int)
    
    predicted_bikes =  model.predict(data_to_feed, batch_size = batch_size)
    predicted_bikes = predicted_bikes[0]
    
    print(col.blue, "[" + str(i + 1) + "] Predichas ", argmax(predicted_bikes), " bicis a las ", hour_encoder.inverse_transform(data_rescaled[:,1].astype(int))[0], col.ENDC)

    data_predicted.append(argmax(predicted_bikes))

    data_rescaled[0][1] += 1 # increase hour interval (+5')

    data_in = scaler.transform(data_rescaled)

    data_to_feed = data_to_feed[0]
    data_to_feed = data_to_feed.reshape((1, n_in * (max_cases +3))) # (...,1,4)

    # discard the oldest sample to shift the data
    data_to_feed = data_to_feed[:,range(max_cases+3, n_in * (max_cases+3))]

    bikes = to_categorical(argmax(predicted_bikes), max_cases)

    data_in = numpy.append(data_in, bikes)
    data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)
    data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +3)) # (...,1,4)

    for i in range(batch_size - 1):
        print_array("DATA TO FEED", data_to_feed)

        data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

    

print(data_predicted)

prepare_plot('Time', 'Bikes', data_predicted, [], 'predictron')

os.system("touch finished")



