from math import sqrt
import numpy
import pandas
import matplotlib.pyplot as plt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame, read_csv
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras import optimizers
from keras.layers import Dense, LSTM, Activation, Dropout
from datetime import datetime
from keras.models import model_from_json
import datetime
from numpy import argmax
from sklearn.externals import joblib
import sys

################################################################################
# Predictron
################################################################################

lstm_neurons = 40
batch_size   = 100
epochs       = 1
n_in         = 12 * 3
n_out        = 1

is_in_debug = True


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers

hour_encoder.classes_ = numpy.load('hour_encoder.npy')
weekday_encoder.classes_ = numpy.load('weekday_encoder.npy')

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
scaler = joblib.load("scaler.save") 

weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

max_cases = 21

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
        print "", col.yellow
        print description
        print "----------------------------------------------------------------------------", col.ENDC
        print x
        print col.yellow, "----------------------------------------------------------------------------", col.ENDC

# Formatted output
def print_array(description, x):
    
    if is_in_debug == True:
        print "", col.yellow
        print description, " ", x.shape
        print "----------------------------------------------------------------------------", col.ENDC
        print x
        print col.yellow, "----------------------------------------------------------------------------", col.ENDC

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
    # plt.title(texto,color="black", alpha=0.3)
    plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

    plt.savefig("plots/" + name + ".png", bbox_inches="tight")
    plt.close()
    print col.HEADER + ">  Plot saved" + col.ENDC



# Makes future predictions by doing iterations, takes some real initial samples
# makes a prediction and then uses the prediction to predict

print col.BOLD, "\n\n------------------------------------------------------------------------"
print "Predicting a whole day of availability"
print "------------------------------------------------------------------------\n\n", col.ENDC

inital_bikes = 15
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

print data_to_feed, data_to_feed.shape

# data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)

# print data_to_feed, data_to_feed.shape

# If there are more than one time-steps create the initial array
if n_in > 1:
    for ts in range(1,n_in):
        print("Timestep" + str(ts))
        three_main[1] += 1
        data_in = scaler.transform([three_main])
        data_in = numpy.append(data_in, inital_bikes)

        data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)


data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +3)) # (...,1,4)

print_array("data_to_feed",  data_to_feed)

auxxx = data_to_feed

for i in range(batch_size - 1):
    # print i
    data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

data_to_feed = data_to_feed.reshape((batch_size, n_in, max_cases +3)) # (...,1,4)
# print_array("data_to_feed",  data_to_feed)

print data_to_feed, data_to_feed.shape


# Generate predictions for 24 hours, as every interval is 5' a whole day it's 288 predictions

for i in range(0,50):

    print col.FAIL, ">>> Prediction n." + str(i), col.ENDC

    # undo the transformation of the input that is in the shape of [batch_size, n_in, 24], 
    # and get the first 3 columns
    # and inverse transform of the first three columns [doy, time, dow]
    datoa = data_to_feed[0][:,range(0, 3)][0]
    data_rescaled = scaler.inverse_transform([datoa]).astype(int)
    
    print_array("DATOA", datoa)
    print_array("RESCALED", data_rescaled[0])
    

    data_to_feed = data_to_feed.reshape((-1, n_in, max_cases +3)) # [batch_size, n_in, 24]

    print_array("data_to_feed PRE PREDICT", data_to_feed)

    print_array("Pre entrenamiento", data_to_feed)
    print_array("Pre entrenamiento", data_to_feed[0])



    predicted_bikes =  model.predict(data_to_feed, batch_size = batch_size)

    predicted_bikes = predicted_bikes[0]
    # undo encoding of the hour column
    print col.blue, "Predichas ", argmax(predicted_bikes), " bicis a las ", hour_encoder.inverse_transform(data_rescaled[:,1].astype(int))[0], col.ENDC

    data_predicted.append(argmax(predicted_bikes))

    data_rescaled[0][1] += 1 # increase hour interval (+5')
    data_in = scaler.transform(data_rescaled)

    data_to_feed = data_to_feed[0]

    data_to_feed = data_to_feed.reshape((1, n_in * (max_cases +3))) # (...,1,4)

    print_array("DATA TO FEED 1", data_to_feed)

    # discard the oldest sample to shift the data
    data_to_feed = data_to_feed[:,range(max_cases+3, n_in*(max_cases+3))]

    print_array("DATA TO FEED 2", data_to_feed)    

    # print "Deleted ", data_to_feed, data_to_feed.shape

    bikes = to_categorical(argmax(predicted_bikes), max_cases)
    data_in = numpy.append(data_in, bikes)
    data_to_feed = numpy.append(data_to_feed, [data_in], axis = 1)
    data_to_feed = data_to_feed.reshape((data_to_feed.shape[0], n_in, max_cases +3)) # (...,1,4)

    for i in range(batch_size - 1):
        # print i
        data_to_feed = numpy.append(data_to_feed, auxxx, axis = 1)    

    
    # print_array("data_to_feed " + str(i),  data_to_feed)

    # print ">>>>> ", data_to_feed, data_to_feed.shape



print data_predicted

prepare_plot('Time', 'Bikes', data_predicted, [], 'predictron')


