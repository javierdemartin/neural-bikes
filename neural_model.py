#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
import datetime
import pickle # Saving MinMaxScaler
from utils import Utils
from Plotter import Plotter
from Data_mgmt import Data_mgmt
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate	
import warnings
import json
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# from keras.utils.vis_utils import plot_model
import pandas as pd
import pandas.core.frame
import os
import sys
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras import backend
from influxdb import InfluxDBClient
		
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class Neural_Model:

	db_ip = "192.168.86.99"
	len_day = 144	
	n_out = len_day
	train_model  = True
	len_day      = 144
	lstm_neurons = 40
	city = ""
	hour_encoder    = LabelEncoder()
	weekday_encoder = LabelEncoder()
	station_encoder = LabelEncoder()   
	station_dict = {}

	def __init__(self):

		self.db_password = sys.argv[2]
		self.dir_path = os.path.dirname(os.path.realpath(__file__))

		self.city = sys.argv[1]
		self.utils = Utils(self.city)
		self.p     = Plotter()
		self.d = Data_mgmt(city=self.city)
		f = open(self.dir_path + "/data/" + self.city +  '/Maximums.pkl','rb')
		self.maximumStations = pickle.load(f)
		f.close()

		with open(self.dir_path + '/config/config.json', 'r') as j:
			configs = json.loads(j.read())

		self.generated_columns = configs['data']['generated_columns']

		self.n_days_in = configs['parameters']['lookback_days']
		self.n_in  = self.len_day * self.n_days_in# Number of previous samples used to feed the Neural Network

		self.model = Sequential()

		bah = self.utils.stations_from_web(self.city)
		bah.drop(bah.columns[[2,3]], axis=1, inplace=True)
		self.station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))

		self.scaler = pickle.load(open(self.dir_path + '/data/' + self.city + "/MinMaxScaler.sav", 'rb'))

		self.batch_size = configs['training'][self.city]['batch_size'] 
		self.epochs = configs['training'][self.city]['epochs']

		self.list_of_stations = list(self.utils.stations_from_web(self.city).values[:,1])

		self.plot_path = "/plots/" + self.city + "/" + str(self.epochs) + "_" + str(self.batch_size) + "/"
		self.utils.check_and_create([
			self.plot_path + "data/" + self.city + "/metrics/",
			self.plot_path,
			"/plots/" + self.city + "/",
			'/data/' +self.city + '/tomorrow/',
			'/plots/' + self.city + '/tomorrow/'])

		self.hour_encoder.classes_, self.weekday_encoder.classes_, self.station_encoder.classes_ = self.load_encoders()
		

	def load_encoders(self):
		return np.load(self.dir_path + '/data/' + self.city + '/encoders/hour_encoder.npy'), np.load(self.dir_path + '/data/' + self.city +'/encoders/weekday_encoder.npy'), np.load(self.dir_path + '/data/' + self.city + '/encoders/station_encoder.npy')
		
	def create_model(self):		

		with open(self.dir_path + '/config/config.json', 'r') as j:
			configs = json.loads(j.read())

		model = Sequential()

		for layer in configs['model'][self.city]['layers']:

			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq, activation=activation))
			if layer['type'] == 'dropout':
				model.add(Dropout(dropout_rate))
			if layer['type'] == 'flatten':
				model.add(Flatten())

		model.compile(loss=configs['model'][self.city]['loss'], optimizer=configs['model'][self.city]['optimizer'], metrics=configs['model'][self.city]['metrics'])

		print(model.summary())

# 		plot_model(model, to_file=self.dir_path + "/model/" + self.city + "/model.png", show_shapes=True, show_layer_names=True)

		return model


	def fit_model(self):

		self.train_x = np.load(self.dir_path + '/data/' + self.city + '/train_x.npy')
		self.train_y = np.load(self.dir_path +'/data/' + self.city + '/train_y.npy')

		self.test_x = np.load(self.dir_path + '/data/' + self.city + '/test_x.npy')
		self.test_y = np.load(self.dir_path + '/data/' + self.city + '/test_y.npy')

		self.validation_x = np.load(self.dir_path + '/data/' + self.city + '/validation_x.npy')
		self.validation_y = np.load(self.dir_path + '/data/' + self.city + '/validation_y.npy')

		print("Train X " + str(self.train_x.shape))
		print("Train Y " + str(self.train_y.shape))
		print("Test X " + str(self.test_x.shape))
		print("Test Y " + str(self.test_y.shape))
		print("Validation X " + str(self.validation_x.shape))
		print("Validation Y " + str(self.validation_y.shape))
		
		self.model = self.create_model()
				
		if self.batch_size == 'half':
			current_batch_size = int(int(self.train_x.shape[0]) / 2)
		elif self.batch_size == 'third':
			current_batch_size = int(int(self.train_x.shape[0]) / 3)
		elif self.batch_size == 'fourth':
			current_batch_size = int(self.train_x.shape[0] / 4)
		else:
			current_batch_size = self.batch_size
			
		print("Training for " + str(current_batch_size) + " and " + str(self.epochs) + " epochs.")
		
		self.utils.check_and_create(["/plots/" + self.city +"/" + str(self.epochs) + "_" + str(current_batch_size) + "/"])
		self.utils.check_and_create(["/model/" + self.city + "/"+ str(self.epochs) + "_" + str(current_batch_size) + "/"])			

		note = str(self.epochs) + " epochs and batch size of " + str(self.batch_size)

		with open(self.dir_path + '/config/config.json', 'r') as j:
			configs = json.loads(j.read())
		
		cb_list = []

		for layer in configs['model'][self.city]['callbacks']:

			mode = layer['mode'] if 'mode' in layer else None
			monitor = layer['monitor'] if 'monitor' in layer else None
			patience = layer['patience'] if 'patience' in layer else None

			if layer['type'] == 'early_stopping':
				cb = EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience)
				cb_list.append(cb)
			if layer['type'] == 'model_checkpoint':
				cb = ModelCheckpoint(self.dir_path + "/model/" + self.city + "/model.h5", monitor=monitor, mode=mode, save_best_only=True)
				cb_list.append(cb)
		
		history = self.model.fit(self.train_x, self.train_y, batch_size=current_batch_size, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = False, callbacks = cb_list) 
		
		self.model.save(self.dir_path + "/model/" + self.city + "/model.h5")
		self.model.save(self.dir_path + "/model/" + self.city + "/" + str(self.epochs) + "_" + str(current_batch_size) + "/model.h5")  # creates a HDF5 file 'my_model.h5'
		
		title_plot = "Training & Validation Acc"
		title_path = "training_acc"
		self.p.two_plot(history.history['acc'], history.history['val_acc'], "Epoch", "Accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "Accuracy", "Validation Accuracy")

		title_plot = "Training & Validation Loss"
		title_path = "training_loss"
		self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, self.dir_path + self.plot_path + title_path, note, "Loss", "Validation Loss")

		title_path = "training_mae"
		title_plot = "Training & Validation MSE"
		self.p.two_plot(history.history['mean_squared_error'], history.history['val_mean_squared_error'], "Epoch", "MSE", title_plot, self.dir_path + self.plot_path + title_path, note, "MAE", "Validation MAE")		

		# title_path = "training_mse"
		# title_plot = "Training & Validation MSE"
		# self.p.two_plot(history.history['mean_squared_error'], history.history['val_mean_squared_error'], "Epoch", "accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "MSE", "Validation MSE")		

		
	def rmse(predictions, targets):
		return np.sqrt(np.mean((predictions-targets)**2))
		
	def test_models_score(self):
	
		self.train_x = np.load(self.dir_path + '/data/' + self.city + '/train_x.npy')
		self.train_y = np.load(self.dir_path +'/data/' + self.city + '/train_y.npy')

		self.test_x = np.load(self.dir_path + '/data/' + self.city + '/test_x.npy')
		self.test_y = np.load(self.dir_path + '/data/' + self.city + '/test_y.npy')

		self.validation_x = np.load(self.dir_path + '/data/' + self.city + '/validation_x.npy')
		self.validation_y = np.load(self.dir_path + '/data/' + self.city + '/validation_y.npy')

		self.model = self.create_model()
		self.model.load_weights(self.dir_path + "/model/" + self.city + "/model.h5")
	
		# Evaluate the model's training
		scores = self.model.evaluate(self.test_x, self.test_y, verbose=1)

		print(scores)
		print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
		
		p = "..:.5"
		list_hours = ["00:00", "00:10", "00:20", "00:30", "00:40", "00:50", "01:00", "01:10", "01:20", "01:30", "01:40", "01:50", "02:00", "02:10", "02:20", "02:30", "02:40", "02:50", "03:00", "03:10", "03:20", "03:30", "03:40", "03:50", "04:00", "04:10", "04:20", "04:30", "04:40", "04:50", "05:00", "05:10", "05:20", "05:30", "05:40", "05:50", "06:00", "06:10", "06:20", "06:30", "06:40", "06:50", "07:00", "07:10", "07:20", "07:30", "07:40", "07:50", "08:00", "08:10", "08:20", "08:30", "08:40", "08:50", "09:00", "09:10", "09:20", "09:30", "09:40", "09:50", "10:00", "10:10", "10:20", "10:30", "10:40", "10:50", "11:00", "11:10", "11:20", "11:30", "11:40", "11:50", "12:00", "12:10", "12:20", "12:30", "12:40", "12:50", "13:00", "13:10", "13:20", "13:30", "13:40", "13:50", "14:00", "14:10", "14:20", "14:30", "14:40", "14:50", "15:00", "15:10", "15:20", "15:30", "15:40", "15:50", "16:00", "16:10", "16:20", "16:30", "16:40", "16:50", "17:00", "17:10", "17:20", "17:30", "17:40", "17:50", "18:00", "18:10", "18:20", "18:30", "18:40", "18:50", "19:00", "19:10", "19:20", "19:30", "19:40", "19:50", "20:00", "20:10", "20:20", "20:30", "20:40", "20:50", "21:00", "21:10", "21:20", "21:30", "21:40", "21:50", "22:00", "22:10", "22:20", "22:30", "22:40", "22:50", "23:00", "23:10", "23:20", "23:30", "23:40", "23:50"]

		a = pd.DataFrame(list_hours)
		a = a[~a[0].str.contains(p)]
		list_hours = [i[0] for i in a.values.tolist()]
	
		yesterday = datetime.datetime.today() - datetime.timedelta(1)
		yesterday = yesterday.strftime('%Y/%m/%d')

		# Load the dictionary that holds the maximum values per station name
		f = open(self.dir_path +"/data/" + self.city +  "/Maximums.pkl", 'rb')
		self.maximumBikesInStation = pickle.load(f)
		f.close()
		
		self.utils.check_and_create(["/plots/" + self.city + "/test/"])
		
		for i,(X,y) in enumerate(zip(self.test_x, self.test_y)):

			X_r = X.reshape(1,X.shape[0], X.shape[1])
			y_hat = self.model.predict(X_r)
			X_rescaled = self.scaler.inverse_transform(X)
			X_rescaled_latest = X_rescaled[self.len_day * (self.n_days_in - 1):]
			
			# Get the latest day, input is from multiple days and the generated output
			# is only for one day
			X = X[self.len_day * (self.n_days_in - 1):]
			
			# Put the predicted samples in the OG dataset to rescale back
			X[:,-1] = y_hat
			
			y_rescaled = self.scaler.inverse_transform(X)
			
			real_vals = X_rescaled_latest[:,-1]

			predo_vals = y_rescaled[:,-1]

			predo_vals = [int(i) for i in predo_vals]

			title = ""

			if "weekday" in self.generated_columns:
				weekday_index = self.generated_columns.index("station_name")
				predicted_station = self.station_encoder.inverse_transform([int(y_rescaled[:,weekday_index][0])])[0]
				title += predicted_station + " "

				if predicted_station in self.maximumBikesInStation:
					maxVal = self.maximumBikesInStation[predicted_station]
				else:
					continue	

				# Undo the percentage scaling and restore using the maximum value each station holds
				predo_vals = [i * (maxVal / 100) for i in predo_vals]
				real_vals = [i * (maxVal / 100) for i in real_vals]

			title += "%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100) #"RMSE " + str(rmse(real_vals, predo_vals))

			if np.isnan(predo_vals).any():
				print("Error, predicted NaN values")
				continue
			
			self.p.two_plot(real_vals, predo_vals, "Tiempo", "Bicicletas", title, self.dir_path + "/plots/" + self.city + "/test/" + str(i), text = "", line_1 = "Real", line_2 = "Prediction")		

	def tomorrow(self, data, append_to_db = False):
		
		self.model = self.create_model()
		self.model.load_weights(self.dir_path + "/model/" + self.city + "/model.h5")
		
		client = InfluxDBClient(self.db_ip, '8086', 'root', self.db_password, 'Bicis_' + self.city +'_Prediction')

		p = "..:.5"
		list_hours = ["00:00", "00:10", "00:20", "00:30", "00:40", "00:50", "01:00", "01:10", "01:20", "01:30", "01:40", "01:50", "02:00", "02:10", "02:20", "02:30", "02:40", "02:50", "03:00", "03:10", "03:20", "03:30", "03:40", "03:50", "04:00", "04:10", "04:20", "04:30", "04:40", "04:50", "05:00", "05:10", "05:20", "05:30", "05:40", "05:50", "06:00", "06:10", "06:20", "06:30", "06:40", "06:50", "07:00", "07:10", "07:20", "07:30", "07:40", "07:50", "08:00", "08:10", "08:20", "08:30", "08:40", "08:50", "09:00", "09:10", "09:20", "09:30", "09:40", "09:50", "10:00", "10:10", "10:20", "10:30", "10:40", "10:50", "11:00", "11:10", "11:20", "11:30", "11:40", "11:50", "12:00", "12:10", "12:20", "12:30", "12:40", "12:50", "13:00", "13:10", "13:20", "13:30", "13:40", "13:50", "14:00", "14:10", "14:20", "14:30", "14:40", "14:50", "15:00", "15:10", "15:20", "15:30", "15:40", "15:50", "16:00", "16:10", "16:20", "16:30", "16:40", "16:50", "17:00", "17:10", "17:20", "17:30", "17:40", "17:50", "18:00", "18:10", "18:20", "18:30", "18:40", "18:50", "19:00", "19:10", "19:20", "19:30", "19:40", "19:50", "20:00", "20:10", "20:20", "20:30", "20:40", "20:50", "21:00", "21:10", "21:20", "21:30", "21:40", "21:50", "22:00", "22:10", "22:20", "22:30", "22:40", "22:50", "23:00", "23:10", "23:20", "23:30", "23:40", "23:50"]

		a = pd.DataFrame(list_hours)
		a = a[~a[0].str.contains(p)]
		list_hours = [i[0] for i in a.values.tolist()]
	
		yesterday = datetime.datetime.today() - datetime.timedelta(1)
		yesterday = yesterday.strftime('%Y/%m/%d')

		current_time = datetime.datetime.today()

		for (stationName, dataToPredict) in data.items():

			json_body = []

			if dataToPredict.shape[1] < self.n_in: continue

			p = self.model.predict(dataToPredict)

			p = p.reshape((self.n_out, 1))
			dataToPredict = dataToPredict.reshape((dataToPredict.shape[1], dataToPredict.shape[2]))
			dataToPredict = self.scaler.inverse_transform(dataToPredict)
							
			# Get the last day
			dataToPredict = dataToPredict[self.len_day * (self.n_days_in - 1):]

			inv_yhat = concatenate((dataToPredict[:,: dataToPredict.shape[1] - 1], p), axis=1)

			predo_vals = [int(i) for i in dataToPredict[:,-1]]

			if stationName in self.maximumStations:
				maxVal = self.maximumStations[stationName]

				predo_vals = [int(i * (maxVal / 100)) for i in predo_vals]

			data = dict(zip(list_hours, predo_vals))

			for i in range(0,self.len_day):

				current_time_aux = current_time.replace(hour=int(list_hours[i].split(':')[0]), minute=int(list_hours[i].split(':')[1]))

				current_time_aux = current_time_aux.strftime('%Y-%m-%dT%H:%M:%SZ')

				meas = {}
				meas["measurement"] = "bikes"
				meas["tags"] = { "station_name" : stationName, "station_id": self.station_dict[stationName]}
				meas["time"] =  current_time_aux
				meas["fields"] = { "value" : predo_vals[i] }
				
				json_body.append(meas)

			if append_to_db: 
				client.write_points(json_body)
			else: 
				
				weekday_index = self.generated_columns.index("weekday")
				weekday = int(dataToPredict[-1][weekday_index]) + self.n_days_in - 1
				
				# Get the correct weekday as a String
				if weekday >= 6: 
				    weekday = 0
				else: 
				    weekday += 1
								
				weekday = self.weekday_encoder.inverse_transform([weekday])[0]
				self.p.two_plot(dataToPredict[:,-1], dataToPredict[:,-1], "Tiempo", "Bicicletas", str("Prediction for " + stationName + " for today (" + weekday + ")"), self.dir_path + "/plots/" + self.city + "/tomorrow/" + self.station_dict[stationName], text = "", line_1 = "Prediction", line_2 = "Real Value")
				