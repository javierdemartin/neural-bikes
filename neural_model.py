#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed, GRU
import datetime
from keras.optimizers import SGD
import pickle # Saving MinMaxScaler
from utils import Utils
from Plotter import Plotter
from Data_mgmt import Data_mgmt
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate	
import warnings
import json
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from keras.utils.vis_utils import plot_model
import pandas as pd
from pandas import concat,DataFrame
import pandas.core.frame # read_csv
from datetime import timedelta
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import os
from subprocess import check_output as qx
import sys

from keras import backend

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

train_model  = True
len_day      = 144
lstm_neurons = 50

class Neural_Model:

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
		self.d = Data_mgmt()

		bah = self.utils.stations_from_web(self.city)
		bah.drop(bah.columns[[2,3]], axis=1, inplace=True)
		self.station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))

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

		self.scaler = pickle.load(open(self.dir_path + '/data/' + self.city + "/MinMaxScaler.sav", 'rb'))

		self.configuration = pd.read_csv(self.dir_path + "/config/training_params.csv")
	
		self.batch_size = list(self.configuration.values[:,0])
		self.epochs = list(self.configuration.values[:,1])

		self.list_of_stations = list(self.utils.stations_from_web(self.city).values[:,1])

		print(self.list_of_stations)
		

		print(self.configuration)
		print(self.batch_size)

		self.plot_path = "/plots/" + self.city + "/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"
		self.utils.check_and_create([self.plot_path + "data/" + self.city + "/metrics/"])
		self.utils.check_and_create([self.plot_path])
		self.utils.check_and_create(["/plots/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"])
		self.utils.check_and_create(["/plots/" + self.city + "/"])
		self.utils.check_and_create(["/model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"])
		self.utils.check_and_create(['/data/' +self.city + '/tomorrow/'])
		self.utils.check_and_create(['/plots/' + self.city + '/tomorrow/'])

		self.hour_encoder.classes_, self.weekday_encoder.classes_, self.station_encoder.classes_ = self.load_encoders()
		
		self.model = self.create_model()

	def load_encoders(self):
		return np.load(self.dir_path + '/data/' + self.city + '/encoders/hour_encoder.npy'), np.load(self.dir_path + '/data/' + self.city +'/encoders/weekday_encoder.npy'), np.load(self.dir_path + '/data/' + self.city + '/encoders/station_encoder.npy')
		
	def create_model(self):		

		model = Sequential()

		model.add(LSTM(lstm_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences = True))
		model.add(LSTM(lstm_neurons, return_sequences = True))
		model.add(LSTM(lstm_neurons, return_sequences = True))
		model.add(LSTM(lstm_neurons))
		model.add(Dense(len_day))
		
		model.compile(loss='mean_absolute_error', optimizer='adam', metrics = [rmse, 'mae', 'mape'])

		return model

	def fit_model(self):

		if len(self.batch_size) > 0:

			for time in range(len(self.batch_size)):	
			
				print(self.validation_x)
				print(self.validation_y)

				print("Training for " + str(self.batch_size[0]) + " and " + str(self.epochs[0]) + " epochs.")

				self.utils.check_and_create(["/plots/" + self.city +"/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"])
				self.utils.check_and_create(["/model/" + self.city + "/"+ str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"])			

				note = "Model trained with " + str(self.epochs[0]) + " epochs and batch size of " + str(self.batch_size[0])


				print(self.validation_x.shape)

				if train_model == True:

					history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size[0], epochs=self.epochs[0], validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = False) 
					

					self.model.save(self.dir_path + "/model/" + self.city + "/model.h5")  # creates a HDF5 file 'my_model.h5'
					self.model.save(self.dir_path + "/model/" + self.city + "/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")  # creates a HDF5 file 'my_model.h5'
					
					title_plot = "Training & Validation Loss"
					title_path = "training_loss"
					
					self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, self.dir_path + self.plot_path + title_path, note, "Loss", "Validation Loss")

					title_path = "training_rmse"
					title_plot = "Training & Validation RMSE"

					self.p.two_plot(history.history['rmse'], history.history['val_rmse'], "Epoch", "accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "RMSE", "Validation RMSE")		

					title_path = "training_mae"
					title_plot = "Training & Validation MAE"

					self.p.two_plot(history.history['mae'], history.history['val_mae'], "Epoch", "accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "MAE", "Validation MAE")

					title_path = "training_mape"
					title_plot = "Training & Validation MAE"

					self.p.two_plot(history.history['mape'], history.history['val_mape'], "Epoch", "accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "MAPE", "Validation MAPE")

				else:

					self.model = self.create_model()
					self.model.load_weights(self.dir_path + "/model/" + self.city + "/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")

				'''
				self.utils.print_warn("Predicting " + str(self.test_x.shape[0]) + " samples with the test set")

				predicted_test_set = np.asarray([self.model.predict(self.test_x)])[0] # Predict on test values
				
				for i in range(0, self.test_x.shape[0] - 1):

					self.test_xx = self.test_x[i] # Get i-th sample
					next_day = self.test_x[i+1]   # Get i-th sample

					predicted_test_set_i = np.array([predicted_test_set[i]])
					predicted_test_set_i = predicted_test_set_i.reshape(predicted_test_set_i.shape[1],1)

					inv_yhat = concatenate((self.test_xx[:,: self.test_xx.shape[1] - 1], predicted_test_set_i), axis=1)
					inv_yhat = self.scaler.inverse_transform(inv_yhat)

					station_name = str(self.station_encoder.inverse_transform([int(inv_yhat[0][2])])[0])

					real_data = self.scaler.inverse_transform(self.test_xx)
					next_day = self.scaler.inverse_transform(next_day)

					self.p.two_plot(real_data[:,4].tolist(), list(map(int, inv_yhat[:,4].tolist())), "Time", "Free Bikes", "Prediction for " + station_name, self.dir_path +  self.plot_path + str(i), note, "Real", "Predicted")
				'''

				# If doing iterative training remove the already used values

				print("\a")
				self.batch_size.pop(0)
				self.epochs.pop(0)

	def tomorrow(self, append_to_db = True):


		from influxdb import InfluxDBClient
		client = InfluxDBClient('localhost', '8086', 'root', self.db_password, 'Bicis_' + self.city +'_Prediction')

		p = "..:.5"
		list_hours = ["00:00","00:10","00:20","00:30","00:40","00:50","01:00","01:10","01:20","01:30","01:40","01:50","02:00","02:10","02:20","02:30","02:40","02:50","03:00","03:10","03:20","03:30","03:40","03:50","04:00","04:10","04:20","04:30","04:40","04:50","05:00","05:10","05:20","05:30","05:40","05:50","06:00","06:10","06:20","06:30","06:40","06:50","07:00","07:10","07:20","07:30","07:40","07:50","08:00","08:10","08:20","08:30","08:40","08:50","09:00","09:10","09:20","09:30","09:40","09:50","10:00","10:10","10:20","10:30","10:40","10:50","11:00","11:10","11:20","11:30","11:40","11:50","12:00","12:10","12:20","12:30","12:40","12:50","13:00","13:10","13:20","13:30","13:40","13:50","14:00","14:10","14:20","14:30","14:40","14:50","15:00","15:10","15:20","15:30","15:40","15:50","16:00","16:10","16:20","16:30","16:40","16:50","17:00","17:10","17:20","17:30","17:40","17:50","18:00","18:10","18:20","18:30","18:40","18:50","19:00","19:10","19:20","19:30","19:40","19:50","20:00","20:10","20:20","20:30","20:40","20:50","21:00","21:10","21:20","21:30","21:40","21:50","22:00","22:10","22:20","22:30","22:40","22:50","23:00","23:10","23:20","23:30","23:40","23:50"]

		a = pd.DataFrame(list_hours)
		a = a[~a[0].str.contains(p)]
		list_hours = [i[0] for i in a.values.tolist()]
	
		date = datetime.datetime.today().strftime('%Y/%m/%d')

		yesterday = datetime.datetime.today() - datetime.timedelta(1)
		yesterday = yesterday.strftime('%Y/%m/%d')

		self.model = self.create_model()
		self.model.load_weights(self.dir_path + "/model/" + self.city + "/model.h5")

		import time

		current_time = datetime.datetime.today() 

		for station in self.list_of_stations:

			json_body = []

			print("Predicting for " + station)

			try:

				dataset = np.load(self.dir_path + '/data/' + self.city + '/yesterday/' + str(self.station_dict[station]) + ".npy")

			except (FileNotFoundError, IOError):
				print("Wrong file or file path for " + self.dir_path + '/data/' + self.city + '/yesterday/' + str(self.station_dict[station]) + ".npy")


			if len(dataset.shape) > 2:
				
				p = self.model.predict(dataset)

				p = p.reshape((len_day, 1))
				dataset = dataset.reshape((dataset.shape[1], dataset.shape[2]))
				dataset = self.scaler.inverse_transform(dataset)

				weekday = int(dataset[0][2])

				# Get the correct weekday as a String
				if weekday == 6: weekday = 0
				else: weekday += 1
				
									
				weekday = self.weekday_encoder.inverse_transform([weekday])[0]

				json.loads


				inv_yhat = concatenate((dataset[:,: dataset.shape[1] - 1], p), axis=1)

				predo_vals = [int(i) for i in dataset[:,-1]]

				data = dict(zip(list_hours, predo_vals))

				with open(self.dir_path + '/data/' + self.city + '/today/' + self.station_dict[station] + '.json', 'r') as file:
					jsonToday = json.load(file)
				

				jsonToday = list(jsonToday.values())


				jsonFile = open(self.dir_path + '/data/' + self.city + '/tomorrow/' + self.station_dict[station] + '.json', 'w')
				jsonFile.write(json.dumps(data))

				for i in range(0,len_day):

					current_time_aux = current_time.replace(hour=int(list_hours[i].split(':')[0]), minute=int(list_hours[i].split(':')[1]))

					current_time_aux = current_time_aux.strftime('%Y-%m-%dT%H:%M:%SZ')

					meas = {}
					meas["measurement"] = "bikes"
					meas["tags"] = { "station_name" : station, "station_id": self.station_dict[station]}
					meas["time"] =  current_time_aux
					meas["fields"] = { "value" : predo_vals[i] }
					
					json_body.append(meas)

				
				if append_to_db: client.write_points(json_body)

				self.p.two_plot(dataset[:,-1], jsonToday, "Tiempo", "Bicicletas", str("Prediction for " + station + " for today (" + weekday + ")"), self.dir_path + "/plots/" + self.city + "/tomorrow/" + self.station_dict[station], text = "", line_1 = "Prediction", line_2 = "Real Value")
