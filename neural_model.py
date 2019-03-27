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

from keras import backend

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


train_model  = False
len_day      = 144
lstm_neurons = 50

class Neural_Model:

	def __init__(self):

		self.dir_path = os.path.dirname(os.path.realpath(__file__))

		self.utils = Utils()
		self.p     = Plotter()
		self.d = Data_mgmt()

		self.hour_encoder    = LabelEncoder()
		self.weekday_encoder = LabelEncoder()
		self.station_encoder = LabelEncoder()

		self.train_x = np.load(self.dir_path + '/data/train_x.npy')
		self.train_y = np.load(self.dir_path +'/data/train_y.npy')

		self.test_x = np.load(self.dir_path + '/data/test_x.npy')
		self.test_y = np.load(self.dir_path + '/data/test_y.npy')

		self.validation_x = np.load(self.dir_path + '/data/validation_x.npy')
		self.validation_y = np.load(self.dir_path + '/data/validation_y.npy')

		print("Train X " + str(self.train_x.shape))
		print("Train Y " + str(self.train_y.shape))
		print("Test X " + str(self.test_x.shape))
		print("Test Y " + str(self.test_y.shape))
		print("Validation X " + str(self.validation_x.shape))
		print("Validation Y " + str(self.validation_y.shape))

		self.scaler = pickle.load(open(self.dir_path + "/MinMaxScaler.sav", 'rb'))
	
		self.batch_size = self.utils.read("batch_size")	
		self.epochs = self.utils.read("epochs")	

		self.plot_path = "/plots/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"
		self.utils.check_and_create(self.plot_path + "data/metrics/")
		self.utils.check_and_create(self.plot_path)
		self.utils.check_and_create("/plots/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/")
		self.utils.check_and_create("/model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/")
		self.utils.check_and_create('/data/tomorrow/')
		self.utils.check_and_create('/plots/tomorrow/')

		self.hour_encoder.classes_, self.weekday_encoder.classes_, self.station_encoder.classes_ = self.load_encoders()
		self.model = self.create_model()

		self.utils.append_tutorial_title("Neural Network Training")
		self.utils.append_tutorial_text("![Model Shape](model/model.png)\n")
		self.utils.append_tutorial_text("* " + str(self.epochs) + " epochs")
		self.utils.append_tutorial_text("* " + str(self.batch_size) + " batch size\n")

	def load_encoders(self):
		return np.load(self.dir_path + '/debug/encoders/hour_encoder.npy'), np.load(self.dir_path + '/debug/encoders/weekday_encoder.npy'), np.load(self.dir_path + '/debug/encoders/station_encoder.npy')
		
	def create_model(self):		

		model = Sequential()

		model.add(GRU(lstm_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences = True))
		model.add(GRU(lstm_neurons, return_sequences = True))
		model.add(GRU(lstm_neurons, return_sequences = True))
		model.add(GRU(lstm_neurons))
		model.add(Dense(len_day))
		
# 		model.add(LSTM(lstm_neurons * 2, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences = True))
# 		model.add(LSTM(lstm_neurons))
# 		model.add(Dense(len_day))
		
		model.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['acc', rmse])

		return model

	def fit_model(self):

		if len(self.batch_size) > 0:

			for time in range(len(self.batch_size)):	

				self.utils.check_and_create("/plots/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/")
				self.utils.check_and_create("/model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/")			

				note = "Model trained with " + str(self.epochs[0]) + " epochs and batch size of " + str(self.batch_size[0])
				

				if train_model == True:

					history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size[0], epochs=self.epochs[0], validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = False) 
					

					self.model.save(self.dir_path + "/model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")  # creates a HDF5 file 'my_model.h5'
					self.model.save(self.dir_path + "/model/model.h5")  # creates a HDF5 file 'my_model.h5'
				
					# title_plot = "Training & Validation Loss"
					# title_path = "training_loss"
					
					# self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, self.dir_path + self.plot_path + title_path, note, "Loss", "Validation Loss")

					title_plot = "Training & Validation Accuracy"
					title_path = "training_acc"

					self.p.two_plot(history.history['acc'], history.history['val_acc'], "Epoch", "accuracy", title_plot, self.dir_path  + self.plot_path + title_path, note, "Accuracy", "Validation Accuracy")		

					title_path = "training_rrmse"
					title_plot = "Training & Validation RMSE"

					self.p.two_plot(history.history['rmse'], history.history['val_rmse'], "Epoch", "accuracy", title_plot, self.dir_path + self.plot_path + title_path, note, "RMSE", "Validation RMSE")		

					self.utils.save_array_txt(self.dir_path + self.plot_path + "data/metrics/acc", history.history['acc'])
					self.utils.save_array_txt(self.dir_path + self.plot_path + "data/metrics/val_acc", history.history['val_acc'])
					self.utils.save_array_txt(self.dir_path +  self.plot_path + "data/metrics/loss", history.history['loss'])
					self.utils.save_array_txt(self.dir_path +  self.plot_path + "data/metrics/val_loss", history.history['val_loss'])

					self.utils.append_tutorial_text("![Training Acc](plots/training_acc.png)\n")
					self.utils.append_tutorial_text("![Training Loss](plots/training_loss.png)\n")
					self.utils.append_tutorial_text("![Training MAPE](plots/training_mape.png)\n")
					self.utils.append_tutorial_text("![Training MSE](plots/training_rmse.png)\n")

				else:

					self.model = self.create_model()
					self.model.load_weights(self.dir_path + "/model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")

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

				self.utils.append_tutorial_text("More prediction samples in [plots/](https://github.com/javierdemartin/neural-bikes/tree/master/plots).")

				# If doing iterative training remove the already used values
				self.batch_size.pop(0)
				self.epochs.pop(0)

	def tomorrow(self):

		# Copy the latest data from the text file
		os.system("cp /Users/javierdemartin/Documents/bicis/data/Bilbao.txt " + self.dir_path + "/data/")

		list_hours = self.utils.read_csv_as_list(self.dir_path + '/debug/utils/list_hours')
	
		date = datetime.datetime.today().strftime('%Y/%m/%d')

		yesterday = datetime.datetime.today() - datetime.timedelta(1)
		yesterday = yesterday.strftime('%Y/%m/%d')

		self.model = self.create_model()
		self.model.load_weights(self.dir_path + "/model/model.h5")

		self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

		f = os.popen("tail -n " + str(len(self.list_of_stations) * len_day* 2) + " " + self.dir_path + "/data/Bilbao.txt")

		out = f.read().splitlines()
		f.close()

		# Ahora out solo tiene los datos de hoy
		with open(self.dir_path + '/tomorrow.txt', 'w') as f:
			for item in out:
				f.write("%s\n" % item)

		out = pandas.read_csv(self.dir_path + '/tomorrow.txt')
		out.columns = ['datetime', 'weekday', 'id', 'station', 'free_docks', 'free_bikes'] # Insert correct column names

		out = out[out.datetime.str.contains(yesterday)]

		for station in self.list_of_stations:

			try:

				dataset = np.load(self.dir_path + '/debug/yesterday/' + str(station) + ".npy")

				today_data = np.load(self.dir_path + '/debug/today/' + str(station) + ".npy")

				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path for " + self.dir_path + '/data/tomorrow/' + station + '.json')

			a = out[out['station'].isin([station])]['free_bikes'].values

			if a.shape[0] > 0:
				
				p = self.model.predict(dataset)

				p = p.reshape((len_day, 1))
				dataset = dataset.reshape((dataset.shape[1], dataset.shape[2]))
				dataset = self.scaler.inverse_transform(dataset)

				weekday = int(dataset[0][2])

				# Get the correct weekday as a String
				if weekday is 6: weekday = 0
				else: weekday += 1
					
				weekday = self.weekday_encoder.inverse_transform([weekday])[0]

				inv_yhat = concatenate((dataset[:,: dataset.shape[1] - 1], p), axis=1)

				predo_vals = [int(i) for i in dataset[:,-1]]

				data = dict(zip(list_hours, predo_vals))

				jsonFile = open(self.dir_path + '/data/tomorrow/' + station + '.json', 'w')
				jsonFile.write(json.dumps(data))

				self.p.two_plot(dataset[:,-1], today_data, "Tiempo", "Bicicletas", str("Prediction for " + station + " for today (" + weekday + ")"), self.dir_path + "/plots/tomorrow/" + station, text = "", line_1 = "Prediction", line_2 = "Real Value")
