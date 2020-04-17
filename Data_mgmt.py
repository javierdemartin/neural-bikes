#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import Utils
from Plotter import Plotter
import pandas.core.frame # read_csv
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import pickle # Saving MinMaxScaler
from pandas import concat,DataFrame

import itertools
import numpy as np
import os
import time
import datetime
from Timer import Timer

import os.path
import sys
from datetime import timedelta, datetime
from influxdb import InfluxDBClient

import inspect

class Data_mgmt:

	db_ip = "192.168.86.99"
	weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
	list_hours       = []
	listOfStations = []
	enable_scale = True
	
	hour_encoder    = LabelEncoder()
	weekday_encoder = LabelEncoder()
	stationEncoder = LabelEncoder()
	
		
	len_day = 144

	city = ""
	dbDaysQueryThresshold = 30
	scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
	availability_db_name = ""
	prediction_db_name = ""
	queries_database = False
	station_dict = {}

	
	n_out = len_day

	def __init__(self, city):

		# Get current working directory 
		self.dir_path = os.path.dirname(os.path.realpath(__file__))

		with open(self.dir_path + '/config/config.json', 'r') as j:
			configs = json.loads(j.read())

		self.og_columns = configs['data']['og_columns']
		self.generated_columns = configs['data']['generated_columns']
		# When generating the output samples (Y) of the supervised problem
		# add the columns you don't want to predict
		self.dont_predict = configs['data']['dont_predict']

		self.n_days_in = configs['parameters']['lookback_days']
		self.n_in  = self.len_day * self.n_days_in# Number of previous samples used to feed the Neural Network

		self.dataset_percentage_reduction = configs['parameters']['dataset_percentage_reduction']

		self.city = city

		self.timer = Timer(city = self.city)

		self.availability_db_name = "Bicis_" + self.city + "_Availability"
		self.prediction_db_name = "Bicis_" + self.city + "_Prediction"

		self.db_password = "root"
		
		self.plotter = Plotter()
		self.utils = Utils(city = self.city)

		self.client = InfluxDBClient(self.db_ip, '8086', 'root', "root", self.availability_db_name) 

		self.utils.check_and_create(["/data/" + self.city])
		self.utils.check_and_create(["/data/" + self.city + "/tomorrow", "/data/" + self.city + "/yesterday", '/data/' + self.city + '/cluster/', '/data/' + self.city + '/today/', "/data/" + self.city + "/today", "/data/" + self.city + "/yesterday/", "/data/" + self.city + "/filled", "/model/" + self.city , "/data/utils/", "/plots/" + self.city, "/data/" + self.city + "/supervised", "/data/" + self.city + "/scaled", "/data/" + self.city + "/filled", "/data/" + self.city + "/encoders", "/data/" + self.city + "/encoded_data"])
		
		self.list_hours = ["00:00","00:10","00:20","00:30","00:40","00:50","01:00","01:10","01:20","01:30","01:40","01:50","02:00","02:10","02:20","02:30","02:40","02:50","03:00","03:10","03:20","03:30","03:40","03:50","04:00","04:10","04:20","04:30","04:40","04:50","05:00","05:10","05:20","05:30","05:40","05:50","06:00","06:10","06:20","06:30","06:40","06:50","07:00","07:10","07:20","07:30","07:40","07:50","08:00","08:10","08:20","08:30","08:40","08:50","09:00","09:10","09:20","09:30","09:40","09:50","10:00","10:10","10:20","10:30","10:40","10:50","11:00","11:10","11:20","11:30","11:40","11:50","12:00","12:10","12:20","12:30","12:40","12:50","13:00","13:10","13:20","13:30","13:40","13:50","14:00","14:10","14:20","14:30","14:40","14:50","15:00","15:10","15:20","15:30","15:40","15:50","16:00","16:10","16:20","16:30","16:40","16:50","17:00","17:10","17:20","17:30","17:40","17:50","18:00","18:10","18:20","18:30","18:40","18:50","19:00","19:10","19:20","19:30","19:40","19:50","20:00","20:10","20:20","20:30","20:40","20:50","21:00","21:10","21:20","21:30","21:40","21:50","22:00","22:10","22:20","22:30","22:40","22:50","23:00","23:10","23:20","23:30","23:40","23:50"]
		
		bah = self.utils.stations_from_web(self.city)
		bah.drop(bah.columns[[2,3]], axis=1, inplace=True)

		self.station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))
		
		self.listOfStations = list(bah.values[:,1])
		
		self.utils.save_array_txt(self.dir_path + "/data/" + self.city +  "/listOfStations", self.listOfStations)
		
		self.hour_encoder.fit(self.list_hours)
		
		self.weekday_encoder.classes_ = self.weekdays
			
	def read_dataset(self, no_date_split = False):
		""" Query the InfluxDB for all the availability data for a city.
		Data will be returnes in the form of a pandas.Dataframe and saved to disk in the
		../data/CITY/CITY.pkl cirectory
		"""

		self.timer.start()

		print("> Reading dataset")
		
		# If file already exists on disk check when was previously downloaded
		if os.path.isfile(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl"):
	
			mtime = os.path.getmtime(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")
	
			last_modified_date = datetime.fromtimestamp(mtime)
	
			timeDiff = datetime.now() - last_modified_date
	
			if timeDiff.days < self.dbDaysQueryThresshold:
	
				print("> Dataset was downloaded " + str(timeDiff.days) + " days ago.")
				dataset = pd.read_pickle(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")

				self.timer.stop("Dataset was downloaded " + str(timeDiff.days) + " days ago.")
					 
			# If the data os old enough query the server
			else: 
			
				# Query to make to the db       
				query_all = 'select * from bikes'

				dataset = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())
				
				#dataset.drop(dataset.columns[[0]], axis = 1, inplace = True) 

				dataset["value"] = pd.to_numeric(dataset["value"])

				if no_date_split == False:

					times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,1]]

					f = lambda x: datetime.strptime(x.split("T")[0],'%Y-%m-%d').timetuple().tm_yday 
					dataset["datetime"] = dataset["time"].apply(f)

					f = lambda x: self.weekdays[datetime.strptime(x.split("T")[0],'%Y-%m-%d').weekday()] 
					dataset["weekday"] = dataset["time"].apply(f)
	
					dataset["time"]  = times

					# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
					ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]
	
					for p in ps:
							dataset = dataset[~dataset['time'].str.contains(p)]
							dataset = dataset[dataset['station_name'].isin(self.listOfStations)] # TODO: Debugging

					# dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']]
					dataset = dataset[self.generated_columns]

				else:
					dataset['time'] = pd.to_datetime(dataset['time'])

				dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row

				# Devuelve un DataFrame con las siguientes columnas
				# [ bikes, time, station_id, station_name, value ]
				# Tratar el df eliminando la primera columna y la de time dividir la fecha en day of the year (datetime) y time.
				dataset.to_pickle(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")    #to save the dataframe, df to 123.pkl

				self.timer.stop("dataset downloaded from db")
		
		# File doesn't exist
		else:
		
			# Query to make to the db       
			query_all = 'select * from bikes'
	
			dataset = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())
					
			#dataset.drop(dataset.columns[[0]], axis = 1, inplace = True) 

			dataset["value"] = pd.to_numeric(dataset["value"])

			if no_date_split == False:

				times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,1]]

				dataset["datetime"] = dataset["time"]
				dataset["weekday"]  = dataset["time"]

				f = lambda x: datetime.strptime(x.split("T")[0],'%Y-%m-%d').timetuple().tm_yday 
				dataset["datetime"] = dataset["datetime"].apply(f)

				f = lambda x: self.weekdays[datetime.strptime(x.split("T")[0],'%Y-%m-%d').weekday()] 
				dataset["weekday"] = dataset["weekday"].apply(f)
		
				dataset["time"]  = times

				# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
				ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]
		
				for p in ps:
					dataset = dataset[~dataset['time'].str.contains(p)]
					dataset = dataset[dataset['station_name'].isin(self.listOfStations)] # TODO: Debugging

				# dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']]
				dataset = dataset[self.og_columns]

			else:
				dataset['time'] = pd.to_datetime(dataset['time'])

			dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row
	
			# Devuelve un DataFrame con las siguientes columnas
			# [ bikes, time, station_id, station_name, value ]
			# Tratar el df eliminando la primera columna y la de time dividir la fecha en day of the year (datetime) y time.
			dataset.to_pickle(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")    #to save the dataframe, df to 123.pkl

			self.timer.stop("dataset downloaded from db")

		return dataset
			
	def encoder_helper(self, dataset):

		# Encode the columns represented by a String with an integer with LabelEncoder()
		values = dataset.values         
		
		if "time" in self.generated_columns:
			hour_index = self.generated_columns.index("time")
			values[:,hour_index] = self.hour_encoder.transform(values[:,hour_index])     # Encode HOUR as an integer value

		if "weekday" in self.generated_columns:
			weekday_index = self.generated_columns.index("weekday")
			values[:,weekday_index] = self.weekday_encoder.transform(values[:,weekday_index])  # Encode WEEKDAY as an integer value

		if "station_name" in self.generated_columns:
			station_index = self.generated_columns.index("station_name")	
			values[:,station_index] = self.stationEncoder.transform(values[:,station_index])  # Encode STATION as an integer value
		
		self.save_encoders()
		
		return values

	def save_encoders(self):
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/hour_encoder.npy', self.hour_encoder.classes_)
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/weekday_encoder.npy', self.weekday_encoder.classes_)
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/stationEncoder.npy', self.stationEncoder.classes_)


	# Calls `series_to_supervised` and then returns a list of arrays, in each one are the values for each station
	def supervised_learning(self, scale=True):
		print("[SUPERVISED LEARNING]")
		self.timer.start()
		self.scaler = self.getMaximums()

		# Encontrar los índices de las columnas a borrar
		#################################################

		if "datetime" in self.generated_columns:
			weekday_index = self.generated_columns.index("datetime")

		list_of_indexes = []

		for to_delete in self.dont_predict:
				indices = [i for i, x in enumerate(self.generated_columns) if x == to_delete]  
				list_of_indexes.append(indices[0])

		# Generar los índices para todas las muestras que están a la salida

		indexesToKeep = []

		for out in range(self.n_out):
			indexesToKeep.append([x + len(self.generated_columns) * out for x in list_of_indexes])

		# Lista `indexesToKeep` es una lista dentro de una lista [[a,b],[c,d]...], flatten para obtener una unica lista
		indexesToKeep = list(itertools.chain.from_iterable(indexesToKeep))

		# Añadir las muestras que faltan de los valores de entrada, esta desplazado hacia la derecha por eso
		indexesToKeep = [x + len(self.generated_columns) * self.n_in for x in indexesToKeep]
		
		for idx, station in enumerate(self.listOfStations):

			try:
				# Load the previously processed data that has been filled with all possible holes
				dataset = np.load(self.dir_path + '/data/' + self.city + '/filled/' + self.station_dict[station] + '.npy')

				print("[" + str(idx) + "/" + str(len(self.listOfStations)) + "] " + str(station), end="\r")

				dataset = dataset.reshape(-1, dataset.shape[-1])

				if scale:
					dataset = self.scaler_helper(self.maximumBikesInStation[station], dataset)

				dataframe = pd.DataFrame(data=dataset, columns=self.generated_columns)

				supervised = self.series_to_supervised(self.generated_columns, dataframe, self.n_in, self.n_out)

				supervised = supervised.drop(supervised.columns[indexesToKeep], axis=1)

				# Eliminar cada N lineas para  no tener las muestras desplazadas
				rows_to_delete = []

				for j in range(supervised.shape[0]):
					if j % self.n_in != 0:
						rows_to_delete.append(j)

				supervised = supervised.drop(supervised.index[rows_to_delete])
				supervised = supervised.reset_index(drop = True)

				array_sum = np.sum(supervised.values)

				if np.isnan(array_sum): 
					print(supervised)
					asdfasdF()

				self.utils.save_array_txt(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station], supervised.values)
				np.save(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station] + '.npy', supervised.values)

				supervised.to_excel(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station] + '.xlsx')
					
			except (FileNotFoundError, IOError):
				print("Wrong file or file path in supervised learning (" + '/data/' + self.city + '/scaled/' + str(self.station_dict[station]) + ".npy)" )
				
		aux = np.load(self.dir_path + "/data/" + self.city  + "/supervised/" + self.station_dict[self.listOfStations[0]] + ".npy")
		final_data = np.empty(aux.shape)

		for key,value in self.station_dict.items():

			try:
				data_read = np.load(self.dir_path + "/data/" + self.city + "/supervised/" + value + ".npy")
				os.remove(self.dir_path + "/data/" + self.city + "/supervised/" + value + ".npy")
				final_data = np.append(final_data, data_read, 0)
				np.save(self.dir_path + "/data/" + self.city + "/supervised/" + str(value) + ".npy", final_data)
				os.remove(self.dir_path + "/data/" + self.city + "/supervised/" + value + ".npy")
					
			except (FileNotFoundError, IOError):
				print("Wrong file or file path (" + "/data/" + self.city + "/supervised/" + value + ".npy")

		self.utils.save_array_txt(self.dir_path + "/data/" + self.city + "/supervised/" + self.city, final_data)

		array_sum = np.sum(final_data)
		array_has_nan = np.isnan(array_sum)

		if array_has_nan: 
			print(final_data)
			asdfasdF()

		np.save(self.dir_path + "/data/" + self.city + "/supervised/" + self.city + ".npy", final_data)

		return final_data

		self.timer.stop("Supervised learning")


	def series_to_supervised(self, columns, data, n_in=1, n_out=1, dropnan=True):

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
				names += [(columns[j] + '(t)') for j in range(n_vars)]
			else:
				names += [(columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
		# put it all together
		agg = concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values

		if dropnan:
			agg.dropna(inplace=True)

		return agg

	def iterate(self, dataset, cluster_data):
		"""
		Iterate through all the stations and fill missing values.
		"""

		print("> Processing the data")

		# Crear diccionario que relacione las estaciones con su cluster
		#self.cluster_data = pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
		self.cluster_data = cluster_data
		# Convert the DataFrame into a JSON
		# Key is the station_name & value is the cluster_id
		self.cluster_data = dict(zip(self.cluster_data.values[:,0], self.cluster_data.values[:,1]))     
		
		self.listOfStations = list(self.cluster_data.keys())
		self.stationEncoder.classes_ = self.listOfStations

		path_to_save = os.path.join(self.dir_path, 'data', self.city, 'filled')
		
		self.timer.start()
						
		for idx, station in enumerate(self.listOfStations):
		
			# NO esta en cluster data asi que no me lo guardes
			if station not in self.cluster_data: 
					self.station_dict.pop(station)

			if station not in self.station_dict:
				print("> Missing key " + station)
				self.listOfStations.remove(station)
				continue

			current_iteration = dataset[dataset['station_id'].isin([self.station_dict[station]])]
			
			# If there aren't more than 2 weeks of data for that station discard it
			if current_iteration.shape[0] <= self.len_day * 7 * 2: 
				print("> " + station + " has less than " + str(7*2) + " days of data")
				continue
			
			current_iteration['time'] = pd.to_datetime(current_iteration['time'])  
								
			firstSample = current_iteration['time'].iloc[0].strftime('%Y-%m-%d')
			lastSample = current_iteration['time'].iloc[current_iteration.shape[0]-1].strftime('%Y-%m-%d')
			
			print("[" + str(idx) + "/" + str(len(self.listOfStations)) + "] " + station + " (" + str(firstSample) + " to " + str(lastSample) + ")", end='\r')
			
			time_range = pd.date_range(firstSample + 'T00:00:00Z', lastSample + 'T00:00:00Z', freq='1D').strftime('%Y-%m-%dT00:00:00Z')
	
			currentStationArray = np.empty((0,self.len_day,len(self.generated_columns)))

			for i in range(0, (len(time_range) - 1)):
			
				query_all = "select * from bikes where station_id = \'" + str(self.station_dict[station]) + "\' and time > \'" + str(time_range[i]) + "\' and time < \'" + str(time_range[i+1]) + "\'"

				daily = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())

				# No proceses nada si el día no tiene más del 80% de las muestras, va a causar muchos errores
				if daily.size < int(self.len_day * 0.8): continue

				daily_range = pd.date_range(time_range[i].split("T")[0] + ' 00:00:00+00:00', time_range[i].split("T")[0] + ' 23:50:00+00:00', freq='10T')

				daily['time']       = pd.to_datetime(daily['time'])
				daily['station_id'] = daily['station_id']
				daily['value']      = pd.to_numeric(daily['value'])     
				
				weekday = self.weekdays[(daily_range[0]).weekday()]

				daily = daily.set_index(keys=['time']).resample('10min').bfill()
				daily = daily.reindex(daily_range, fill_value=np.NaN)

				daily['value'] = daily['value'].interpolate(limit_direction='both')
				daily['station_name'] = station
				daily['station_id'] = daily['station_id'].interpolate(limit_direction='both')

				daily = daily.reset_index()

				daily['weekday']  = weekday 
				daily['datetime'] = (daily_range[0]).timetuple().tm_yday
				daily["time"]     = self.list_hours

				daily['label'] = daily['station_name']
				
				daily.drop(['station_id', 'index'], axis=1, inplace=True)
				
				daily = daily.replace({'label': self.cluster_data})
				
				# Reorder columns
				daily = daily[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']]
										
				daily = pd.DataFrame(data=daily, columns=['datetime', 'time', 'weekday', 'station_name', 'label', 'value'])

				daily = daily[self.generated_columns]
				
				# Encode columns that are strings to be numbers
				daily = self.encoder_helper(daily)
				
				daily = daily.reshape((1,self.len_day,len(self.generated_columns)))

				array_sum = np.sum(daily)
				array_has_nan = np.isnan(array_sum)

				if array_has_nan: 
					print(daily)
					asdfasdF()
				
				currentStationArray = np.concatenate((currentStationArray,daily), axis = 0)

			aux_path = os.path.join(path_to_save, self.station_dict[station])
			
			np.save(aux_path, currentStationArray)

		self.timer.stop(" " + str(inspect.stack()[0][3]) + " for " + station + " (" + self.station_dict[station] + ") " + str(firstSample) + " to " + str(lastSample) + ")")


	maximumBikesInStation = {}

	# Read all the files and set the maximum values for each column,
	# RETURNS:
	#       · The scaler object
	def getMaximums(self):

		# Get the maximum values for
		scaler_aux = MinMaxScaler(feature_range=(0,1))

		print("> Finding data range")

		a = np.empty((0,len(self.generated_columns)))

		for i in range(0, len(self.listOfStations)):
				
			try:
				dataset = np.load(self.dir_path + "/data/" + self.city + "/filled/" + self.station_dict[self.listOfStations[i]] + ".npy")
				
				if dataset.shape[1] == 0: continue

				dataset = dataset.reshape(-1, dataset.shape[-1])
		
				a = np.concatenate((a,dataset), axis = 0)

				print(self.listOfStations[i])

				self.maximumBikesInStation[self.listOfStations[i]] = max(a[:,-1])
			
			except (FileNotFoundError, IOError):
				print("Wrong file or file path (" + self.dir_path + '/data/' + self.city + '/scaled/' + str(self.station_dict[self.listOfStations[i]]) + ".npy)" )
												
		self.scaler.fit_transform(a)

		print(self.maximumBikesInStation)

		f = open(self.dir_path +"/data/" + self.city +  "/Maximums.pkl", 'wb')
		pickle.dump(self.maximumBikesInStation, f)
		f.close()

		values_index = self.generated_columns.index("value")

		self.scaler.data_max_[values_index] = 100.0
		self.scaler.data_range_[values_index] = 100.0


		print("data min " + str(self.scaler.data_min_))
		print("data max " + str(self.scaler.data_max_))
		print("data rng " + str(self.scaler.data_range_))

		f = open(self.dir_path +"/data/" + self.city +  "/MinMaxScaler.sav", 'wb')
		pickle.dump(self.scaler, f)
		f.close()

		return self.scaler

	def scaler_helper(self, maximumBikes, dataset):

		"""
		Loads previously saved MinMaxScaler and scales an array.

		Parameters
		----------
		array : Numpy.ndarray((1,144,6))
				

		Returns
		-------
		no_missing_samples: Int
				Number of missing samples in the 
		missing_days: Int

		"""

		scaler = MinMaxScaler()

		f = open(self.dir_path + "/data/" + self.city +  '/MinMaxScaler.sav','rb')
		scaler = pickle.load(f)
		f.close()

		values_index = self.generated_columns.index("value")

		dataset[:,values_index] = dataset[:,values_index] / maximumBikes * 100

		if dataset.shape[0] > 0:                
			dataset = scaler.transform(dataset)

		return dataset


	def split_input_output(self, dataset, n_in, n_out):
		"""
		Data has been previously shuffled
		"""

		x, y = dataset[:,range(0,len(self.generated_columns) * n_in)], dataset[:,-n_out:] #dataset[:,n_out]
		
		x = x.reshape((x.shape[0], n_in, len(self.generated_columns))) # (...,n_in,4)  

		return x,y

	def load_datasets(self):

		"""
		Loads datasets used in the training from disk
		"""

		train_x      = np.load(self.dir_path + '/data/' + self.city + '/train_x.npy')
		train_y      = np.load(self.dir_path + '/data/'+ self.city + '/train_y.npy')
		test_x       = np.load(self.dir_path + '/data/' + self.city + '/test_x.npy')
		test_y       = np.load(self.dir_path + '/data/' +self.city + '/test_y.npy')
		validation_x = np.load(self.dir_path + '/data/' + self.city + '/validation_x.npy')
		validation_y = np.load(self.dir_path + '/data/' + self.city + '/validation_y.npy')

		return train_x, train_y, validation_x, validation_y, test_x, test_y

	def split_sets(self, training_size, validation_size, test_size):

		"""
		* Shuffle the dataset
		* Reduce (if necessary) the dataset's size
		* Create the train, validation & test datasets
		"""

		# Dataset with all the 
		values = np.load(self.dir_path + "/data/" + self.city + "/supervised/" + self.city + ".npy")
		
		# Reduce dataset's size as my computer cant handle all the dataset			
		number_of_rows = values.shape[0]
		number_of_rows_trimmed = int(number_of_rows * (100 - self.dataset_percentage_reduction)/100)
		
		print("> Datased thinned from " + str(number_of_rows) +  " rows to " + str(number_of_rows_trimmed) + " rows")
		
		values = values[:number_of_rows_trimmed]


		# Calculate the number of samples for each set based on the overall dataset size
		train_size_samples      = int(len(values) * training_size)
		validation_size_samples = int(len(values) * validation_size)
		test_size_samples       = int(len(values) * test_size)

		# Previously the data was stored in an array the stations were contiguous, shuffle them so when splitting
		# the datasets every station is spreaded across the array
		np.random.shuffle(values)

		# Divide the dataset into the three smaller groups, 
		# Each one contrains both the input and output values for the supervised problem
		train      = values[0:train_size_samples,:]
		validation = values[train_size_samples:train_size_samples + validation_size_samples, :]
		test       = values[train_size_samples + validation_size_samples:train_size_samples + validation_size_samples + test_size_samples, :]

		# Get the input and output values for each subset
		train_x, train_y           = self.split_input_output(train, self.n_in, self.n_out)
		validation_x, validation_y = self.split_input_output(validation, self.n_in, self.n_out)
		test_x, test_y             = self.split_input_output(test, self.n_in, self.n_out)

		# Save all the values to disk
		np.save(self.dir_path + '/data/' + self.city + '/train_x.npy', train_x)
		np.save(self.dir_path + '/data/' + self.city + '/train_y.npy', train_y)
		np.save(self.dir_path + '/data/' + self.city + '/test_x.npy', test_x)
		np.save(self.dir_path + '/data/' + self.city + '/test_y.npy', test_y)
		np.save(self.dir_path + '/data/' + self.city + '/validation_x.npy', validation_x)
		np.save(self.dir_path + '/data/' + self.city + '/validation_y.npy', validation_y)

		print("Train X " + str(train_x.shape))
		print("Train Y " + str(train_y.shape))
		print("Test X " + str(test_x.shape))
		print("Test Y " + str(test_y.shape))
		print("Validation X " + str(validation_x.shape))
		print("Validation Y " + str(validation_y.shape))


	def prepare_tomorrow(self, cluster_data = None):

		"""
		Queries InfluxDB database for yesterday's data, fills possible holes in the dataset and saves it into a NumPy array to later be fed to the trained model.

		The predictions are for tomorrow but they have to be done on that same day so the data gathered for the day is complete. Starts querying the database for
		each station and for the availability between yesterday and today. Later it gives it the necessary format, encodes, normalizes it and then saves it for later
		use predicting tomorrow's values with the neural_model script.

		"""

		print("> Getting " + str(self.n_days_in) + " days of availability from the database")

		if cluster_data is None:
			self.cluster_data = pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")

		# Load the dictionary that holds the maximum values per station name
		f = open(self.dir_path +"/data/" + self.city +  "/Maximums.pkl", 'rb')
		self.maximumBikesInStation = pickle.load(f)
		f.close()
		
		self.cluster_data = dict(zip(self.cluster_data.values[:,0], self.cluster_data.values[:,1]))     
		
		self.hour_encoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/hour_encoder.npy')
		self.weekday_encoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/weekday_encoder.npy')
		self.stationEncoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/station_encoder.npy')

		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
		
		d = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))

		today     = datetime.today() 
		weekday   = self.weekdays[(today - timedelta(days=self.n_days_in)).weekday()]
		yesterday = today - timedelta(days=self.n_days_in)
		yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
		today     = today.strftime('%Y-%m-%dT00:00:00Z')

		informationList = {}
		
		for station in self.listOfStations:

			stationElement = {}
		
			if station not in self.station_dict:
				self.listOfStations.remove(station)
				continue

			# Occurs in cases where the station has stopped being available, therefore
			# no predictions can be made
			if station not in self.cluster_data:
					continue

			query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + today + '\' and station_id=\'' + str(self.station_dict[station]) + '\''

			data = pd.DataFrame(self.client.query(query, chunked=True).get_points())

			# If no data is available for that station continue with the execution
			if data.size == 0: continue
			
			data['time']       = pd.to_datetime(data['time'])
			#data['station_id'] = pd.to_numeric(data['station_id'])
			data['value']      = pd.to_numeric(data['value'])

			date_str = data['time'].iloc[0].strftime('%Y-%m-%d')
			date_str_end = data['time'].iloc[data.shape[0]-1].strftime('%Y-%m-%d')
			
			time_range = pd.date_range(date_str + ' 00:00:00+00:00', date_str_end + ' 23:50:00+00:00', freq='10T')

			data = data.set_index(keys=['time']).resample('10min').bfill()
			data = data.reindex(time_range, fill_value=np.NaN)

			data['value']        = data['value'].interpolate(limit_direction='both')
			data['station_name'] = station
			data['station_id']   = data['station_id'].interpolate(limit_direction='both')

			data = data.reset_index()

			data['weekday']  = weekday
			
			if data.shape[0] < self.n_in: continue

			data['datetime'] = (datetime.today() - timedelta(days=1)).timetuple().tm_yday
			data["time"]     = self.list_hours * self.n_days_in
			
			data.drop(['station_id', 'index'], axis=1, inplace=True)

			data['label'] = self.cluster_data[station]

			data = data[self.generated_columns] # Sort the DataFrame's columns

			if station not in self.maximumBikesInStation:
				continue
			
			# Encode columns that are strings to be numbers
			data = self.encoder_helper(data)
			data = self.scaler_helper(self.maximumBikesInStation[station], data)         
			
			# Reshape the data to be 3-Dimensional
			data = data.reshape(1,self.n_in,len(self.generated_columns))

			informationList[station] = data

		return informationList

