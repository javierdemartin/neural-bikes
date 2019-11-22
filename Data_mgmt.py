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

import os.path
import sys
from datetime import timedelta, datetime
from influxdb import InfluxDBClient
# from cluster import Cluster

# Global Configuration Variables
# ------------------------------------------------------------------

weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers
station_encoder = LabelEncoder() # Encode columns that are not numbers

enable_scale = True

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values

len_day = 144

n_in  = len_day # Number of previous samples used to feed the Neural Network
n_out = len_day

class Data_mgmt:

	list_hours       = []
	list_of_stations = []
	city = ""
	availability_db_name = ""
	prediction_db_name = ""
	station_dict = {}


	def __init__(self):

		if len(sys.argv) < 3: raise ValueError("[USAGE] python3 script CITY INFLUX_DB_PASSWORD")

		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		# Guarda las estaciones que existen realmente en un CSV

	
		self.city = sys.argv[1]
		self.availability_db_name = "Bicis_" + self.city + "_Availability"
		self.prediction_db_name = "Bicis_" + self.city + "_Prediction"

		self.db_password = sys.argv[2]

		self.plotter = Plotter()
		self.utils = Utils(city = self.city)

		self.client = InfluxDBClient('localhost', '8086', 'root', self.db_password, self.availability_db_name) 

		self.columns = ['datetime', 'time', 'weekday', 'station', 'label', 'free_bikes']

		self.utils.check_and_create(["/data/" + self.city])
		self.utils.check_and_create(["/data/" + self.city + "/tomorrow", "/data/" + self.city + "/yesterday", '/data/' + self.city + '/cluster/', '/data/' + self.city + '/today/', "/data/" + self.city + "/today", "/data/" + self.city + "/yesterday/", "/data/" + self.city + "/filled", "/model/" + self.city , "/data/utils/", "/plots/" + self.city, "/data/" + self.city + "/supervised", "/data/" + self.city + "/scaled", "/data/" + self.city + "/filled", "/data/" + self.city + "/encoders", "/data/" + self.city + "/encoded_data"])
		
		self.list_hours = ["00:00","00:10","00:20","00:30","00:40","00:50","01:00","01:10","01:20","01:30","01:40","01:50","02:00","02:10","02:20","02:30","02:40","02:50","03:00","03:10","03:20","03:30","03:40","03:50","04:00","04:10","04:20","04:30","04:40","04:50","05:00","05:10","05:20","05:30","05:40","05:50","06:00","06:10","06:20","06:30","06:40","06:50","07:00","07:10","07:20","07:30","07:40","07:50","08:00","08:10","08:20","08:30","08:40","08:50","09:00","09:10","09:20","09:30","09:40","09:50","10:00","10:10","10:20","10:30","10:40","10:50","11:00","11:10","11:20","11:30","11:40","11:50","12:00","12:10","12:20","12:30","12:40","12:50","13:00","13:10","13:20","13:30","13:40","13:50","14:00","14:10","14:20","14:30","14:40","14:50","15:00","15:10","15:20","15:30","15:40","15:50","16:00","16:10","16:20","16:30","16:40","16:50","17:00","17:10","17:20","17:30","17:40","17:50","18:00","18:10","18:20","18:30","18:40","18:50","19:00","19:10","19:20","19:30","19:40","19:50","20:00","20:10","20:20","20:30","20:40","20:50","21:00","21:10","21:20","21:30","21:40","21:50","22:00","22:10","22:20","22:30","22:40","22:50","23:00","23:10","23:20","23:30","23:40","23:50"]
		
		bah = self.utils.stations_from_web(self.city)
		bah.drop(bah.columns[[2,3]], axis=1, inplace=True)
		self.station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))
		
		self.utils.save_array_txt(self.dir_path + "/data/" + self.city +  "/list_of_stations", self.list_of_stations)
		
		hour_encoder.fit(self.list_hours)
		
		weekday_encoder.classes_ = weekdays
		
	def read_dataset(self, no_date_split = False):
		'''
		Pide todos los datos almacenados a la base de datos de InfluxDB y realiza el formateo adecuado de los datos, lo devuelve y guarda en ../data/CIUDAD/CIUDAD.pkl
		No usado en el entrenamiento
		'''
			
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

			f = lambda x: weekdays[datetime.strptime(x.split("T")[0],'%Y-%m-%d').weekday()] 
			dataset["weekday"] = dataset["weekday"].apply(f)
			
			dataset["time"]  = times

			# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
			ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]
			
			for p in ps:
				dataset = dataset[~dataset['time'].str.contains(p)]
				dataset = dataset[dataset['station_name'].isin(self.list_of_stations)] # TODO: Debugging

			# dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']]
			dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'value']]

		else:
			dataset['time'] = pd.to_datetime(dataset['time'])

		dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row
		
		# Devuelve un DataFrame con las siguientes columnas
		# [ bikes, time, station_id, station_name, value ]
		# Tratar el df eliminando la primera columna y la de time dividir la fecha en day of the year (datetime) y time.
		dataset.to_pickle(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")    #to save the dataframe, df to 123.pkl

		return dataset
		
	def encoder_helper(self, dataset):

		# Encode the columns represented by a String with an integer with LabelEncoder()
		values = dataset.values	
		

		values[:,1] = hour_encoder.transform(values[:,1])     # Encode HOUR as an integer value
		values[:,2] = weekday_encoder.transform(values[:,2])  # Encode WEEKDAY as an integer value
		values[:,3] = station_encoder.transform(values[:,3])  # Encode STATION as an integer value
		
		return values

	def save_encoders(self):
	
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/hour_encoder.npy', hour_encoder.classes_)
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/weekday_encoder.npy', weekday_encoder.classes_)
		np.save(self.dir_path + '/data/' +  self.city +  '/encoders/station_encoder.npy', station_encoder.classes_)


	# Calls `series_to_supervised` and then returns a list of arrays, in each one are the values for each station
	def supervised_learning(self):

		columns = ['datetime', 'time', 'weekday', 'station', 'label', 'free_bikes']

		dont_predict = ['datetime', 'time', 'weekday', 'station', 'label']

		self.scaler = self.get_maximums_pre_scaling()

		# Encontrar los índices de las columnas a borrar
		#################################################

		list_of_indexes = []

		for to_delete in dont_predict:
			indices = [i for i, x in enumerate(columns) if x == to_delete]	

			list_of_indexes.append(indices[0])

		# Generar los índices para todas las muestras que están a la salida

		final_list_indexes = []

		for out in range(n_out):

			final_list_indexes.append([x+ len(columns)*out for x in list_of_indexes])

		# Lista `final_list_indexes` es una lista dentro de una lista [[a,b],[c,d]...], flatten para obtener una unica lista
		final_list_indexes = list(itertools.chain.from_iterable(final_list_indexes))

		# Añadir las muestras que faltan de los valores de entrada, esta desplazado hacia la derecha por eso
		final_list_indexes = [x+ len(columns)*n_in for x in final_list_indexes]

		for station in self.list_of_stations:

			print("> " + station)

			try:
				dataset = np.load(self.dir_path + '/data/' + self.city + '/filled/' + self.station_dict[station] + '.npy')
				
				dataset = dataset.reshape(-1, dataset.shape[-1])


				#if dataset.shape[0] == 0: continue

				dataset = self.scaler_helper(dataset)

				dataframe = pd.DataFrame(data=dataset, columns=columns)

				supervised = self.series_to_supervised(columns, dataframe, n_in, n_out)

				supervised = supervised.drop(supervised.columns[final_list_indexes], axis=1)

				# Eliminar cada N lineas para  no tener las muestras desplazadas
				rows_to_delete = []

				for j in range(supervised.shape[0]):

					if j % n_in != 0:
						rows_to_delete.append(j)

				supervised = supervised.drop(supervised.index[rows_to_delete])
				supervised = supervised.reset_index(drop = True)

				self.utils.save_array_txt(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station], supervised.values)
				np.save(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station] + '.npy', supervised.values)
				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path (" + self.dir_path + '/data/' + self.city + '/scaled/' + str(self.station_dict[station]) + ".npy)" )
			
		aux = np.load(self.dir_path + "/data/" + self.city  + "/supervised/" + self.station_dict[self.list_of_stations[0]] + ".npy")
		final_data = np.empty(aux.shape)

		for key,value in self.station_dict.items():

			try:
				data_read = np.load(self.dir_path + "/data/" + self.city + "/supervised/" + value + ".npy")
				final_data = np.append(final_data, data_read, 0)
				np.save(self.dir_path + "/data/" + self.city + "/supervised/" + str(value) + ".npy", final_data)
				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path")

		self.utils.save_array_txt(self.dir_path + "/data/" + self.city + "/supervised/FINAL", final_data)
		np.save(self.dir_path + "/data/" + self.city + "/supervised/FINAL.npy", final_data)


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

		return agg

	def iterate(self):
	
		print("*************************")
		print("** Assembling the data **")
		print("*************************")

		path_to_save = os.path.join(self.dir_path, 'data')
		path_to_save = os.path.join(path_to_save, self.city)
		path_to_save = os.path.join(path_to_save, 'filled')
		
		
		# Crear diccionario que relacione las estaciones con su cluster
		self.cluster_data = pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
		self.cluster_data = dict(zip(self.cluster_data.values[:,0], self.cluster_data.values[:,1]))	
		
		print(self.cluster_data)
		

		
		self.list_of_stations = list(self.cluster_data.keys())  # list(bah.values[:,1]) #["Colombia", "Delicias", "Prosperidad"] #list(bah.values[:,1])
		station_encoder.classes_ = self.list_of_stations
		
		self.save_encoders()
		
		print(self.list_of_stations)

		
		
		print(self.cluster_data)
		
		for station in self.list_of_stations:
		
			# NO esta en cluster data asi que no me lo guardes
			if station not in self.cluster_data: 
				print("NONO QUITALO")
				self.list_of_stations.remove(station)
				self.station_dict.pop(station)
				continue
		

			# Obtener el primer y ultimo dia de las muestras para iterar, hacer esta petición para 
			query_all = "SELECT * FROM bikes where station_id = \'" + str(self.station_dict[station]) + "\' GROUP BY * ORDER BY ASC LIMIT 1"
			first_sample = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())
			first_sample['time']       = pd.to_datetime(first_sample['time'])	
			first_sample = first_sample['time'].iloc[0].strftime('%Y-%m-%d')
	
			query_all = "SELECT * FROM bikes where station_id = \'" + str(self.station_dict[station]) + "\' GROUP BY * ORDER BY DESC LIMIT 1"
			last_sample = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())
			last_sample['time']       = pd.to_datetime(last_sample['time'])	
			last_sample = last_sample['time'].iloc[0].strftime('%Y-%m-%d')
			
			print("> " + station + " has samples from " + first_sample + " to " + last_sample)
# 
			time_range = pd.date_range(first_sample + 'T00:00:00Z', last_sample + 'T00:00:00Z', freq='1D').strftime('%Y-%m-%dT00:00:00Z')
			
			# Si no hay mas de una semana por estacion pues no lo toques
			# if len(time_range) < 7:
# 				continue
		
			aux = np.empty((0,144,6))

			# Formatear los datos, evitar utilizar el último porque no está completo y puede ser el día de hoy
			for i in range(0, (len(time_range) - 1)):
			
				query_all = "select * from bikes where station_id = \'" + str(self.station_dict[station]) + "\' and time > \'" + str(time_range[i]) + "\' and time < \'" + str(time_range[i+1]) + "\'"

				daily = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())

				# No proceses nada si el día no tiene más del 80% de las muestras, va a causar muchos errores
				if daily.size < int(len_day * 0.8): continue

				daily_range = pd.date_range(time_range[i].split("T")[0] + ' 00:00:00+00:00', time_range[i].split("T")[0] + ' 23:50:00+00:00', freq='10T')

				daily['time']       = pd.to_datetime(daily['time'])
				daily['station_id'] = daily['station_id']
				daily['value']      = pd.to_numeric(daily['value'])	
				
				weekday = weekdays[(daily_range[0]).weekday()]

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

				daily = daily[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']]

				# Encode columns that are strings to be numbers
				daily = self.encoder_helper(daily)
				
				daily = daily.reshape((1,144,6))
				
				aux = np.concatenate((aux,daily), axis = 0)



			aux_path = os.path.join(path_to_save, self.station_dict[station])

			np.save(aux_path, aux)

	def new_fill_holes(self, data):

		initial_rows = data.shape[0]

		station = data['station_name'].iloc[0]

		data['time']       = pd.to_datetime(data['time'])
		data['station_id'] = data['station_id']
		data['value']      = pd.to_numeric(data['value'])	

		date_str = data['time'].iloc[0].strftime('%Y-%m-%d')

		hour = data['time'].iloc[-1].strftime("%H:%M")

		time_range = pd.date_range(date_str + ' 00:00:00+00:00', date_str + ' ' + str(hour) + ':00+00:00', freq='10T')

		data = data.set_index(keys=['time']).resample('10min').bfill()
		data = data.reindex(time_range, fill_value=np.NaN)

		data['value'] = data['value'].interpolate(limit_direction='both')
		data['station_name'] = station
		data['station_id'] = data['station_id'].interpolate(limit_direction='both')

		data = data.reset_index()

		final_rows = data.shape[0]

		print("Added " + str(final_rows - initial_rows) + " rows")



		return data

	# Read all the files and set the maximum values for each column,
	# RETURNS:
	#	· The scaler object
	def get_maximums_pre_scaling(self):


		# Get the maximum values for
		scaler_aux = MinMaxScaler(feature_range=(0,1))

		print(self.station_dict)
		print(self.list_of_stations)
		print("---------------------------")

		print(self.dir_path +  '/data/' + self.city + '/filled/' + self.station_dict[self.list_of_stations[0]] + '.npy')
		a = np.load(self.dir_path +  '/data/' + self.city + '/filled/' + self.station_dict[self.list_of_stations[0]] + '.npy')

		a = a.reshape(-1, a.shape[-1])

		for i in range(1, len(self.list_of_stations)):

			print(self.list_of_stations[i] + " - " + str(self.station_dict[self.list_of_stations[i]]))

			dataset = np.load(self.dir_path + "/data/" + self.city + "/filled/" + self.station_dict[self.list_of_stations[i]] + ".npy")

			if dataset.shape[1] == 0: continue

			dataset = dataset.reshape(-1, dataset.shape[-1])
			
			a = np.concatenate((a,dataset), axis = 0)
		
		np.savetxt(self.dir_path + "/output.txt", a, delimiter=" ", newline = "\n", fmt="%s")
				
		scaler.fit_transform(a)

		pickle.dump(scaler, open(self.dir_path +"/data/" + self.city +  "/MinMaxScaler.sav", 'wb'))

		return scaler

	def scale_dataset(self):
		
		if enable_scale is True:

			# Coger primero todos los máximos valores para luego escalar todos los datos poco a poco
			self.scaler = self.get_maximums_pre_scaling()

			for station in self.list_of_stations:

					dataset = np.load(self.dir_path + '/data/filled/' + str(self.station_dict[station]) + '_filled.npy')
					
					if dataset.shape[0] > (len_day*2):

						dataset = scaler.transform(dataset)

						np.save(self.dir_path + '/data/scaled/' + str(self.station_dict[station]) + ".npy", dataset)
						self.utils.save_array_txt(self.dir_path + '/data/scaled/' + str(self.station_dict[station]), dataset)

			pickle.dump(scaler, open(self.dir_path + "/data/" + self.city + "/MinMaxScaler.sav", 'wb'))

	def scaler_helper(self, dataset):

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
		scaler = pickle.load(open(self.dir_path + "/data/" + self.city + "/MinMaxScaler.sav", 'rb'))


		if dataset.shape[0] > 0:		
			dataset = scaler.transform(dataset)

		return dataset


	def split_input_output(self, dataset):

		columns = ['datetime', 'time', 'weekday', 'station', 'label', 'free_bikes']

		x, y = dataset[:,range(0,len(columns) * n_in)], dataset[:,-n_out:] #dataset[:,n_out]
		
		x = x.reshape((x.shape[0], n_in, len(columns))) # (...,n_in,4)	
		
		

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
		Loads the main dataset previously encoded, normalized and transformed into a supervised learning problem to then split it into the datasets used for training
		"""

		# Dataset with all the 
		values = np.load(self.dir_path + "/data/" + self.city + "/supervised/FINAL.npy")

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
		train_x, train_y           = self.split_input_output(train)
		validation_x, validation_y = self.split_input_output(validation)
		test_x, test_y             = self.split_input_output(test)

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
		
		print(train_x)
		print(train_y)
	

	def prepare_tomorrow(self):

		"""
		Queries InfluxDB database for yesterday's data, fills possible holes in the dataset and saves it into a NumPy array to later be fed to the trained model.

		The predictions are for tomorrow but they have to be done on that same day so the data gathered for the day is complete. Starts querying the database for
		each station and for the availability between yesterday and today. Later it gives it the necessary format, encodes, normalizes it and then saves it for later
		use predicting tomorrow's values with the neural_model script.

		"""

		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
		
		d = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))

		today     = datetime.today() 
		weekday   = weekdays[(today - timedelta(days=1)).weekday()]
		yesterday = today - timedelta(days=1)
		yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
		today     = today.strftime('%Y-%m-%dT00:00:00Z')
		
		for station in self.list_of_stations:

			query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + today + '\' and station_id=\'' + str(self.station_dict[station]) + '\''

			print(query)

			data = pd.DataFrame(self.client.query(query, chunked=True).get_points())

			# If no data is available for that station continue with the execution
			if data.size == 0: continue

			data['time']       = pd.to_datetime(data['time'])
			#data['station_id'] = pd.to_numeric(data['station_id'])
			data['value']      = pd.to_numeric(data['value'])

			date_str = data['time'].iloc[0].strftime('%Y-%m-%d')
			time_range = pd.date_range(date_str + ' 00:00:00+00:00', date_str + ' 23:50:00+00:00', freq='10T')

			data = data.set_index(keys=['time']).resample('10min').bfill()
			data = data.reindex(time_range, fill_value=np.NaN)

			data['value']        = data['value'].interpolate(limit_direction='both')
			data['station_name'] = station
			data['station_id']   = data['station_id'].interpolate(limit_direction='both')

			data = data.reset_index()

			data['weekday']  = weekday
			data['datetime'] = (datetime.today() - timedelta(days=1)).timetuple().tm_yday
			data["time"]     = self.list_hours
			
			data.drop(['station_id', 'index'], axis=1, inplace=True)


			data['label'] = self.cluster_data[station]

			data = data[['datetime', 'time', 'weekday', 'station_name', 'label', 'value']] # Sort the DataFrame's columns

			# Encode columns that are strings to be numbers
			data = self.encoder_helper(data)

			data = self.scaler_helper(data)		

	
			# Reshape the data to be 3-Dimensional
			data = data.reshape(1,144,6)

			# Save the data to a NumPy array
			np.save(self.dir_path + "/data/" + self.city + "/yesterday/" + self.station_dict[station] + ".npy", data)
				
	def prepare_today(self):

		"""
		Queries InfluxDB's availability database to give the correct format to the data and then saves it to the JSON file. 
		Saved to /data/today
		"""

		# Get current time, format the hour, minutes and seconds to be midnight
		# This is to get all the values for the day
		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))



		for station in self.list_of_stations:

			print("> " + str(station))

			dataset = self.get_today_data_from_station(station)

			# If no data is available for 2ddthat station continue with the execution
			if dataset.size == 0: continue
			values = dataset['value'].values

			values = [int(i) for i in values] # Cast from int64 to int

			data = dict(zip(self.list_hours, values))
			

			jsonFile = open(self.dir_path + '/data/' + self.city + '/today/' + str(self.station_dict[station]) + '.json', 'w+')
			jsonFile.write(json.dumps(data))
			jsonFile.close()

	

	def get_today_data_from_station(self, station_name):

		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
		query = 'select * from bikes where time > \'' + str(current_time) + '\' and station_id = \'' + str(self.station_dict[station_name]) + '\''

		dataset = pd.DataFrame(self.client.query(query, chunked=True).get_points())
		
		# Fill the holes for today's availability in case the server was unresponsive
		dataset = self.new_fill_holes(dataset)
		dataset['index'] = dataset['index'].apply(lambda x: x.strftime('%Y-%m-%dT00:00:00Z'))

		#dataset.drop(['station_id','station_name'], axis=1, inplace=True) # Remove unused columns

		times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,0]]

		dataset["time"] = times

		# Delete incorrectly sampled hours that don't match ten minute intervals
		dataset = dataset[dataset['time'].isin(self.list_hours)]

		return dataset

	def plot_today(self):

		"""
		Get today's prediction and availability and make a plot overlying them to compare the predictions vs the real value.
		"""

		client_pred = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_' + self.city + '_Prediction')

		today = datetime.today()
		weekday = weekdays[(today - timedelta(days=1)).weekday()]
		yesterday = today - timedelta(days=1)
		yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
		today = today.strftime('%Y-%m-%dT00:00:00Z')

		for station in self.list_of_stations:

			# query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + str(today)  + '\' and station_name=\'' + str(station) + '\''
			query = 'select * from bikes where time > \'' + str(today)  + '\' and station_name=\'' + str(station) + '\''

			# Query today's availability and get the free bikes from Pandas DataFrame
			data_today = pd.DataFrame(self.client.query(query, chunked=True).get_points())

			if data_today.size == 0: continue

			data_today = [int(i) for i in data_today['value'].values]

			print("##################################################")

			query = 'select * from bikes where time > \'' + str(today)  + '\' and station_name=\'' + str(station) + '\''


			# Query today's prediction and get the free bikes from Pandas DataFrame
			data_pred = pd.DataFrame(client_pred.query(query, chunked=True).get_points())

			data_pred = [int(i) for i in data_pred['value'].values]

			self.plotter.two_plot(data_today, data_pred, "tiempo", "bicis", "Prediction vs real values for " + str(today.split('T')[0]) + " (" + str(weekday) + ")", self.dir_path + "/plots/" + self.city + "/tomorrow/" + station)

