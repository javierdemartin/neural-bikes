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
	list_of_stations = []
	enable_scale = True
	
	hour_encoder    = LabelEncoder()
	weekday_encoder = LabelEncoder()
	station_encoder = LabelEncoder()
	
	dataset_percentage_reduction = 0
	
# 	og_columns = ['datetime','time', 'weekday', 'station_name', 'value']

	

	

	# generated_columns = ['time', 'weekday', 'station_name', 'label', 'value']
	
	len_day = 144

	city = ""
	dbDaysQueryThresshold = 30
	scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values
	availability_db_name = ""
	prediction_db_name = ""
	queries_database = False
	station_dict = {}

	
	n_out = len_day

	def __init__(self):

			if len(sys.argv) < 3: raise ValueError("[USAGE] python3 script CITY INFLUX_DB_PASSWORD")

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

	
			self.city = sys.argv[1]

			self.timer = Timer(city = self.city)

			self.availability_db_name = "Bicis_" + self.city + "_Availability"
			self.prediction_db_name = "Bicis_" + self.city + "_Prediction"

			self.db_password = sys.argv[2]
			
			self.plotter = Plotter()
			self.utils = Utils(city = self.city)

			self.client = InfluxDBClient(self.db_ip, '8086', 'root', self.db_password, self.availability_db_name) 

			self.utils.check_and_create(["/data/" + self.city])
			self.utils.check_and_create(["/data/" + self.city + "/tomorrow", "/data/" + self.city + "/yesterday", '/data/' + self.city + '/cluster/', '/data/' + self.city + '/today/', "/data/" + self.city + "/today", "/data/" + self.city + "/yesterday/", "/data/" + self.city + "/filled", "/model/" + self.city , "/data/utils/", "/plots/" + self.city, "/data/" + self.city + "/supervised", "/data/" + self.city + "/scaled", "/data/" + self.city + "/filled", "/data/" + self.city + "/encoders", "/data/" + self.city + "/encoded_data"])
			
			self.list_hours = ["00:00","00:10","00:20","00:30","00:40","00:50","01:00","01:10","01:20","01:30","01:40","01:50","02:00","02:10","02:20","02:30","02:40","02:50","03:00","03:10","03:20","03:30","03:40","03:50","04:00","04:10","04:20","04:30","04:40","04:50","05:00","05:10","05:20","05:30","05:40","05:50","06:00","06:10","06:20","06:30","06:40","06:50","07:00","07:10","07:20","07:30","07:40","07:50","08:00","08:10","08:20","08:30","08:40","08:50","09:00","09:10","09:20","09:30","09:40","09:50","10:00","10:10","10:20","10:30","10:40","10:50","11:00","11:10","11:20","11:30","11:40","11:50","12:00","12:10","12:20","12:30","12:40","12:50","13:00","13:10","13:20","13:30","13:40","13:50","14:00","14:10","14:20","14:30","14:40","14:50","15:00","15:10","15:20","15:30","15:40","15:50","16:00","16:10","16:20","16:30","16:40","16:50","17:00","17:10","17:20","17:30","17:40","17:50","18:00","18:10","18:20","18:30","18:40","18:50","19:00","19:10","19:20","19:30","19:40","19:50","20:00","20:10","20:20","20:30","20:40","20:50","21:00","21:10","21:20","21:30","21:40","21:50","22:00","22:10","22:20","22:30","22:40","22:50","23:00","23:10","23:20","23:30","23:40","23:50"]
			
			bah = self.utils.stations_from_web(self.city)
			bah.drop(bah.columns[[2,3]], axis=1, inplace=True)
			self.station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))
			
			self.list_of_stations = list(bah.values[:,1])
			
			self.utils.save_array_txt(self.dir_path + "/data/" + self.city +  "/list_of_stations", self.list_of_stations)
			
			self.hour_encoder.fit(self.list_hours)
			
			self.weekday_encoder.classes_ = self.weekdays
			
	def read_dataset(self, no_date_split = False):
			""" Query the InfluxDB for all the availability data for a city.
			Data will be returnes in the form of a pandas.Dataframe and saved to disk in the
			../data/CITY/CITY.pkl cirectory
			"""

			self.timer.start()
			
			# If file already exists on disk check when was previously downloaded
			if os.path.isfile(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl"):
			
					mtime = os.path.getmtime(self.dir_path + "/data/" + self.city + "/"  + self.city + ".pkl")
			
					last_modified_date = datetime.fromtimestamp(mtime)
			
					timeDiff = datetime.now() - last_modified_date
			
					if timeDiff.days < self.dbDaysQueryThresshold:
			
							print("Dataset was downloaded " + str(timeDiff.days) + " days ago.")
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

# 									dataset["datetime"] = dataset["time"]
# 									dataset["weekday"]  = dataset["time"]

									f = lambda x: datetime.strptime(x.split("T")[0],'%Y-%m-%d').timetuple().tm_yday 
									dataset["datetime"] = dataset["time"].apply(f)

									f = lambda x: self.weekdays[datetime.strptime(x.split("T")[0],'%Y-%m-%d').weekday()] 
									dataset["weekday"] = dataset["time"].apply(f)
					
									dataset["time"]  = times

									# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
									ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]
					
									for p in ps:
											dataset = dataset[~dataset['time'].str.contains(p)]
											dataset = dataset[dataset['station_name'].isin(self.list_of_stations)] # TODO: Debugging

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
									dataset = dataset[dataset['station_name'].isin(self.list_of_stations)] # TODO: Debugging

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
				values[:,station_index] = self.station_encoder.transform(values[:,station_index])  # Encode STATION as an integer value
			
			self.save_encoders()
			
			return values

	def save_encoders(self):
		
	
			np.save(self.dir_path + '/data/' +  self.city +  '/encoders/hour_encoder.npy', self.hour_encoder.classes_)
			np.save(self.dir_path + '/data/' +  self.city +  '/encoders/weekday_encoder.npy', self.weekday_encoder.classes_)
			np.save(self.dir_path + '/data/' +  self.city +  '/encoders/station_encoder.npy', self.station_encoder.classes_)


	# Calls `series_to_supervised` and then returns a list of arrays, in each one are the values for each station
	def supervised_learning(self):
	
			print("[SUPERVISED LEARNING]")
			self.timer.start()
			
			self.scaler = self.get_maximums_pre_scaling()

			# Encontrar los índices de las columnas a borrar
			#################################################

			list_of_indexes = []

			for to_delete in self.dont_predict:
					indices = [i for i, x in enumerate(self.generated_columns) if x == to_delete]  
					list_of_indexes.append(indices[0])

			# Generar los índices para todas las muestras que están a la salida

			final_list_indexes = []

			for out in range(self.n_out):
					final_list_indexes.append([x+ len(self.generated_columns)*out for x in list_of_indexes])

			# Lista `final_list_indexes` es una lista dentro de una lista [[a,b],[c,d]...], flatten para obtener una unica lista
			final_list_indexes = list(itertools.chain.from_iterable(final_list_indexes))

			# Añadir las muestras que faltan de los valores de entrada, esta desplazado hacia la derecha por eso
			final_list_indexes = [x+ len(self.generated_columns)*self.n_in for x in final_list_indexes]
			
			for idx, station in enumerate(self.list_of_stations):

					print("[" + str(idx) + "/" + str(len(self.list_of_stations)) + "] " + station, end="\r")

					try:
							dataset = np.load(self.dir_path + '/data/' + self.city + '/filled/' + self.station_dict[station] + '.npy')
							
							dataset = dataset.reshape(-1, dataset.shape[-1])
							
							#if dataset.shape[0] == 0: continue

							dataset = self.scaler_helper(dataset)

							dataframe = pd.DataFrame(data=dataset, columns=self.generated_columns)

							supervised = self.series_to_supervised(self.generated_columns, dataframe, self.n_in, self.n_out)

							supervised = supervised.drop(supervised.columns[final_list_indexes], axis=1)

							# Eliminar cada N lineas para  no tener las muestras desplazadas
							rows_to_delete = []

							for j in range(supervised.shape[0]):

									if j % self.n_in != 0:
											rows_to_delete.append(j)

							supervised = supervised.drop(supervised.index[rows_to_delete])
							supervised = supervised.reset_index(drop = True)

							self.utils.save_array_txt(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station], supervised.values)
							np.save(self.dir_path + "/data/" + self.city + "/supervised/" + self.station_dict[station] + '.npy', supervised.values)
							
					except (FileNotFoundError, IOError):
							print("Wrong file or file path (" + '/data/' + self.city + '/scaled/' + str(self.station_dict[station]) + ".npy)" )
					
			aux = np.load(self.dir_path + "/data/" + self.city  + "/supervised/" + self.station_dict[self.list_of_stations[0]] + ".npy")
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
			np.save(self.dir_path + "/data/" + self.city + "/supervised/" + self.city + ".npy", final_data)

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

	def iterate(self, dataset, cluster_data):
			"""
			Iterate through all the stations and fill missing values.
			"""
	
			# Crear diccionario que relacione las estaciones con su cluster
			#self.cluster_data = pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
			self.cluster_data = cluster_data
			# Convert the DataFrame into a JSON
			# Key is the station_name & value is the cluster_id
			self.cluster_data = dict(zip(self.cluster_data.values[:,0], self.cluster_data.values[:,1]))     
			
			self.list_of_stations = list(self.cluster_data.keys())
			self.station_encoder.classes_ = self.list_of_stations

			path_to_save = os.path.join(self.dir_path, 'data', self.city, 'filled')
			
			my_data = dataset #self.read_dataset()

			self.timer.start()
							
			for idx, station in enumerate(self.list_of_stations):
			
					# NO esta en cluster data asi que no me lo guardes
					if station not in self.cluster_data: 
							self.station_dict.pop(station)

					if station not in self.station_dict:
							print("> Missing key " + station)
							self.list_of_stations.remove(station)
							continue

					current_iteration = my_data[my_data['station_id'].isin([self.station_dict[station]])]
					
					# If there aren't more than 2 weeks of data for that station discard it
					if current_iteration.shape[0] <= self.len_day * 7 * 2: continue
					
					current_iteration['time'] = pd.to_datetime(current_iteration['time'])  
										
					first_sample = current_iteration['time'].iloc[0].strftime('%Y-%m-%d')
					last_sample = current_iteration['time'].iloc[current_iteration.shape[0]-1].strftime('%Y-%m-%d')
					
					print("[" + str(idx) + "/" + str(len(self.list_of_stations)) + "] " + station + " (" + str(first_sample) + " to " + str(last_sample) + ")", end='\r')
					
					time_range = pd.date_range(first_sample + 'T00:00:00Z', last_sample + 'T00:00:00Z', freq='1D').strftime('%Y-%m-%dT00:00:00Z')
			
					aux = np.empty((0,self.len_day,len(self.generated_columns)))

					# Formatear los datos, evitar utilizar el último porque no está completo y puede ser el día de hoy
					for i in range(0, (len(time_range) - 1)):
					
							query_all = "select * from bikes where station_id = \'" + str(self.station_dict[station]) + "\' and time > \'" + str(time_range[i]) + "\' and time < \'" + str(time_range[i+1]) + "\'"

							daily = pd.DataFrame(self.client.query(query_all, chunked=True).get_points())

							# daily = dataset.loc[dataset['station_id'] == self.station_dict[station]]
							# date_mask = (dataset['time'] > time_range[i]) & (dataset['time'] <= time_range[i+1])
							# daily = daily[date_mask]

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
							
							aux = np.concatenate((aux,daily), axis = 0)

					aux_path = os.path.join(path_to_save, self.station_dict[station])
					
					np.save(aux_path, aux)

			self.timer.stop(" " + str(inspect.stack()[0][3]) + " for " + station + " (" + self.station_dict[station] + ") " + str(first_sample) + " to " + str(last_sample) + ")")

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

			return data

	# Read all the files and set the maximum values for each column,
	# RETURNS:
	#       · The scaler object
	def get_maximums_pre_scaling(self):

			# Get the maximum values for
			scaler_aux = MinMaxScaler(feature_range=(0,1))


			#print(self.dir_path +  '/data/' + self.city + '/filled/' + self.station_dict[self.list_of_stations[0]] + '.npy\b', end="")
			a = np.load(self.dir_path +  '/data/' + self.city + '/filled/' + self.station_dict[self.list_of_stations[0]] + '.npy')

			a = a.reshape(-1, a.shape[-1])

			for i in range(1, len(self.list_of_stations)):
					
					try:
							dataset = np.load(self.dir_path + "/data/" + self.city + "/filled/" + self.station_dict[self.list_of_stations[i]] + ".npy")
							
							if dataset.shape[1] == 0: continue

							dataset = dataset.reshape(-1, dataset.shape[-1])
					
							a = np.concatenate((a,dataset), axis = 0)
					
					except (FileNotFoundError, IOError):
							print("Wrong file or file path (" + self.dir_path + '/data/' + self.city + '/scaled/' + str(self.station_dict[self.list_of_stations[i]]) + ".npy)" )

													
			self.scaler.fit_transform(a)

			pickle.dump(self.scaler, open(self.dir_path +"/data/" + self.city +  "/MinMaxScaler.sav", 'wb'))

			return self.scaler

	def scale_dataset(self):
			
			if enable_scale is True:

					# Coger primero todos los máximos valores para luego escalar todos los datos poco a poco
					self.scaler = self.get_maximums_pre_scaling()

					for station in self.list_of_stations:

									dataset = np.load(self.dir_path + '/data/filled/' + str(self.station_dict[station]) + '_filled.npy')
									
									if dataset.shape[0] > (len_day*2):

											dataset = self.scaler.transform(dataset)

											np.save(self.dir_path + '/data/scaled/' + str(self.station_dict[station]) + ".npy", dataset)
											self.utils.save_array_txt(self.dir_path + '/data/scaled/' + str(self.station_dict[station]), dataset)

					pickle.dump(self.scaler, open(self.dir_path + "/data/" + self.city + "/MinMaxScaler.sav", 'wb'))

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


	def split_input_output(self, dataset, n_in, n_out):
			"""
			Data has been previously shuffled
			"""

# 			columns = ['datetime', 'time', 'weekday', 'station_name', 'label', 'value']
			
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

	def prepare_tomorrow(self, cluster_data):

			"""
			Queries InfluxDB database for yesterday's data, fills possible holes in the dataset and saves it into a NumPy array to later be fed to the trained model.

			The predictions are for tomorrow but they have to be done on that same day so the data gathered for the day is complete. Starts querying the database for
			each station and for the availability between yesterday and today. Later it gives it the necessary format, encodes, normalizes it and then saves it for later
			use predicting tomorrow's values with the neural_model script.

			"""
			
			self.cluster_data = cluster_data #pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
			self.cluster_data = dict(zip(self.cluster_data.values[:,0], self.cluster_data.values[:,1]))     
			
			self.hour_encoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/hour_encoder.npy')
			self.weekday_encoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/weekday_encoder.npy')
			self.station_encoder.classes_ = np.load(self.dir_path + '/data/' +  self.city +  '/encoders/station_encoder.npy')

			current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
			
			d = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))

			today     = datetime.today() 
			weekday   = self.weekdays[(today - timedelta(days=self.n_days_in)).weekday()]
			yesterday = today - timedelta(days=self.n_days_in)
			yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
			today     = today.strftime('%Y-%m-%dT00:00:00Z')
			
			for station in self.list_of_stations:
			
					if station not in self.station_dict:
							self.list_of_stations.remove(station)
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
					
					print(station)

					data['label'] = self.cluster_data[station]

					data = data[self.generated_columns] # Sort the DataFrame's columns
					
					# Encode columns that are strings to be numbers
					data = self.encoder_helper(data)
					data = self.scaler_helper(data)         
					
					# Reshape the data to be 3-Dimensional
					data = data.reshape(1,self.n_in,len(self.generated_columns))

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

			client_pred = InfluxDBClient(self.db_ip, '8086', 'root', 'root', 'Bicis_' + self.city + '_Prediction')

			today = datetime.today()
			weekday = self.weekdays[(today - timedelta(days=1)).weekday()]
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

