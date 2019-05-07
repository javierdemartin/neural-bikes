#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import Utils
from Plotter import Plotter
import pandas.core.frame # read_csv
import datetime
import json
from color import color
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle # Saving MinMaxScaler
from pandas import concat,DataFrame
import itertools
import numpy as np
import os
from datetime import timedelta

# Global Configuration Variables
# ------------------------------------------------------------------

dir_path = os.path.dirname(os.path.realpath(__file__))

train_model = True
enable_scale = True

weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers
station_encoder = LabelEncoder() # Encode columns that are not numbers

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values

len_day = 144

n_in  = len_day # Number of previous samples used to feed the Neural Network
n_out = len_day

class Data_mgmt:

	list_hours = []
	list_of_stations = []

	def __init__(self):

		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		
		# os.system("cp /Users/javierdemartin/Documents/bicis/data/Bilbao.txt " + self.dir_path + "/data/")

		# os.system("chmod 755 " + self.dir_path + "/data/Bilbao.txt")

		self.plotter = Plotter()
		self.utils = Utils()
		self.utils.init_tutorial()

		self.columns=['datetime', 'time', 'weekday', 'station', 'free_bikes']

		self.utils.check_and_create("/debug/encoders")
		self.utils.check_and_create("/debug/encoded_data")
		self.utils.check_and_create("/debug/filled")
		self.utils.check_and_create("/debug/scaled")
		self.utils.check_and_create("/debug/supervised")
		self.utils.check_and_create("/stats/stations")
		self.utils.check_and_create("/plots/")
		self.utils.check_and_create("/debug/utils/")
		self.utils.check_and_create("/model/")
		self.utils.check_and_create("/debug/filled")
		self.utils.check_and_create("/debug/yesterday/")
		self.utils.check_and_create("/debug/today")
		self.utils.check_and_create('/data/today/')
		
		self.list_hours = self.get_hour_list()

		self.list_of_stations = self.new_get_stations()

	def get_hour_list(self):

		p = "..:.5"
		list_hours = self.utils.read_csv_as_list(self.dir_path + '/debug/utils/list_hours')

		a = pd.DataFrame(list_hours)
		a = a[~a[0].str.contains(p)]
		list_hours = [i[0] for i in a.values.tolist()]

		return list_hours
		
	def new_read_dataset(self, save_path):
		
		'''
		Pide todos los datos almacenados a la base de datos de InfluxDB y realiza el formateo adecuado
		'''
		
		from influxdb import InfluxDBClient
		client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')
		
		query_all = 'select * from bikes'
		
		dataset = pd.DataFrame(client.query(query_all, chunked=True).get_points())
		
		# dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'value']]
		
		dataset.drop(dataset.columns[[0]], axis = 1, inplace = True) 


		times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,1]]

		dataset["datetime"] = dataset["time"]
		dataset["weekday"] = dataset["time"]

		f = lambda x: datetime.datetime.strptime(x.split("T")[0],'%Y-%m-%d').timetuple().tm_yday #  len(x["review"].split("disappointed")) -1
		dataset["datetime"] = dataset["datetime"].apply(f)

		f = lambda x: weekdays[datetime.datetime.strptime(x.split("T")[0],'%Y-%m-%d').weekday()] #  len(x["review"].split("disappointed")) -1
		dataset["weekday"] = dataset["weekday"].apply(f)
		
		dataset["time"] = times
		# dataset = dataset[dataset['time'].isin(self.list_hours)]
		dataset["value"] = pd.to_numeric(dataset["value"])

		dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'value']]

		# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
		ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]

		print(self.list_of_stations)
		
		for p in ps:
			dataset = dataset[~dataset['time'].str.contains(p)]
			dataset = dataset[dataset['station_name'].isin(self.list_of_stations)] # TODO: Debugging

		dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row

		
		# Devuelve un DataFrame con las siguientes columnas
		# [ bikes, time, station_id, station_name, value ]
		# Tratar el df eliminando la primera columna y la de time dividir la fecha en day of the year (datetime) y time.
		dataset.to_pickle(self.dir_path + save_path)    #to save the dataframe, df to 123.pkl
		

	def read_dataset(self, path, save_path):

		'''
		Reads dataset from file
		'''

		from influxdb import InfluxDBClient
		client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')

		self.utils.append_tutorial_title("Reading Dataset")

		# Read dataset from the CSV file
		dataset = pandas.read_csv(self.dir_path + path)

		dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] # Insert correct column names

		# Remove ID of the sation and free docks, I am not interested in using them
		dataset.drop(dataset.columns[[2,5]], axis = 1, inplace = True) 

		self.list_of_stations = self.get_list_of_stations(dataset.values[:,2])

		# Separar la columna de tiempos en fecha + hora
		times = [x.split(" ")[1] for x in dataset.values[:,0]]

		dataset['datetime'] = [datetime.datetime.today().strptime(x, '%Y/%m/%d %H:%M').timetuple().tm_yday for x in dataset.values[:,0]]

		# Insertar en la columna time las hoas
		dataset.insert(loc = 1, column = 'time', value = times)

		# Delete incorrectly sampled hours that don't match five minute intervals
		dataset = dataset[dataset['time'].isin(self.list_hours)]

		# Eliminar muestras queno hayan sido recogidas correctamente a horas que no sean intervalos de 10 minutos
		ps = ["..:.1", "..:.2", "..:.3", "..:.4", "..:.5", "..:.6", "..:.7", "..:.8", "..:.9"]
		
		for p in ps:
			dataset = dataset[~dataset['time'].str.contains(p)]
			dataset = dataset[dataset['station'].isin(self.list_of_stations)] # TODO: Debugging

		dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row

		text = "Reading dataset, data gathered every ten minutes."

		self.utils.append_tutorial(text, dataset.head(20))

		# Save the DataFrame to a Pickle file
		dataset.to_pickle(self.dir_path + save_path)    #to save the dataframe, df to 123.pkl

		os.system("chmod 755 " + self.dir_path + save_path)

	def new_get_stations(self):

		from influxdb import InfluxDBClient
		client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')
		query_all = 'select * from bikes'
		
		dataset = pd.DataFrame(client.query(query_all, chunked=True).get_points())

		list_of_stations = list(np.unique(dataset["station_name"].values))

		print(list_of_stations)

		client.close()

		self.utils.save_array_txt(self.dir_path + '/debug/utils/list_of_stations', list_of_stations)

		return list_of_stations


	# Gets a list of values and returns the list of stations
	def get_list_of_stations(self, array):

		"""
		Iterate through all the available given data and saves a list for all the available stations.

		Parameters
		----------
		array : Numpy.ndarray

		Returns
		-------
		array: Numpy.ndarray 
			Array of all the available stations

		"""

		if array is not None:

			if os.stat(self.dir_path + '/debug/utils/list_of_stations').st_size == 0:

				array = np.asarray(array)

				self.utils.save_array_txt(self.dir_path + '/debug/utils/list_of_stations', list(np.unique(array)))

				return list(np.unique(array))
					
			else:

				return self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")
			

		elif array is None:
			return None

	# Codificar las columnas seleccionadas con LabelEncoder
	def encode_data(self, read_path, save_path):

		"""
		Iterates through all the stations from debug/encoded_data and counts the missing samples 

		Parameters
		----------
		array : Numpy.ndarray
			

		Returns
		-------
		no_missing_samples: Int
			Number of missing samples in the 
		missing_days: Int

		"""

		dataset = pd.read_pickle(self.dir_path + read_path)

		print("READ DATASETO ")

		print(dataset)

		hour_encoder.fit(self.list_hours)
		station_encoder.classes_ = self.list_of_stations

		# Init the LabelEncoder for the weekdays with the previously saved data
		weekday_encoder.classes_ = weekdays

		self.utils.append_tutorial_title("Encoding Data")
		self.utils.append_tutorial_text("Encode each column as integers")
		self.utils.append_tutorial("Got list of " + str(len(self.list_of_stations)) + " stations before encoding", self.list_of_stations)
		self.utils.append_tutorial_title("Creating Label Encoders and then encoding the previously read dataset")

		self.utils.append_tutorial("Hour Encoder (" + str(len(hour_encoder.classes_)) + " values)", hour_encoder.classes_)
		self.utils.append_tutorial("Weekday Encoder (" + str(len(weekday_encoder.classes_)) + " values)", weekday_encoder.classes_)
		self.utils.append_tutorial("Station Encoder (" + str(len(station_encoder.classes_)) + " values)", station_encoder.classes_)

		# Save as a numpy array
		np.save(self.dir_path + '/debug/encoders/hour_encoder.npy', hour_encoder.classes_)
		np.save(self.dir_path + '/debug/encoders/weekday_encoder.npy', weekday_encoder.classes_)
		np.save(self.dir_path + '/debug/encoders/station_encoder.npy', station_encoder.classes_)

		# Encode the columns represented by a String with an integer with LabelEncoder()

		values = self.encoder_helper(dataset)

		dataset = pd.DataFrame(data=values, columns=self.columns)

		self.utils.append_tutorial("columns used in the training set", self.columns)

		# Save encoded data for each station to an independent file as a .npy file
		for station in self.list_of_stations:

			file_name = self.dir_path + save_path + station + ".npy"

			station_encoded_number = station_encoder.transform([station])[0]

			np.save(file_name, dataset[dataset['station'].isin([station_encoded_number])].reset_index(drop = True).values)


		self.utils.append_tutorial("Encoded dataset", dataset.head(20))

	def encoder_helper(self, dataset):

		hour_encoder.fit(self.list_hours)
		station_encoder.classes_ = self.list_of_stations

		weekday_encoder.classes_ = weekdays

		# Encode the columns represented by a String with an integer with LabelEncoder()
		values = dataset.values		

		values[:,1] = hour_encoder.transform(values[:,1])     # Encode HOUR as an integer value
		values[:,2] = weekday_encoder.transform(values[:,2])  # Encode WEEKDAY as an integer value
		values[:,3] = station_encoder.transform(values[:,3])  # Encode STATION as an integer value

		return values


	def load_encoders(self):
		return np.load(self.dir_path + '/debug/encoders/hour_encoder.npy'), np.load(self.dir_path +'/debug/encoders/weekday_encoder.npy'), np.load(self.dir_path + '/debug/encoders/station_encoder.npy')

	


	# Calls `series_to_supervised` and then returns a list of arrays, in each one are the values for each station
	def supervised_learning(self):

		self.utils.append_tutorial_title("Supervised Learning")
		self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

		columns = ['datetime', 'time', 'weekday', 'station', 'free_bikes']

		# dont_predict = ['datetime', 'time', 'weekday', 'station', 'free_bikes']
		dont_predict = ['datetime', 'time', 'weekday', 'station']

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

		self.utils.append_tutorial_text("| Station | Days | ")
		self.utils.append_tutorial_text("| --- | --- |")

		for station in self.list_of_stations:

			try:
				dataset = np.load(self.dir_path + '/debug/scaled/' + str(station) + ".npy")
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

				# print("-----------------------------------")
				# print(supervised)

				self.utils.append_tutorial_text("| " + station + " | " + str(supervised.shape[0]) + " | ")

				self.utils.save_array_txt(self.dir_path + "/debug/supervised/" + station, supervised.values)
				np.save(self.dir_path + "/debug/supervised/" + station + '.npy', supervised.values)
				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path (" + self.dir_path + '/debug/scaled/' + str(station) + ".npy)" )
			

		self.utils.append_tutorial_text("\n")

		final_data = np.load(self.dir_path + "/debug/supervised/" + self.list_of_stations[0] + ".npy")

		# Hacerlo con todas las estaciones
		for i in range(1,len(self.list_of_stations)):

			try:
				data_read = np.load(self.dir_path + "/debug/supervised/" + self.list_of_stations[i] + ".npy")
				final_data = np.append(final_data, data_read, 0)
				np.save(self.dir_path + "/debug/supervised/" + str(self.list_of_stations[i]) + ".npy", final_data)
				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path")


		self.utils.save_array_txt(self.dir_path + "/debug/supervised/FINAL", final_data)
		np.save(self.dir_path + "/debug/supervised/FINAL.npy", final_data)


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

	# Iterates through every station file looking for holes
	def iterate(self):
		"""
		Fills holes

		Iterates through all the stations from debug/encoded_data and fills in the gaps caused by the server not collecting data correctly.

		Parameters
		----------
		arg1 : int
			Description of arg1
		arg2 : str
			Description of arg2

		Returns
		-------
		int
			Des cription of return value

		"""

		print("> Finding holes")


		self.utils.append_tutorial_title("Finding holes in dataset")
		self.utils.append_tutorial_text("Los datos son recogidos cada 10' en el servidor y puede que en algunos casos no funcione correctamente y se queden huecos, arreglarlo inventando datos en esos huecos.\n")
		self.utils.append_tutorial_text("| Estación | Missing Samples | Missing Whole Days")
		self.utils.append_tutorial_text("| --- | --- | --- |")

		for station in self.list_of_stations:

			station_read = np.load(self.dir_path + "/debug/encoded_data/" + station + ".npy")

			# Problema cuadno aparece una estación nueva y se entrena el modelo con menos de un día de datos, no iterar si es nueva
			if station_read.shape[0] > len_day:

				no_missing_samples, missing_days = self.find_holes(station_read)

				filled_array = self.fill_holes(station_read, no_missing_samples)

				self.utils.append_tutorial_text(" | " + station +  " | " + str(no_missing_samples) + " | " + str(missing_days) + " | ")

				to_del = []
				i = 0

				# Delete rows that are zerossss
				for r in filled_array:
					if (r == np.array([0.0,0.0,0.0,0.0,0.0])).all() == True:
						to_del.append(i)

					i += 1

				filled_array = np.delete(filled_array,to_del,0)

				# Borrar para que empiecen en dia completo las muestras
				filled_array = np.delete(filled_array, (range(len_day - int(filled_array[0][1]))), axis=0)
				# Borrar las muestras finales que hacen que el día no esté completo
				filled_array = filled_array[:- (int(filled_array[filled_array.shape[0]-1][1])+1) ,:]

				self.utils.save_array_txt(self.dir_path + "/debug/filled/" + station + "_filled", filled_array)

				np.save(self.dir_path + '/debug/filled/' + station + '_filled.npy', filled_array)				

				if enable_scale is False: np.save(self.dir_path + '/debug/scaled/' + str(station) + ".npy", filled_array)

		self.utils.append_tutorial_text("\n\n")


	def get_hour_str(self, hour):

		return hour_encoder.inverse_transform(hour)


	def new_fill_holes(self, array, no_missing_samples):

		# Create the array with the final size
		rows = array.shape[0] + no_missing_samples
		columns = array.shape[1]

		filled_array = np.zeros((rows, columns))


	# El propósito es contar el número de muestras que faltan 
	def find_holes(self, array):

		"""
		Find holes

		Iterates through all the stations from debug/encoded_data and counts the missing samples 

		Parameters
		----------
		array : Numpy.ndarray
			

		Returns
		-------
		no_missing_samples: Int
			Number of missing samples in the 
		missing_days: Int

		"""

		current_row = array[0]

		no_missing_samples = 0
		missing_days       = 0

		# Iterar por todas las muestras del array leido
		# Comparando la muestra actual con la siguiente
		for i in range(1,array.shape[0]):

			# Dos muestras dentro del mismo día para rellenar
			if current_row[0] == array[i][0]:

				# No hay huecos entre una muestra y la siguiente, es correcto esto
				if (current_row[1]) != (array[i][1]-1):

					# Contar las muestras que sobran
					difference = array[i][1] - current_row[1] - 1

					no_missing_samples += difference

			# Días diferentes, hay cambio y comprobar si se ha hecho 
			elif current_row[0] != array[i][0]:

				# ✅ Inserta posibles muestras perdidas a las 00:00, sólo probado con una, faltaría si existe más de una pérdida
				if array[i][1] != 0 or current_row[1] != (len_day - 1):

					
					# ✅ Comprobar que son días seguidos, no rellenar más de un día
					# Se ha cortado entre un día y otro la lectura ->
					#   [41 143 6 0 17] - [42 1 0 0 15]
					if array[i][0] == (current_row[0] + 1):

						difference = len_day - current_row[1] + array[i][1] - 1

						no_missing_samples += difference

						# raise ValueError("Error que no has mirado, diferentes días " + str(current_row) + " - " + str(array[i]))

					# ✅ Hay diferencia de más de un dia incompleto, eliminar tanto el inicial como el final ya que los dos estan incompletos
					# --> [312 84 3 0 20] - [322 112 6 0 14] 
					else: 

						no_missing_samples += (len_day - current_row[1]) + array[i][1]

			# Conseguir nueva muestra
			current_row = array[i]

		return no_missing_samples, missing_days

	# Toma el número de muestras que faltan de la función de 'find_holes' e inserta los valores que faltan
	def fill_holes(self, array, no_missing_samples):

		# Create the array with the final size
		rows = array.shape[0] + no_missing_samples
		columns = array.shape[1]

		# Array en el que se insertarán las muestras, inicialmente está vacío y se introducen las muestras que están presentes
		# Luego en una segunda pasada se crean las nuevas que faltan
		filled_array = np.zeros((rows, columns))

		index = 0

		current_row = array[0]
		filled_array[0] = current_row

		for i in range(1, array.shape[0]):

			# Both samples are of the same day
			if current_row[0] == array[i][0]:

				# No coincide normalmente
				if (current_row[1]) != (array[i][1]-1):

					difference = array[i][1] - current_row[1] - 1

					# ✅ diferencia positiva entre muestras, rellenar huecos duplicando valores
					if (difference) > 0:

						missing_samples = array[i][1] - current_row[1] - 1

						# Iterar por toda la cantidad de muestras que faltan
						for j in range(0, missing_samples):

							filled_array[i + index] = current_row 
							filled_array[i + index][1] += 1 + j

							index += 1

						filled_array[i + index] = array[i]  

					# Diferencia negativa, cambio de hora probable
					else: 

						raise ValueError("Diferencia negativa de tiempos")

						for j in range(abs(array[i][1] - current_row[1] - 1)):

							filled_array[i + index] = current_row
							index -= 1
					
				# Dentro del mismo dia muestras okay, rellenar normal
				else:

					filled_array[i + index] = array[i]

			# ✅ Diferentes dias
			elif current_row[0] != array[i][0]:

				# ✅ Inserta posibles muestras perdidas a las 00:00, sólo probado con una, faltaría si existe más de una pérdida
				if array[i][1] != 0 or current_row[1] != (len_day - 1):


					
					# ✅ Comprobar que son días seguidos, no rellenar más de un día
					# [41 143 6 0 17] - [42 1 0 0 15]
					if array[i][0] == (current_row[0] + 1):

						
						
						filled_array[i + index] = array[i]
						filled_array[i + index][1] -= 1 

						index += 1

						filled_array[i + index] = array[i]

					# ✅ Hay diferencia de más de un dia incompleto, eliminar tanto el inicial como el final ya que los dos estan incompletos
					# [312 84 3 0 20] - [322 112 6 0 14]
					else: 

						# raise ValueError("Error que no has mirado, diferentes días " + str(current_row) + " - " + str(array[i]))				


						# Rellenar muestras trailing que faltan del día ANTERIOR
						# ------------------------------------------------------
						for j in range(0,len_day - current_row[1] - 1):

							filled_array[i + index] = current_row  # Introducir la muestra actual como la recreada
							filled_array[i + index][1] += 1 + j # Incrementar la hora de l amuestra que falta

							index += 1 # Incrementar el índice ya que se ha insertado una muestra

						# Rellenar muestras iniciales que falten del siguiente día
						for j in range(0, array[i][1]):

							filled_array[i + index] = array[i] 
							filled_array[i + index][1] = j

							index += 1

						# Añadir la última muestra que se quedaría como [0. 0. 0. 0. 0.]
						filled_array[i + index] = array[i]
						filled_array[i + index][1] = array[i][1]

				# ✅ Insertar las muestras a las 00:00 de forma normal
				else: 

					# raise ValueError("Error que no has mirado, diferentes días " + str(current_row) + " - " + str(array[i]))				

					filled_array[i + index] = array[i]

			current_row = array[i]

		return filled_array

	# Read all the files and set the maximum values for each column,
	# RETURNS:
	#	· The scaler object
	def get_maximums_pre_scaling(self):

		list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

		# Get the maximum values for
		scaler_aux = MinMaxScaler(feature_range=(0,1))

		dataset = np.load(self.dir_path +  '/debug/filled/' + list_of_stations[0] + '_filled.npy')

		a = dataset

		for i in range(1, len(list_of_stations)):

			dataset = np.load(self.dir_path + '/debug/filled/' + list_of_stations[i] + '_filled.npy')
			
			a = np.concatenate((a,dataset), axis = 0)

		scaler.fit_transform(a)

		return scaler

	def scale_dataset(self):
		
		if enable_scale is True:

			self.utils.append_tutorial_title("Scaling dataset")

			list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

			# Coger primero todos los máximos valores para luego escalar todos los datos poco a poco
			self.scaler = self.get_maximums_pre_scaling()

			for station in list_of_stations:

					dataset = np.load(self.dir_path + '/debug/filled/' + station + '_filled.npy')

					if dataset.shape[0] > (len_day*2):

						dataset = scaler.transform(dataset)

						np.save(self.dir_path + '/debug/scaled/' + str(station) + ".npy", dataset)
						self.utils.save_array_txt(self.dir_path + '/debug/scaled/' + str(station), dataset)

			pickle.dump(scaler, open(self.dir_path + "/MinMaxScaler.sav", 'wb'))

			self.utils.append_tutorial_text("| Values | datetime | time | weekday | station | free_bikes |")
			self.utils.append_tutorial_text("| --- | --- | --- | --- | --- | --- |")
			self.utils.append_tutorial_text("| Minimum Values | " + str(scaler.min_[0]) + " | " + str(scaler.min_[1]) + " | " + str(scaler.min_[2]) + " | " + str(scaler.min_[3]) + " | " + str(scaler.min_[4]) + " | ")
			self.utils.append_tutorial_text("| Data Max | " + str(scaler.data_max_[0]) + " | " + str(scaler.data_max_[1]) + " | " + str(scaler.data_max_[2]) + " | " + str(scaler.data_max_[3]) + " | " + str(scaler.data_max_[4]) + " | ")
			self.utils.append_tutorial_text("| Data Min | " + str(scaler.data_min_[0]) + " | " + str(scaler.data_min_[1]) + " | " + str(scaler.data_min_[2]) + " | " + str(scaler.data_min_[3]) + " | " + str(scaler.data_min_[4]) + " | ")
			self.utils.append_tutorial_text("| Data Range | " + str(scaler.data_range_[0]) + " | " + str(scaler.data_range_[1]) + " | " + str(scaler.data_range_[2]) + " | " + str(scaler.data_range_[3]) + " | " + str(scaler.data_range_[4]) + " | ")
			self.utils.append_tutorial_text("| Scale | " + str(scaler.scale_[0]) + " | " + str(scaler.scale_[1]) + " | " + str(scaler.scale_[2]) + " | " + str(scaler.scale_[3]) + " | " + str(scaler.scale_[4]) + " | \n\n")

	def scaler_helper(self, dataset):

		"""
		Scale dataset

		Parameters
		----------
		array : Numpy.ndarray
			

		Returns
		-------
		no_missing_samples: Int
			Number of missing samples in the 
		missing_days: Int

		"""

		scaler = MinMaxScaler()
		scaler = pickle.load(open(self.dir_path +  "/MinMaxScaler.sav", 'rb'))

		dataset = scaler.transform(dataset)

		return dataset


	def split_input_output(self, dataset):

		columns = ['datetime', 'time', 'weekday', 'station', 'free_bikes']

		x, y = dataset[:,range(0,len(columns) * n_in)], dataset[:,-n_out:] #dataset[:,n_out]
		
		x = x.reshape((x.shape[0], n_in, len(columns))) # (...,n_in,4)	

		return x,y

	def load_datasets(self):

		train_x = np.load(self.dir_path + '/data/train_x.npy')
		train_y = np.load(self.dir_path + '/data/train_y.npy')
		test_x = np.load(self.dir_path + '/data/test_x.npy')
		test_y = np.load(self.dir_path + '/data/test_y.npy')
		validation_x = np.load(self.dir_path + '/data/validation_x.npy')
		validation_y = np.load(self.dir_path + '/data/validation_y.npy')

		return train_x, train_y, validation_x, validation_y, test_x, test_y

	# Datasets utilizados para el problema:
	#	* [Train] Empleado para establecer los pesos de la red neuronal en el entrenamiento
	#	* [Validation] this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over then validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.
	#	* [Test] Used only for testing the final solution in order to confirm the actual predictive power of the network.
	def split_sets(self, training_size, validation_size, test_size):

		self.utils.append_tutorial_title("Split datasets")
		self.utils.append_tutorial_text("Dividing whole dataset into training " + str(training_size*100) + "%, validation " + str(validation_size*100) + "% & test " + str(test_size*100) + "%")

		values = np.load(self.dir_path + "/debug/supervised/FINAL.npy")

		if train_model == True:

			train_size_samples = int(len(values) * training_size)
			validation_size_samples = int(len(values) * validation_size)
			test_size_samples = int(len(values) * test_size)

			# As previously the data was stored in an array the stations were contiguous, shuffle them so when splitting
			# the datasets every station is spreaded across the array
			np.random.shuffle(values)

			train      = values[0:train_size_samples,:]
			validation = values[train_size_samples:train_size_samples + validation_size_samples, :]
			test       = values[train_size_samples + validation_size_samples:train_size_samples + validation_size_samples + test_size_samples, :]

			train_x, train_y           = self.split_input_output(train)
			validation_x, validation_y = self.split_input_output(validation)
			test_x, test_y             = self.split_input_output(test)

			np.save(self.dir_path + '/data/train_x.npy', train_x)
			np.save(self.dir_path + '/data/train_y.npy', train_y)
			np.save(self.dir_path + '/data/test_x.npy', test_x)
			np.save(self.dir_path + '/data/test_y.npy', test_y)
			np.save(self.dir_path + '/data/validation_x.npy', validation_x)
			np.save(self.dir_path + '/data/validation_y.npy', validation_y)


			print("Train X " + str(train_x.shape))
			print("Train Y " + str(train_y.shape))
			print("Test X " + str(test_x.shape))
			print("Test Y " + str(test_y.shape))
			print("Validation X " + str(validation_x.shape))
			print("Validation Y " + str(validation_y.shape))

			self.utils.append_tutorial_text("\n| Dataset | Percentage | Samples |")
			self.utils.append_tutorial_text("| --- | --- | --- |")
			self.utils.append_tutorial_text("| Training | " + str(training_size*100) + " | " + str(train_size_samples) + " | ")
			self.utils.append_tutorial_text("| Validation | " + str(validation_size*100) + " | " + str(validation_size_samples) + " | ")
			self.utils.append_tutorial_text("| Test | " + str(test_size*100) + " | " + str(test_size_samples) + " | \n\n")


	def prepare_tomorrow(self):

		"""
		Saves for each station an independent file with yesterday's availability to predict today's.


		Parameters
		----------
		array : Numpy.ndarray
			

		Returns
		-------
		no_missing_samples: Int
			Number of missing samples in the 
		missing_days: Int

		"""


		from datetime import datetime, timedelta
		from influxdb import InfluxDBClient
		import time


		client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')

		

		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
		
		self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

		dataset = pd.read_pickle(self.dir_path + '/data/Bilbao.pkl')


		self.utils.check_and_create("/debug/tomorrow")
		self.utils.check_and_create("/debug/yesterday")
		self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

		d = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))

		today = datetime.today()
		weekday = weekdays[(today - timedelta(days=1)).weekday()]
		yesterday = today - timedelta(days=1)
		yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
		today = today.strftime('%Y-%m-%dT00:00:00Z')

		

		print(yesterday)
		print(today)
		print(weekday)

		for station in self.list_of_stations:

			try:

				query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + today + '\' and station_name=\'' + str(station) + '\''


				

				dataset = pd.DataFrame(client.query(query, chunked=True).get_points())

				print(dataset)

				if dataset.size > 0:

					dataset.drop(['station_id'], axis=1, inplace=True)

					print("################################")
					print(dataset)

					# ['datetime', 'weekday', 'id', 'station', 'free_bikes', 'free_docks'] # Insert correct column names

					

					print(dataset)

					dataset['weekday'] = weekday
					dataset['datetime'] = (datetime.today() - timedelta(days=1)).timetuple().tm_yday


					times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,1]]

					dataset["time"] = times
					dataset = dataset[dataset['time'].isin(self.list_hours)]

					dataset = dataset[['datetime', 'time', 'weekday', 'station_name', 'value']]

					# print(dataset)


					dataset["value"] = pd.to_numeric(dataset["value"])

					out = self.encoder_helper(dataset)

					if out.shape[0] > 0:

						today_data = dataset[dataset['station_name'].isin(self.list_of_stations)] # TODO: Debugging
						today_data = today_data[today_data.datetime.isin([today])]
						today_data = today_data[today_data['station_name'].isin([station])]['value'].values

						out = self.encoder_helper(dataset)

						n_holes = self.find_holes(out)[0]


						out = self.fill_holes(out,n_holes)
						out = self.scaler_helper(out)
						out = out.reshape(1,144,5)

						print(out)



						np.save(self.dir_path + "/debug/yesterday/" + station + ".npy", out)
				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path (" + self.dir_path + "/debug/yesterday/" + station + ".npy)")

	# def prepare_tomorrow(self):

	# 	"""
	# 	Saves for each station an independent file with yesterday's availability to predict today's.


	# 	Parameters
	# 	----------
	# 	array : Numpy.ndarray
			

	# 	Returns
	# 	-------
	# 	no_missing_samples: Int
	# 		Number of missing samples in the 
	# 	missing_days: Int

	# 	"""

	# 	dataset = pd.read_pickle(self.dir_path + '/data/Bilbao.pkl')

	# 	print("READ DATASET LOCO")
	# 	print("---------------------")

	# 	print(dataset)

	# 	print("------------------------------------------------------------------------------------------------------------------------------")


	# 	self.utils.check_and_create("/debug/tomorrow")
	# 	self.utils.check_and_create("/debug/yesterday")
	# 	self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")

	# 	we = LabelEncoder()
 
	# 	he, we.classes_, se = self.load_encoders()

	# 	today = datetime.datetime.today()
	# 	today = today.strftime('%Y/%m/%d')
	# 	today = datetime.datetime.strptime(today, '%Y/%m/%d').timetuple().tm_yday		

	# 	yesterday = datetime.datetime.today() - datetime.timedelta(1)
	# 	yesterday = yesterday.strftime('%Y/%m/%d')
	# 	yesterday = datetime.datetime.strptime(yesterday, '%Y/%m/%d').timetuple().tm_yday

	# 	for station in self.list_of_stations:

	# 		try:
				


	# 			# Guardar los datos
	# 			dataset = dataset[dataset['station'].isin(self.list_of_stations)] # TODO: Debugging
	# 			out = dataset[dataset.datetime.isin([yesterday])]
	# 			out = out[out['station'].isin([station])] # ['free_bikes'].values

	# 			if out.shape[0] > 0:

	# 				# today_data = dataset[dataset['station'].isin(self.list_of_stations)] # TODO: Debugging
	# 				# today_data = today_data[today_data.datetime.isin([today])]
	# 				# today_data = today_data[today_data['station'].isin([station])]['free_bikes'].values

	# 				out = self.encoder_helper(out)

	# 				n_holes = self.find_holes(out)[0]


	# 				out = self.fill_holes(out,n_holes)
	# 				out = self.scaler_helper(out)
	# 				out = out.reshape(1,144,5)

	# 				print(out)
	# 				sadfas()


	# 				# np.save(self.dir_path + "/debug/today/" + station + '.npy', today_data)
	# 				np.save(self.dir_path + "/debug/yesterday/" + station + ".npy", out)
				
	# 		except (FileNotFoundError, IOError):
	# 			print("Wrong file or file path (" + self.dir_path + "/debug/yesterday/" + station + ".npy)")


	def prepare_today(self):

		"""
		Saves for each station an independent file with yesterday's availability to predict today's.


		Parameters
		----------
		array : Numpy.ndarray
			

		Returns
		-------
		no_missing_samples: Int
			Number of missing samples in the 
		missing_days: Int

		"""

		from influxdb import InfluxDBClient
		client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')

		import time

		current_time = time.strftime('%Y-%m-%dT00:00:00Z',time.localtime(time.time()))
		
		self.list_of_stations = self.utils.read_csv_as_list(self.dir_path + "/debug/utils/list_of_stations")
		
		today = datetime.datetime.today()
		today = today.strftime('%Y/%m/%d')
		today = datetime.datetime.strptime(today, '%Y/%m/%d').timetuple().tm_yday	
		
		for station in self.list_of_stations:


			try:

				dataset = pd.DataFrame(client.query('select * from bikes where time > \'' + str(current_time) + '\' and station_name=\'' + str(station) + '\'', chunked=True).get_points())

				print(dataset.size)

				if dataset.size > 0:

					dataset.drop(['station_id','station_name'], axis=1, inplace=True)

					print(dataset)

					times = [x.split("T")[1].replace('Z','')[:-3] for x in dataset.values[:,0]]

					dataset["time"] = times


					# dataset['datetime'] = [datetime.datetime.today().strptime(x, '%Y-%m-%dT%H:%M:%SZ').timetuple().tm_yday for x in dataset.values[:,0]]

					# Insertar en la columna time las hoas
					# dataset.insert(loc = 1, column = 'time', value = times)

					# Delete incorrectly sampled hours that don't match five minute intervals
					dataset = dataset[dataset['time'].isin(self.list_hours)]

					print(dataset)

					values = dataset.values

					out = [int(i) for i in values[:,1]]
					data = dict(zip(self.list_hours, out))


					print(data)


					jsonFile = open(self.dir_path + '/data/today/' + station + '.json', 'w+')
					jsonFile.write(json.dumps(data))
				jsonFile.close()


			except(pandas.errors.EmptyDataError):
				print("NO FOR STATION " + str(station))
