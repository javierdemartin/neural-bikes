from utils import Utils
from Plotter import Plotter
import pandas.core.frame # read_csv
import datetime
from color import color
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

import numpy as np

train_model = True
statistics_enabled = False

weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers
station_encoder = LabelEncoder() # Encode columns that are not numbers


class Data_mgmt:

	list_hours = []

	def __init__(self):
		
		self.utils = Utils()

		# print("INITED")
		# os.system("rm -rf debug/") # Delete previous debug records
		self.utils.check_and_create("debug/encoders")
		self.utils.check_and_create("debug/encoded_data")
		self.utils.check_and_create("debug/filled")
		self.utils.check_and_create("debug/supervised")
		self.utils.check_and_create("stats/stations")
		self.utils.check_and_create("plots/")
		self.utils.check_and_create("debug/utils/")
		self.utils.check_and_create("model/")

		self.plotter = Plotter()
		
		self.list_hours = self.utils.read_csv_as_list('list_hours.txt')


	def read_dataset(self):

		dataset = None

		if train_model == True:

			dataset = pandas.read_csv('data/Bilbao.txt')
			dataset.columns = ['datetime', 'weekday', 'id', 'station', 'free_docks', 'free_bikes'] # Insert correct column names
			dataset.drop(dataset.columns[[2]], axis = 1, inplace = True) # Remove ID of the sation and free docks

			values = dataset.values

			times = [x.split(" ")[1] for x in values[:,0]]

			dataset['datetime'] = [datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timetuple().tm_yday for x in values[:,0]]

			dataset.insert(loc = 1, column = 'time', value = times)

			# Delete incorrectly sampled hours that don't match five minute intervals
			dataset = dataset[dataset['time'].isin(self.list_hours)]

			# dataset = dataset[dataset['station'].isin(["ZUNZUNEGI"])] # TODO: Debugging

			dataset = dataset.reset_index(drop = True) # Reset indexes, so they match the current row
			# dataset = dataset.head(180000)

			print("Read dataset (" + str(dataset.shape[0]) + " lines and " + str(dataset.shape[1]) + " columns)")
			self.utils.print_array("Dataset with unwanted columns removed", dataset.head(15))
		else:
			print("Not reading dataset")

		return dataset

	# Gets a list of values and returns the list of stations
	def get_list_of_stations(self, array):

		if array is not None:

			array = np.asarray(array)

			print(list(np.unique(array)))

			self.utils.save_array_txt('debug/utils/list_of_stations', list(np.unique(array)))

			return list(np.unique(array))

		elif array is None:
			return None

	# Codificar las columnas seleccionadas con LabelEncoder
	def encode_data(self, dataset):

		if train_model == True:

			list_of_stations = self.get_list_of_stations(dataset.values[:,3])


			hour_encoder.fit(self.list_hours)
			weekday_encoder.fit(weekdays)
			station_encoder.fit(list_of_stations)

			# Save as readable text to check
			self.utils.save_array_txt("debug/encoders/hour_encoder", hour_encoder)       
			self.utils.save_array_txt("debug/encoders/weekday_encoder", weekday_encoder)
			self.utils.save_array_txt("debug/encoders/station_encoder", station_encoder)

			# Save as a numpy array
			np.save('debug/encoders/hour_encoder.npy', hour_encoder.classes_)
			np.save('debug/encoders/weekday_encoder.npy', weekday_encoder.classes_)
			np.save('debug/encoders/station_encoder.npy', station_encoder.classes_)


			#TODO: Debugging, only doing it for one station	
			# list_of_stations = ["ZUNZUNEGI"]

			values = dataset.values		

			values[:,1] = hour_encoder.transform(values[:,1])     # Encode HOUR as an integer value
			values[:,2] = weekday_encoder.transform(values[:,2])  # Encode WEEKDAY as an integer value
			values[:,3] = station_encoder.transform(values[:,3])  # Encode STATION as an integer value

			dataset = pd.DataFrame(data=values, columns=['datetime', 'time', 'weekday', 'station', 'free_docks', 'free_bikes'])

			# Save encoded data for each station to an independent file as a .npy file
			for st in list_of_stations:

				xxx = station_encoder.transform([st])[0]

				file_name = "debug/encoded_data/" + st + ".npy"

				np.save(file_name, dataset[dataset['station'].isin([xxx])].reset_index(drop = True).values)

			self.utils.print_array("Encoded dataset", dataset.head(15))

			return dataset, list_of_stations

		elif train_model == False:
			return None, None

	def stats_for_station(self):

		if statistics_enabled == True:

			i = 0
			k = 0

			list_of_stations = self.utils.read_csv_as_list("debug/utils/list_of_stations")

			global_average = np.empty([31,288])

			print("GLOBAL SHAPE " + str(global_average.shape))

			for station in list_of_stations:

				station_read = np.load("debug/encoded_data/" + station + ".npy")

				# print("Read station " + station)
				# print(station_read)

				station_read = pd.DataFrame(station_read, columns = ['datetime', 'time', 'weekday', 'station', 'free_docks', 'free_bikes'])

				averaged_data = []

				# AVERAGE TOTAL
				for i in range(288):

					# print("La media de las " + str(i))
					filtered_by_hour = station_read[station_read['time'].isin([i])].values
					averaged = int(sum(filtered_by_hour[:,5]) / len(filtered_by_hour))
					averaged_data.append(averaged)

				self.plotter.plot(averaged_data, "Time", "Average Bike Availability", str(station) + "_average_availability", "stats/stations/")
				self.utils.save_array_txt("stats/stations/" + str(station) + "_average_availability", averaged_data)



	# Calls `series_to_supervised` and then returns a list of arrays, in each one are the values for each station
	def supervised_learning(self, data):

		if train_model == True:

			index = 0

			columns = ['datetime', 'time', 'weekday', 'station', 'free_docks', 'free_bikes']

			columns_to_drop = []

			# Deleting columns that are not going to be predicted, tener en cuenta eliminar los time-steps posteriores enteros
			for time_step in range(n_out):

				position = (len(columns)) * (n_in + time_step)

				for j in range(position, position + len(columns) - 1):
					columns_to_drop.append(j)

			rows_to_drop = []

			print(data[0])

			left = int(288 - data[0][data[0].shape[0]-1][1])

			print("LEEFFFFT " + str(left) + " -- - - " + str(data[0][data[0].shape[0]-1][1]))

			for i in range(int((len(data[0]))/n_in)-2):
				for j in range(i*n_in + 1, (i+1)*n_in):
					rows_to_drop.append(j)

			supervised_data = []
			
			for st in data:

				print("Calling series to supervised for station " + str(st.shape))
				print(st)
				print("---------------------------------------------------------------")

				a = self.series_to_supervised(columns, st, n_in, n_out)

				# a = a.drop(columns_to_drop, axis=1, inplace=True)

				# Eliminar las columnas que no se quieren
				a.drop(a.columns[columns_to_drop], axis=1, inplace = True)  # df.columns is zero-based pd.Index 
				a = a.reset_index(drop = True)

				print("Shape before drop rows " + str(a.shape))
				# Eliminar cada n_in filas asi no se repiten los datos
				a.drop(a.index[rows_to_drop], inplace = True)

				n = 288 - left + 2

				a.drop(a.tail(n).index,inplace=True)

				a = a.reset_index(drop = True)

				print("DROPPPED " + str(a.columns) + " shaped " + str(a.values.shape))
				print(a.values)

				self.utils.save_array_txt("debug/supervised/supervised_for " + str(index), a)

				supervised_data.append(a)

				index+=1

			# Hasta aqui SUPERVISED_DATA es una lista con arrays dentro
			flattened = self.flatten_list_supervised(supervised_data)

			return flattened

		elif train_model == False:

			return None

	def flatten_list_supervised(self, data):

		flattened_data = np.array(data[0].values)

		print("FLATTENED FIRST" + str(flattened_data))

		for i in range(1, len(data)):
			flattened_data = np.append(flattened_data, data[i].values, axis=0)

		print("FLATTENED FINAL" + str(flattened_data))

		self.utils.save_array_txt("debug/supervised/final", flattened_data)

		return flattened_data

	def series_to_supervised(self, columns, data, n_in=1, n_out=1, dropnan=True):

		print("COLUMNAS " + str(columns))

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

		del dataset
		del cols
		gc.collect()

		print("Droppin NAN")

		if dropnan:
			agg.dropna(inplace=True)

		self.utils.print_array("Reframed dataset after converting series to supervised", agg.head())

		return agg

	# Iterates through every station file looking for holes
	def iterate(self):

		if train_model == True:

			list_of_stations = self.utils.read_csv_as_list("debug/utils/list_of_stations")
			list_of_stations = ["ZUNZUNEGI"]

			for station in list_of_stations:

				print("Reading station " + str(station))

				station_read = np.load("debug/encoded_data/" + station + ".npy")

				no_missing_samples, missing_days = self.find_holes(station_read)
				filled_array = self.fill_holes(station_read, no_missing_samples)

				# self.utils.check_and_create("debug/filled")

				self.utils.save_array_txt("debug/filled/" + station + "_filled", filled_array)

				np.save('debug/filled/' + station + '_filled.npy', filled_array)				


	def find_holes(self, array):

		print_debug = True

		print("Find holes in " + str(array.shape))
		print("##################################################################")

		

		# Detect and count holes
		# ------------------------------------------------------------------------------------------
		current_row = array[0]

		no_missing_samples = 0
		missing_days = 0

		# Iterar por todas las muestras
		for i in range(1,array.shape[0]):

			# Both samples are of the same day
			if current_row[0] == array[i][0]:
				if (current_row[1]) != (array[i][1]-1):

					if (array[i][1] - current_row[1] - 1) > 0:

						no_missing_samples += array[i][1] - current_row[1] - 1

						if print_debug: print("↳  (" + str(array[i][1] - current_row[1] - 1) + ") ⟶ " + str(current_row) + " ⟷ " + str(array[i]))

					# Negative difference, probably the time of the server was changed, need to delete some samples
					else:
						if print_debug: print("↳ ✘⁃ (" + str(array[i][1] - current_row[1] - 1) + ") ⟶ " + str(current_row) + " ⟷ " + str(array[i]))

						no_missing_samples -= array[i][1] - current_row[1] - 1

					# print("No hora " + str(current_row) + " and " + str(station_read[i]) + " (" + str(no_missing_samples) + ")") 

			# Días diferentes
			elif current_row[0] != array[i][0]:

				if current_row[1] != 287 or array[i][1] != 0:
					
					# print("Diferentes dias " + str(current_row) + " and " + str(station_read[i]) + " (" + str(no_missing_samples) + ")") 

					# Si faltan muestras de más de un día no rellenar, se han perdido datos
					if (current_row[0]+1) != array[i][0]:

						missing_days += array[i][0] - current_row[0]
						no_missing_samples += (288 - current_row[1]) + array[i][1]
						if print_debug: print(color.HEADER + "➜ " + str(array[i][0] - current_row[0]) + " dias perdidos " + str(current_row) + " y " + str(array[i]) + " added " + str((288 - current_row[1]) + array[i][1]) + color.ENDC)					

					else:
						no_missing_samples += 287 - current_row[1] + array[i][1]

						if print_debug: print(color.green + "↳ (" + str(287 - current_row[1] + array[i][1]) + ") ⟶ " + str(current_row) + " ⟷ " + str(array[i]) + color.ENDC)

			current_row = array[i]

		print("##################################################################")

		if array[array.shape[0] - 1][1] != 287:
			print("ORIG " + str(no_missing_samples))

			# no_missing_samples -= array[array.shape[0] - 1][1] 

			print("Restado " + str(no_missing_samples))



		print("Final missing " + str(no_missing_samples) + " samples")
		print("First sample was " + str(array[0]) + " and last " + str(array[array.shape[0] - 1]))

		return no_missing_samples, missing_days

	def fill_holes(self, array, no_missing_samples):

		print_debug = False

		# Create the array with the final size
		rows = array.shape[0] + no_missing_samples
		columns = array.shape[1]

		if print_debug: print("Original shape is " + str(array.shape) + " filled array will be (" + str(rows) + ", "  + str(columns) + ") " + str(no_missing_samples) + " rows will be added")
		if print_debug: print("-----------------------------------------------------------------------------")


		filled_array = np.zeros((rows, columns))

		index = 0

		current_row = array[0]
		filled_array[0] = current_row

		if print_debug: print("First element of the array to fill is " + str(array[0]) + " and last is " + str(array[array.shape[0] - 1]))

		for i in range(1, array.shape[0]):

			# Both samples are of the same day
			if current_row[0] == array[i][0]:

				# No coincide normalmente
				if (current_row[1]) != (array[i][1]-1):
					

					# diferencia positiva entre muestras, rellenar huecos duplicando valores
					if (array[i][1] - current_row[1] - 1) > 0:

						missing_samples = array[i][1] - current_row[1] - 1

						if print_debug: print("Missing samples " + str(array[i]) + " - " + str(current_row) + " (" + str(missing_samples) + ")")

						# Iterar por toda la cantidad de muestras que faltan
						for j in range(0, missing_samples):

							if print_debug: print("  ↳ " + str(j+1))

							filled_array[i + index] = current_row 
							filled_array[i + index][1] += 1 + j

							index += 1

						if print_debug: print("↳ (" + str(i + index) + ") ⟶ " + str(array[i]))

						filled_array[i + index] = array[i]  

					# Diferencia negativa, cambio de hora probable
					else: 

						print("Diferencia negativa " + str(array[i]) + " y " + str(current_row))
						print("Indice actual " + str(index) + " despues " + str(array[i][1] - current_row[1] - 1))

						for j in range(abs(array[i][1] - current_row[1] - 1)):

							filled_array[i + index] = current_row
							index -= 1
							

						# TODO: MIRA ESTO HOSTIA JAVI
					
				# Dentro del mismo dia muestras okay
				else:

					if print_debug: print("↳ ✔︎ (" + str(i + index) + ") ⟶ " + str(array[i]))

					filled_array[i + index] = array[i]

			# Diferentes dias
			elif current_row[0] != array[i][0]:

				# Inserta posibles muestras perdidas a las 00:00, sólo probado con una, faltaría si existe más de una pérdida
				if array[i][1] != 0 or current_row[1] != 287:

					# Comprobar que son días seguidos, no rellenar más de un día
					if array[i][0] == (current_row[0] + 1):

						print("Missing samples " + str(array[i]) + " - " + str(current_row) + " (" + str(missing_samples) + ")")

						filled_array[i + index] = array[i]
						filled_array[i + index][1] -= 1 

						index += 1

						filled_array[i + index] = array[i]

					# Hay diferencia de más de un dia incompleto, eliminar tanto el inicial como el final ya que los dos estan incompletos
					else: 
						

						missing_samples = (288 - current_row[1]) + array[i][1]

						# index += missing_samples

						print("Incomplete days INITIAL " + str(current_row) + " - " + str(array[i]) + " (" + str(missing_samples) + ")")

						# Rellenar muestras trailing que faltan
						for j in range(0,288 - current_row[1] - 1):

							if print_debug: print(color.yellow + "  ↳ " + str(j+1) + color.ENDC)

							filled_array[i + index] = current_row 
							filled_array[i + index][1] += 1 + j

							index += 1



						# Rellenar muestras iniciales que falten

						for j in range(0, array[i][1]):

							if print_debug: print(color.yellow + "  ↳ " + str(j+1) + color.ENDC)

							filled_array[i + index] = array[i] 
							filled_array[i + index][1] = j

							index += 1

						filled_array[i + index] = array[i]
						filled_array[i + index][1] = array[i][1]


				# Insertar las muestras a las 00:00 de forma normal
				else: 
					filled_array[i + index] = array[i]
					# index += 1


			current_row = array[i]

		aux = 0

		for i in range(len(filled_array)):

			print(str(aux) + " " + str(filled_array[i]))
			aux += 1

			if aux == 288:
				aux = 0


		return filled_array


	def scale_dataset(self, dataset):

		if train_model == True:
			# print("Scaled dataset pre shape" + str(dataset.shape))
			aux_array = dataset[0]

			print("AUX ARRAY")
			print(aux_array)

			for i in range(1, len(dataset)):
				aux_array = np.concatenate((aux_array, dataset[i]))

			print("Apepnded array final " + str(aux_array.shape))
			print(aux_array)

			scaler.fit_transform(aux_array)

			print("Scaler data ")		
			print("min " + str(scaler.min_))
			print("scale " + str(scaler.scale_))
			print("data_min_ " + str(scaler.data_min_))
			print("data_max_ " + str(scaler.data_max_))
			print("data_range_ " + str(scaler.data_range_))

			print("RECEIVED DATASET ")
			print(dataset)

			for i in range(len(dataset)):
				dataset[i] = scaler.transform(dataset[i])
				# print("SCALED ")
				# print(dataset[i])

			pickle.dump(scaler, open("MinMaxScaler.sav", 'wb'))

			return dataset

		elif train_model == False:
			return None



	def split_input_output(self, dataset):

		columns = ['datetime', 'time', 'weekday', 'station', 'free_docks', 'free_bikes']
		print("Initial dataset shape " + str(dataset.shape))

		x, y           = dataset[:,range(0,len(columns) * n_in)], dataset[:,-n_out:] #dataset[:,n_out]

		self.utils.save_array_txt("debug/x", x)
		self.utils.save_array_txt("debug/y", y)

		x       = x.reshape((x.shape[0], n_in, len(columns))) # (...,n_in,4)	

		return x,y

	def load_datasets(self):

		train_x = np.load('train_x.npy')
		train_y = np.load('train_y.npy')
		test_x = np.load('test_x.npy')
		test_y = np.load('test_y.npy')
		validation_x = np.load('validation_x.npy')
		validation_y = np.load('validation_y.npy')

		return train_x, train_y, validation_x, validation_y, test_x, test_y

	# Datasets utilizados para el problema:
	#	* [Train] Empleado para establecer los pesos de la red neuronal en el entrenamiento
	#	* [Validation] this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over then validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.
	#	* [Test] Used only for testing the final solution in order to confirm the actual predictive power of the network.
	def split_sets(self, values, training_size, validation_size, test_size):


		if train_model == True:

			print("DATASET SHAPE" + " TYPE " + str(type(values)))
			print("DATASET[0] SHAPE" + str(values[0].shape) + " TYPE " + str(type(values[0])))

			train_size_samples = int(len(values) * training_size)
			validation_size_samples = int(len(values) * validation_size)
			test_size_samples = int(len(values) * test_size)

			# As previously the data was stored in an array the stations were contiguous, shuffle them so when splitting
			# the datasets every station is spreaded across the array
			np.random.shuffle(values)

			print("RANDOM VALUES ")

			print(values)

			# self.utils.save_array_txt("debug/supervised/final_shuffled", values)

			print("Dataset size is " + str(len(values)) + " samples")
			print("\t Train Size (" + str(training_size) + "%) is " + str(train_size_samples) + " samples")
			print("\t Validation Size (" + str(validation_size) + "%) is " + str(validation_size_samples) + " samples")
			print("\t Test Size (" + str(test_size) + "%) is " + str(test_size_samples) + " samples")

			train = values[0:train_size_samples,:]
			validation = values[train_size_samples:train_size_samples + validation_size_samples, :]
			test = values[train_size_samples + validation_size_samples:train_size_samples + validation_size_samples + test_size_samples, :]

			train_x, train_y           = self.split_input_output(train)
			validation_x, validation_y = self.split_input_output(validation)
			test_x, test_y             = self.split_input_output(test)

			np.save('train_x', train_x)
			np.save('train_y', train_y)
			np.save('test_x', test_x)
			np.save('test_y', test_y)
			np.save('validation_x', validation_x)
			np.save('validation_y', validation_y)

		else:
			train_x, train_y, validation_x, validation_y, test_x, test_y = self.load_datasets()


		return train_x, train_y, validation_x, validation_y, test_x, test_y

	

	