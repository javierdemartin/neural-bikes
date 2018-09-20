
#--------------------------------------------------------------------------------------------------------------------------------
# Initial Considerations
#--------------------------------------------------------------------------------------------------------------------------------
# Samples are collected on the server every five minutes (288 samples/day)

# Imports
#--------------------------------------------------------------------------------------------------------------------------------
# Libraries and custom classes

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import pandas.core.frame # read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
import datetime
from pandas import concat,DataFrame
import matplotlib
matplotlib.use('Agg') # Needed to plot some things when running on headless server
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import gc # Freeing memory
import csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, LSTM, Dropout, Activation
import pickle # Saving MinMaxScaler
from numpy import concatenate	
from keras.models import load_model


weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

hour_encoder    = LabelEncoder() # Encode columns that are not numbers
weekday_encoder = LabelEncoder() # Encode columns that are not numbers
station_encoder = LabelEncoder() # Encode columns that are not numbers

n_in = 288 # Number of previous samples used to feed the Neural Network
n_out = 288

scaler = MinMaxScaler(feature_range=(0,1)) # Normalize values

# Global Configuration Variables
# ------------------------------------------------------------------
# They might speed the script as they do non-vital things such as recollecting
# statistics

statistics_enabled = False
train_model = True

# Colorful prints in the terminal
class color:
	HEADER    = '\033[95m'
	blue      = '\033[94m'
	green     = '\033[92m'
	yellow    = '\033[93m'
	FAIL      = '\033[91m'
	ENDC      = '\033[0m'
	BOLD      = '\033[1m'

class Utils:	

	def print_smth(self,description, x):
	
		print("", color.yellow)
		print(description)
		print("----------------------------------------------------------------------------", color.ENDC)
		print(x)
		print(color.yellow, "----------------------------------------------------------------------------", color.ENDC)

	# Print an array with a description and its size
	def print_array(self, description, array):
		
		print("", color.yellow)
		print(description, " ", array.shape)
		print("----------------------------------------------------------------------------", color.ENDC)
		print(array)
		print(color.yellow, "----------------------------------------------------------------------------", color.ENDC)

	# Reads the list in the PATH and returns a LIST
	def read_csv_as_list(self, path):
		with open(path, 'r') as f:
			reader = csv.reader(f)
			your_list = list(reader)[0]


		return your_list

	# Checks if de current directory exists, if not it's created
	def check_and_create(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	# Save an array/list/... for future debugging
	def save_array_txt(self, path, array):

		# print("Saving " + str(path))

		# Guardar array con la función nativa de NumPy
		if type(array) is np.ndarray:
			np.savetxt(path, array, delimiter=',', fmt='%.0f')
		# Guardar LabelEncoders como una lista con cada elemento codificado en una linea
		elif type(array) is LabelEncoder:
			f = open(path, 'w' )
			for i in range(len(array.classes_)):
				f.write('{:>4}'.format(i) + " " + str(array.classes_[i]) + "\n")
			f.close()
		elif type(array) is DataFrame:
			array.to_csv(path, sep=',')
		elif type(array) is list:

			with open(path,"w") as f:
				wr = csv.writer(f,delimiter=",")
				wr.writerow(array)
		else:

			with open(path, 'w', newline='\n') as myfile:

				for element in array:
					myfile.write(str(element) + "\n")

class Plotter:

	def __init__(self):
		print("")

	def plot(self, data, xlabel, ylabel, title, path):

		min_y = min(data)
		max_y = max(data)

		plt.figure(figsize=(12, 9))
		ax = plt.subplot(111)
		ax = plt.axes(frameon=False)

		ax.spines["top"].set_visible(False)
		ax.spines["bottom"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()

		plt.xlabel(xlabel, color = 'black', fontsize = 14)
		plt.ylabel(ylabel, color = 'black', fontsize = 14)

		# lines  = plt.plot(data,  linestyle = '--', label = 'train', color = '#458DE1')
		lines  = plt.plot(data, label = 'train', color = '#458DE1')

		# plt.xticks(range(len(data)), hour_encoder.classes_) #, rotation='vertical')

		# start, end = ax.get_xlim()
		# ax.xaxis.set_ticks(np.arange(start, end, 0.125))

		plt.setp(lines, linewidth=3)

		plt.title(title,color="black") #, alpha=0.3)
		plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on") #, colors = 'silver')

		# plt.text(0.5,0.5, title, fontsize=20)

		plt.savefig(path + title + ".png", bbox_inches="tight")
		plt.close()

		# print("Plot saved " + str(path) + str(title))

	def two_plot(self, data_1, data_2, xlabel, ylabel, title, path, text = None):

		min_y = min(min(data_1), min(data_2))
		max_y = max(max(data_1), max(data_2))

		plt.figure(figsize=(12, 9))

		# plt.rc('font', weight='thin')
		# plt.rc('xtick.major', size=5, pad=7)
		# plt.rc('xtick', labelsize=12)

		ax = plt.subplot(111)

		# ax = plt.axes(frameon=False)

		# ax.spines["top"].set_visible(False)
		# ax.spines["bottom"].set_visible(False)
		# ax.spines["right"].set_visible(False)
		# ax.spines["left"].set_visible(False)

		# ax.get_xaxis().tick_bottom()
		# ax.get_yaxis().tick_left()

		plt.xlabel(xlabel, color = 'black', fontsize = 12)
		plt.ylabel(ylabel, color = 'black', fontsize = 12)

		# lines  = plt.plot(data,  linestyle = '--', label = 'train', color = '#458DE1')
		lines  = plt.plot(data_1, label = 'train', color = '#16a085')
		lines  += plt.plot(data_2, label = 'train', color = '#2980b9')

		# plt.xticks(range(len(data_1)), hour_encoder.classes_) #, rotation='vertical')

		# start, end = ax.get_xlim()
		# ax.xaxis.set_ticks(np.arange(start, end, 0.125))

		plt.setp(lines, linewidth=2)

		plt.title(title,color="black") #, alpha=0.3)
		plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on") #, colors = 'silver')
		# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode = "expand", ncol = 2, fancybox = False)


		if text is not None:

			import textwrap as tw

			fig_txt = tw.fill(tw.dedent(text), width=80)

			# The YAxis value is -0.07 to push the text down slightly
			plt.figtext(0.5, 0.0, fig_txt, horizontalalignment='center',fontsize=12, multialignment='right',
				# bbox=dict(boxstyle="round", facecolor='#D8D8D8',ec="0.5", pad=0.5, alpha=1), 
				# fontweight='bold'
				)

		plt.savefig(path + ".png", bbox_inches="tight")
		plt.close()

		# print("Plot saved " + str(path))

final_availability = []

class Data_mgmt:

	list_hours = []

	def __init__(self):
		
		self.utils = Utils()

		# print("INITED")
		# os.system("rm -rf debug/") # Delete previous debug records
		self.utils.check_and_create("debug/encoders")
		self.utils.check_and_create("debug/encoded_data")
		self.utils.check_and_create("debug/holes")
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

			# Reset indexes, so they match the current row
			dataset = dataset.reset_index(drop = True)
			# dataset = dataset.head(80000) 

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
			list_of_stations = ["IRALA"]

			for station in list_of_stations:

				print("Reading station " + str(station))

				station_read = np.load("debug/encoded_data/" + station + ".npy")



				station_read = station_read[1240:3270]

				no_missing_samples, missing_days = self.find_holes(station_read)
				self.fill_holes(station_read, no_missing_samples)
				


	def find_holes(self, station_read):

		# Read saved data for each station
		

		# Detect and count holes
		# ------------------------------------------------------------------------------------------
		current_row = station_read[0]

		no_missing_samples = 0
		missing_days = 0

		for i in range(1,station_read.shape[0]):

			# Both samples are of the same day
			if current_row[0] == station_read[i][0]:
				if (current_row[1] + 1) != station_read[i][1]:

					no_missing_samples += station_read[i][1] - current_row[1] - 1

					print("No hora " + str(current_row) + " and " + str(station_read[i]) + " (" + str(no_missing_samples) + ")") 

			# Días diferentes
			elif current_row[0] != station_read[i][0]:
				if current_row[1] != 287 or station_read[i][1] != 0:

					no_missing_samples += 287 - current_row[1] + station_read[i][1]
					
					print("Diferentes dias " + str(current_row) + " and " + str(station_read[i]) + " (" + str(no_missing_samples) + ")") 

					# Si faltan muestras de más de un día no rellenar, se han perdido datos
					if (current_row[0]+1) != station_read[i][0]:
						# Comprobar si empieza en 0

						missing_days += 1
						print("\tMas de un dia " + str(missing_days))					

			current_row = station_read[i]

		return no_missing_samples, missing_days

	def fill_holes(self, array, no_missing_samples):

		# for i in range(array.shape[0]):
		# 	print(str(i) + " " + str(array[i]))

		# Create the array with the final size
		rows = array.shape[0] + no_missing_samples
		columns = array.shape[1]

		print("Original shape is " + str(array.shape) + " filled array will be (" + str(rows) + ", "  + str(columns) + ") " + str(no_missing_samples) + " rows will be added")

		filled_array = np.zeros((rows, columns))

		index = 0

		current_row = array[0]

		for i in range(1, array.shape[0]):

			# Both samples are of the same day
			if current_row[0] == array[i][0]:

				if (current_row[1]) != (array[i][1]-1):
					print("\t" + str(i + index) + " Missing hour " + str(current_row[0]) + " " + str(current_row) + " and " + str(array[i]) + " " + str(array[i][1] - current_row[1]))
					
					for j in range(0, array[i][1] - current_row[1] - 1):
						# print("Insertando " + str(current_row) + " entre " + str(filled_array[i + index]) + " y " + str(filled_array[i + index + 1]))

						# filled_array[i + index] = [-1,-1,-1,-1,-1,-1] #current_row
						filled_array[i + index] = current_row  #[-1,-1,-1,-1,-1,-1] #current_row
						filled_array[i + index][1] += 1 + j
						# print(str(filled_array[i + index]) + ("*"))

						index += 1


						# filled_array[i + index] = array[i]  #[-1,-1,-1,-1,-1,-1] #current_row

					filled_array[i + index] = array[i]  #[-1,-1,-1,-1,-1,-1] #current_row
					# filled_array[i + index][1] += 1

					# index += 1
					
				else:
					
					filled_array[i + index] = array[i]
					# print(str(filled_array[i + index]))
			# Diferentes dias
			# elif current_row[0] != array[i][0]:

			# 	if current_row[1] != 287 and array[i][1] != 0:
			# 		print("diforontos " + str(current_row[0]) + " " + str(current_row) + " and " + str(array[i]) + str(287 - current_row[1] + array[i][1]))
			# 		index += 287 - current_row[1] + array[i][1]
			# 	elif current_row[1] == 287 and array[i][1] != 0:
			# 		print("mal dia " + str(current_row[0]) + " " + str(current_row) + " and " + str(array[i]))

			current_row = array[i]

		print("JAVO EL FINAL")

		for i in range(filled_array.shape[0]):
			print(str(i) + " " + str(filled_array[i]))



		# Iterar fila por fila pasando al array final las filas que no necesitan ser rellenadas


			# for station in list_of_stations:
				

				# index = [i for i,x in enumerate(list_of_stations) if x == station][0]

				# # print(station + " with index " + str(index))
				# # print(dataset)

				# values = dataset.loc[dataset['station'] == index].values #dataset[dataset['station'].isin([index])].values

				# self.utils.check_and_create("debug/holes/" + str(station))
				# self.utils.save_array_txt("debug/holes/" + str(station) + "/original_data_" + station, values)

				# number_to_insert = []

				# # Contar el número de huecos que hace falta insertar
				# for row in range(len(values) - 1):
				# 	if values[row][1] != 287:
				# 		# No coinciden los elementos que están pegados
				# 		if values[row][1] != (values[row + 1][1] - 1):

				# 			# Comprobar que los elementos son del mismo dia
				# 			if values[row][0] == values[row+1][0]:
				# 				number_to_insert.append(values[row + 1][1] - values[row][1] - 1)
				# 			else:
				# 				number_to_insert.append(288 - values[row][1]-1)

				# new_values = np.zeros(shape=(values.shape[0] + sum(number_to_insert), values.shape[1]))

				# print("SHAPE NEW " + str(new_values) + str(new_values.shape))

				# offset = 0

				# print("LEN VALS " + str(len(values)))

				# # Buscar huecos
				# for i in range(len(values)):
					
				# 	if i < (len(values)-1):
				# 		if values[i][1] != 287:

				# 			if values[i][1] != (values[i+1][1] - 1):
				# 				if values[i][0] == values[i+1][0]:
				# 					new_values[i + offset] = values[i]
				# 					offset += values[i+1][1] - values[i][1] - 1
				# 				else:
				# 					print("Diferente dia " + str(values[i]) + "->" + str(values[i+1]) + " offset " + str(287 - values[i][1] - 1))
				# 					offset += 288 - values[i][1] - 1
				# 			else:
				# 				new_values[i + offset] = values[i]
				# 		else:
				# 			new_values[i + offset] = values[i]
				# 	else:
				# 		new_values[i + offset] = values[i]

				# print(new_values)

				# # Guardar el array de datos generado con los huecos (ceros) en los que faltan datos por generar manualmente
				# self.utils.save_array_txt("debug/holes/" + str(station) + "/data_zeros_" + str(station), new_values)

				# # Rellenar los huecos con [0,0,0,0,0,0] con los elementos que tienen que ser, duplicando los que faltan
				# for i in range(len(new_values)):

				# 	# Comprobar si la fila actual tiene que ser rellenada con datos extra
				# 	if (new_values[i] == np.asarray([0,0,0,0,0,0])).all():

				# 		new_values[i] = np.array((new_values[i-1][0], new_values[i-1][1]+1, new_values[i-1][2], new_values[i-1][3],new_values[i-1][4], new_values[i-1][5]))

				# ranges = []

				# # Iterar de nuevo para eliminar los dias que no estén completos
				# # porque el servidor se apagó a mitad del día o ha habido otros fallos
				# for i in range(1,len(new_values)):
				# 	# Si empieza un nuevo día
				# 	if new_values[i][0] != new_values[i-1][0]:
						
				# 		# La hora del anterior son las 23:55 (287) -> El día está completo
				# 		# La hora del siguiente no son las 00:00 (0) -> Empezar a eliminar este día
				# 		if new_values[i-1][1] == 287 and new_values[i][1] != 0:
				# 			for w in range(i, i + 288 - int(new_values[i][1])):
				# 				ranges.append(w)

				# # Delete the days that are not complete
				# new_values = np.delete(new_values, ranges, axis=0)
				# list_of_processed_stations.append(new_values)
				# self.utils.save_array_txt("debug/holes/" + str(station) + "/holes_filled_" + str(station), new_values)


			# for st in list_of_processed_stations:
			# 	print("ESTACION NUMERO " + str(st))

		# return list_of_processed_stations

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

	

	
class Neural_Model:

	def __init__(self, train_x, train_y, test_x, test_y, epochs, validation_x, validation_y, batch_size):

		print("Inited NeuralModel class")
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.epochs = epochs
		self.model = self.create_model()

		self.validation_x = validation_x
		self.validation_y = validation_y

		print("Array shapes")
		print("Train X " + str(self.train_x.shape))
		print("Train Y " + str(self.train_y.shape))
		print("Test X " + str(self.test_x.shape))
		print("Test Y " + str(self.test_y.shape))
		print("Validation X " + str(self.validation_x.shape))
		print("Validation Y " + str(self.validation_y.shape))

		self.scaler = pickle.load(open("MinMaxScaler.sav", 'rb'))


		self.batch_size = batch_size

		self.utils = Utils()
		self.p = Plotter()

		self.hour_encoder = LabelEncoder()
		self.hour_encoder.classes_ = np.load('debug/encoders/hour_encoder.npy')

		self.weekday_encoder = LabelEncoder()
		self.weekday_encoder.classes_ = np.load('debug/encoders/weekday_encoder.npy')

		self.station_encoder = LabelEncoder()
		self.station_encoder.classes_ = np.load('debug/encoders/station_encoder.npy')
		
	def create_model(self):

		lstm_neurons = 200

		model = Sequential()
		model.add(LSTM(lstm_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), stateful=False, return_sequences=True))
		model.add(LSTM(lstm_neurons, return_sequences = True))
		model.add(LSTM(lstm_neurons))
		model.add(Dense(288))
		# model.add(Activation('softmax'))
		model.compile(loss='mae', optimizer='adam', metrics = ['mse', 'acc'])

		return model

	def fit_model(self):

		self.utils.check_and_create("plots/data/metrics")

		if train_model == True:

			history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = True)
			self.model.save('model/model.h5')  # creates a HDF5 file 'my_model.h5'

			print("\a Finished model training")

			title_plot = "Training & Validation Loss"
			title_path = "training_loss"
			note = "Model trained with " + str(self.epochs) + " epochs and batch size of " + str(self.batch_size)

			self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, "plots/" + title_path, note)

			title_plot = "Training & Validation Accuracy"
			title_path = "training_acc"

			self.p.two_plot(history.history['acc'], history.history['val_acc'], "Epoch", "accuracy", title_plot, "plots/" + title_path, note)		

			self.utils.save_array_txt("plots/data/metrics/acc", history.history['acc'])
			self.utils.save_array_txt("plots/data/metrics/val_acc", history.history['val_acc'])
			self.utils.save_array_txt("plots/data/metrics/loss", history.history['loss'])
			self.utils.save_array_txt("plots/data/metrics/val_loss", history.history['val_loss'])

			predicted = self.model.predict(self.test_x)[0]

			print("PREDICTED 1 SHAPE "+ " VALUES " + str(predicted))

			predicted = [x * 35 for x in predicted]
			predicted = [int(x) for x in predicted]

			self.p.plot(predicted, "xlabel", "ylabel", "Predicted_" + str(self.batch_size), "plots/")

			print("PREDICTED 1 SHAPE " + " VALUES " + str(predicted))

			print("SCALER MAXXX " + str(self.scaler.data_max_))
		else:

			self.model.load_weights("model/model.h5")
			print("Loaded model from disk")

		self.predict_test_set()

	def multiple_runs(self):

		p = Plotter()

		# collect data across multiple repeats
		train_loss = DataFrame()
		val_loss = DataFrame()

		train_acc = DataFrame()
		val_acc = DataFrame()

		pred = DataFrame()


		title = "Multiple runs with " + str(self.epochs) + " epochs and batch size of " + str(self.batch_size)

		for i in range(4):

			print("\a")
			self.model = self.create_model()

			# fit model
			history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=2, shuffle = False)
			self.model.save('model/model.h5')  # creates a HDF5 file 'my_model.h5'
			# story history
			train_loss[str(i)] = history.history['loss']
			val_loss[str(i)] = history.history['val_loss']

			train_acc[str(i)] = history.history['acc']
			val_acc[str(i)] = history.history['val_acc']			

			predicted = self.model.predict(self.test_x)[0]
			predicted = [x * 35 for x in predicted]
			predicted = [int(x) for x in predicted]

			pred[str(i)] = predicted

		# plot train and validation loss across multiple runs
		self.p.two_plot(train_loss, val_loss, "xlabel", "ylabel", title + "_loss", "plots/", note)

		self.p.two_plot(train_acc, val_acc, "xlabel", "ylabel", title + "_acc", "plots/", note)

		self.p.two_plot(self.test_y, pred, "Time", "Average Bike Availability", title + "_availability", "plots/", note)

	# Given the whole test set make multiple predictions to test the model
	def predict_test_set(self):

		self.utils.check_and_create("plots/data/")
		self.utils.check_and_create("plots/test_set_predictions/")

		print("Original test shape " + str(self.test_x.shape))

		average_error = np.zeros((288, len(self.test_x)))
		mierda = np.zeros((288, len(self.test_x)))

		print("DIMENSION AV " + str(len(self.test_x)))

		for i in range(len(self.test_x)):

			sample = self.test_x[i]
			out = self.test_y[i]

			s = sample.reshape((1, sample.shape[0], sample.shape[1]))
			predicted = self.model.predict(s)[0]

			inv_yhat = self.get_bikes_from_array(sample, predicted)

			print("INV_YHAT shape " + str(inv_yhat.shape))
			print(inv_yhat)
			predicted = inv_yhat[:,-1:].reshape((1, len(inv_yhat[:,-1:])))[0]

			print("Predicted values " + str(predicted.shape))
			print(str(predicted))

			inv_y = self.get_bikes_from_array(sample, out)

			dia = str(int(inv_y[0][0]))
			weekday = str(self.weekday_encoder.inverse_transform([int(inv_y[0][2])])[0])
			station = str(self.station_encoder.inverse_transform([int(inv_y[0][3])])[0])

			self.utils.check_and_create("plots/test_set_predictions/" + station)

			real = inv_y[:,-1:].reshape((1, len(inv_y[:,-1:])))[0]

			print("Plotting data")
			print(" - Real data " + str(len(real)))
			print(" - Pred data " + str(len(predicted)))

			title_path = station + "_" + dia + "_th_day"
			title_plot = "Prediction vs Real Data"

			note = "Model trained with " + str(self.epochs) + " epochs and batch size of " + str(self.batch_size)

			self.utils.check_and_create("plots/test_set_predictions/" + str(station) + "/")
			self.utils.save_array_txt("plots/test_set_predictions/" + str(station) + "/real_" + title_path, real)
			self.utils.save_array_txt("plots/test_set_predictions/" + str(station) + "/predicted_" + title_path, predicted)



			for j in range(288):

				diff = abs(real[j] - predicted[j])
				diff2 = abs(real[j] - final_availability[j])


				average_error[j][i] = diff # diff
				mierda[j][i] = diff2


			note = "Predicted vs real values for station " + station + " and day " + dia + " for a " + weekday

			self.p.two_plot(real, predicted, "Error (bikes)", "Time", title_plot, "plots/test_set_predictions/" + station + "/" + title_path, note)

		average_final = []
		average_final2 = []

		print("LISTA")
		for i in average_error:
			print(i)

		for i in range(288):

			print(str(i) + " >>>>>" + str(average_error[i]))

			print("Sum " + str(sum(average_error[i])) + " Len " + str(len(average_error[i])) + " Average " + str(sum(average_error[i])/len(average_error[i])))

			average_final.append(sum(average_error[i])/len(average_error[i]))
			average_final2.append(sum(mierda[i])/len(mierda[i]))

		note = "Averaged diference by hour of real and predicted value for all test set samples"

		self.p.two_plot(average_final,[0.0], "Time", "No. Bikes", "Averaged Error", "ERROR", note)	
		self.utils.save_array_txt("averaged_error", average_final)
		self.utils.save_array_txt("averaged_error2", average_final2)

	def get_bikes_from_array(self, test_x, test_y):

		test_y = test_y.reshape((len(test_y), 1))
		inv_y = concatenate((test_x[:,:5],test_y), axis=1)
		inv_y = self.scaler.inverse_transform(inv_y)

		free_bikes = inv_y[:,-1:].reshape((1, len(inv_y[:,-1:])))

		return inv_y

	# Taking the predicted and real sets of availability calculates the average error for each time interval
	def average_errors_per_hour(real, predicted):

		error_diff_per_hour = []

		for i in range(len(real)):

			error_diff_per_hour.append(predicted[i] - real[i])

		return error_diff_per_hour



