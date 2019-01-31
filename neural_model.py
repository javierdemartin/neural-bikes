import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed
from keras.optimizers import SGD
import pickle # Saving MinMaxScaler
from utils import Utils
from Plotter import Plotter
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate	
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from keras.utils.vis_utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


train_model  = True
len_day      = 144
lstm_neurons = 40

learning_rate = 0.1	
momentum = 0.8

class Neural_Model:

	def __init__(self):

		print("Inited NeuralModel class")

		self.utils = Utils()
		self.p     = Plotter()

		self.hour_encoder    = LabelEncoder()
		self.weekday_encoder = LabelEncoder()
		self.station_encoder = LabelEncoder()

		self.train_x = np.load('data/train_x.npy')
		self.train_y = np.load('data/train_y.npy')

		self.test_x = np.load('data/test_x.npy')
		self.test_y = np.load('data/test_y.npy')

		self.validation_x = np.load('data/validation_x.npy')
		self.validation_y = np.load('data/validation_y.npy')

		

		print("Array shapes")
		print("Train X " + str(self.train_x.shape))
		print("Train Y " + str(self.train_y.shape))
		print("Test X " + str(self.test_x.shape))
		print("Test Y " + str(self.test_y.shape))
		print("Validation X " + str(self.validation_x.shape))
		print("Validation Y " + str(self.validation_y.shape))

		self.scaler = pickle.load(open("MinMaxScaler.sav", 'rb'))
	
		self.batch_size = self.utils.read("batch_size")	
		self.epochs = self.utils.read("epochs")	

		self.plot_path = "plots/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/"
		self.utils.check_and_create(self.plot_path + "data/metrics/")
		self.utils.check_and_create(self.plot_path)
		self.utils.check_and_create("model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]))

		print("READIN")
		print(self.batch_size)

		self.hour_encoder.classes_, self.weekday_encoder.classes_, self.station_encoder.classes_ = self.load_encoders()
		self.model = self.create_model()


		self.utils.append_tutorial_title("Neural Network Training")
		self.utils.append_tutorial_text("![Model Shape](model/model.png)\n")
		self.utils.append_tutorial_text("* " + str(self.epochs) + " epochs")
		self.utils.append_tutorial_text("* " + str(self.batch_size) + " batch size\n")

	def load_encoders(self):
		print("> Loaded encoders ")
		return np.load('debug/encoders/hour_encoder.npy'), np.load('debug/encoders/weekday_encoder.npy'), np.load('debug/encoders/station_encoder.npy')
		
	def create_model(self):		
		

		model = Sequential()
		
		# model.add(LSTM(lstm_neurons, return_sequences=True)) # input_shape=(self.train_x.shape[1], self.train_x.shape[2]), stateful=False))
		model.add(LSTM(lstm_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences = True))
		model.add(Dropout(0.1))
		# model.add(LSTM(lstm_neurons, return_sequences = True))
		# model.add(Dropout(0.1))
		# model.add(LSTM(lstm_neurons, return_sequences = True))
		# model.add(Dropout(0.1))
		model.add(LSTM(lstm_neurons))
		model.add(Dropout(0.1))
		model.add(Dense(len_day))
		
		# sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


		model.compile(loss='mae', optimizer='adam', metrics = ['acc', 'mape', 'mse'])

		plot_model(model, to_file="model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + '/model.png', show_shapes=True, show_layer_names=True)

		return model

	def fit_model(self):


		if len(self.batch_size) > 0:

			for time in range(len(self.batch_size)):				

				note = "Model trained with " + str(self.epochs[0]) + " epochs and batch size of " + str(self.batch_size[0])
				# self.utils.check_and_create(plot_path)
				

				if train_model == True:

					history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size[0], epochs=self.epochs[0], validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = True) 
					# history = self.model.fit(self.train_x, self.train_y, batch_size=300, epochs=self.epochs[0], validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = True) 
					# history = self.model.fit(self.train_x, self.train_y, batch_size=30, epochs=self.epochs[0], validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = True) 
					
					self.model.save("model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")  # creates a HDF5 file 'my_model.h5'
				

					print("\a Finished model training")

					title_plot = "Training & Validation Loss"
					title_path = "training_loss"
					

					self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, self.plot_path + title_path, note, "Loss", "Validation Loss")

					title_plot = "Training & Validation Accuracy"
					title_path = "training_acc"

					self.p.two_plot(history.history['acc'], history.history['val_acc'], "Epoch", "accuracy", title_plot, self.plot_path + title_path, note, "Accuracy", "Validation Accuracy")		

					title_path = "training_mape"
					title_plot = "Training & Validation MAPE"

					self.p.two_plot(history.history['mean_absolute_percentage_error'], history.history['val_mean_absolute_percentage_error'], "Epoch", "accuracy", title_plot, self.plot_path + title_path, note, "MAPE", "Validation MAPE")		

					title_path = "training_mse"
					title_plot = "Training & Validation MSE"

					self.p.two_plot(history.history['mean_squared_error'], history.history['val_mean_squared_error'], "Epoch", "accuracy", title_plot, self.plot_path + title_path, note, "MSE", "Validation MSE")		

					self.utils.save_array_txt(self.plot_path + "data/metrics/acc", history.history['acc'])
					self.utils.save_array_txt(self.plot_path + "data/metrics/val_acc", history.history['val_acc'])
					self.utils.save_array_txt(self.plot_path + "data/metrics/loss", history.history['loss'])
					self.utils.save_array_txt(self.plot_path + "data/metrics/val_loss", history.history['val_loss'])

					self.utils.append_tutorial_text("![Training Acc](plots/training_acc.png)\n")
					self.utils.append_tutorial_text("![Training Loss](plots/training_loss.png)\n")
					self.utils.append_tutorial_text("![Training MAPE](plots/training_mape.png)\n")
					self.utils.append_tutorial_text("![Training MSE](plots/training_mse.png)\n")

				else:

					self.model = self.create_model()
					self.model.load_weights("model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")
					print("Loaded model from disk")

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

					self.p.two_plot(real_data[:,4].tolist(), list(map(int, inv_yhat[:,4].tolist())), "Time", "Free Bikes", "Prediction for " + station_name, self.plot_path + str(i), note, "Real", "Predicted")

					# self.p.two_plot(
					# 	real_data[:,4].tolist() +  list(map(int, inv_yhat[:,4].tolist())), 
					# 	real_data[:,4].tolist() +  next_day[:,4].tolist(), 
					# 	"Time", "Free Bikes", "Prediction for " + station_name, "plots/" + str(i) + "_WTF", note, "Real", "Predicted")

				self.utils.append_tutorial_text("![Prediction Sample 1](plots/1.png)\n")
				self.utils.append_tutorial_text("![Prediction Sample 2](plots/2.png)\n")
				self.utils.append_tutorial_text("![Prediction Sample 3](plots/3.png)\n")
				self.utils.append_tutorial_text("More prediction samples in [plots/](https://github.com/javierdemartin/neural-bikes/tree/master/plots).")


				self.batch_size.pop(0)
				self.epochs.pop(0)

	def tomorrow(self):

		self.model = self.create_model()
		self.model.load_weights("model/" + str(self.epochs[0]) + "_" + str(self.batch_size[0]) + "/model.h5")
		print("Loaded model from disk")

		self.utils.check_and_create("plots/tomorrow")
		self.list_of_stations = self.utils.read_csv_as_list("debug/utils/list_of_stations")

		for station in self.list_of_stations:

			try:
				dataset = np.load('debug/tomorrow/' + str(station) + ".npy")
				print("Loaded " + str(station) + " - " + str(dataset.shape))

				p = self.model.predict(dataset)

				p = p.reshape((144, 1))
				dataset = dataset.reshape((dataset.shape[1], dataset.shape[2]))
				dataset = self.scaler.inverse_transform(dataset)
				
				print("PREDICTION " + str(p.shape))
				print(p)

				

				print("DATASET " + str(dataset.shape))
				print(dataset)

				weekday = int(dataset[0][2])

				if weekday is 6:
					weekday = 0
				else:
					weekday += 1

				weekday = self.weekday_encoder.inverse_transform(weekday)

				print("WDAY " + str(weekday))


				inv_yhat = concatenate((dataset[:,: dataset.shape[1] - 1], p), axis=1)

				# print("INV_YHAT " + str(inv_yhat.shape))
				# print(inv_yhat)


				# p = self.scaler.inverse_transform(inv_yhat)
				# print(p)

				# columns=['datetime', 'time', 'weekday', 'station', 'free_bikes']

				# weekday = ""

				# if int(inv_yhat[:,2][0]) is 6:
				# 	weekday = self.weekday_encoder.inverse_transform(int(0))
				# else:
				# 	weekday = self.weekday_encoder.inverse_transform(int(inv_yhat[:,2][0] + 1))

				print(dataset[:,-1])


				self.p.two_plot(dataset[:,-1], [0], "xlabel", "ylabel", str(station + " for " + weekday), "plots/tomorrow/" + station, text = None, line_1 = None, line_2 = None)



				

				
			except (FileNotFoundError, IOError):
				print("Wrong file or file path")

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
			history = self.model.fit(self.train_x, self.train_y, batch_size=30, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=2, shuffle = False)
			history = self.model.fit(self.train_x, self.train_y, batch_size=300, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=2, shuffle = False)
			self.model.save('model/model.h5')  # creates a HDF5 file 'my_model.h5'
			# story history
			train_loss[str(i)] = history.history['loss']
			val_loss[str(i)] = history.history['val_loss']

			train_acc[str(i)] = history.history['acc']
			val_acc[str(i)] = history.history['val_acc']			

			predicted = self.model.predict(self.test_x)[0]
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

		average_error = np.zeros((len_day, len(self.test_x)))
		mierda = np.zeros((len_day, len(self.test_x)))

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

			# predicted = [x / 35 for x in predicted]

			print("Predicted values " ) #+ str(predicted.shape))
			print(str(predicted))

			inv_y = self.get_bikes_from_array(sample, out)

			print("JAVOOO")
			print(self.scaler.inverse_transform(inv_y))

			print("INVERSE PROBLEMA")
			print(inv_y[0])

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
			

			for j in range(len_day):

				diff = abs(real[j] - predicted[j])
				diff2 = abs(real[j] - predicted[j])


				average_error[j][i] = diff # diff
				mierda[j][i] = diff2


			note = "Predicted vs real values for station " + station + " and day " + dia + " for a " + weekday

			self.p.two_plot(real, predicted, "Error (bikes)", "Time", title_plot, "plots/test_set_predictions/" + station + "/" + title_path, note)



	def get_bikes_from_array(self, test_x, test_y):

		test_y = test_y.reshape((len(test_y), 1))
		inv_y = concatenate((test_x[:,:5],test_y), axis=1)
		inv_y = self.scaler.inverse_transform(inv_y)

		free_bikes = inv_y[:,-1:].reshape((1, len(inv_y[:,-1:])))

		return inv_y



