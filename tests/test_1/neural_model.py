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

train_model  = True
len_day      = 144
lstm_neurons = 40

learning_rate = 0.1	
momentum = 0.8

class Neural_Model:

	def __init__(self, epochs, batch_size):

		print("Inited NeuralModel class")
		self.train_x = np.load('train_x.npy')
		self.train_y = np.load('train_y.npy')

		self.test_x = np.load('test_x.npy')
		self.test_y = np.load('test_y.npy')

		self.validation_x = np.load('validation_x.npy')
		self.validation_y = np.load('validation_y.npy')

		self.epochs = epochs

		print("Array shapes")
		print("Train X " + str(self.train_x.shape))
		print("Train Y " + str(self.train_y.shape))
		print("Test X " + str(self.test_x.shape))
		print("Test Y " + str(self.test_y.shape))
		print("Validation X " + str(self.validation_x.shape))
		print("Validation Y " + str(self.validation_y.shape))

		self.scaler = pickle.load(open("MinMaxScaler.sav", 'rb'))

		# Train on batch size
		if batch_size == 'b' :
			self.batch_size = self.train_x.shape[0]
		elif batch_size == 'hb':
			self.batch_size = int(self.train_x.shape[0]/2)
		else :
			self.batch_size = batch_size		

		self.utils = Utils()
		self.p = Plotter()

		self.hour_encoder    = LabelEncoder()
		self.weekday_encoder = LabelEncoder()
		self.station_encoder = LabelEncoder()

		self.hour_encoder.classes_, self.weekday_encoder.classes_, self.station_encoder.classes_ = self.load_encoders()
		self.model = self.create_model()

	def load_encoders(self):
		print("> Loaded encoders ")
		return np.load('debug/encoders/hour_encoder.npy'), np.load('debug/encoders/weekday_encoder.npy'), np.load('debug/encoders/station_encoder.npy')
		
	def create_model(self):		



		decay_rate = learning_rate / self.epochs
		

		model = Sequential()
		
		# model.add(LSTM(lstm_neurons, return_sequences=True)) # input_shape=(self.train_x.shape[1], self.train_x.shape[2]), stateful=False))
		model.add(LSTM(lstm_neurons, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences = True))
		model.add(Dropout(0.1))
		model.add(LSTM(lstm_neurons))
		model.add(Dropout(0.1))
		model.add(Dense(len_day))

		
		sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


		model.compile(loss='mse', optimizer=sgd, metrics = ['acc', 'mape', 'mse'])

		return model

	def fit_model(self):

		decay_rate = learning_rate / self.epochs

		note = "Model trained with " + str(self.epochs) + " epochs and batch size of " + str(self.batch_size) + "\nLearning Rate " + str(learning_rate) + ", momentum " + str(momentum) + " & decay rate " + str(decay_rate)
		self.utils.check_and_create("plots/data/metrics")

		if train_model == True:

			# for i in range(0,len(self.train_x[0])):
			# 	print(str(self.train_x[1][:,4][i]) + " - " + str(self.train_y[0][i]))

			history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.validation_x, self.validation_y), verbose=1, shuffle = False)
			self.model.save('model/model.h5')  # creates a HDF5 file 'my_model.h5'

			print("\a Finished model training")

			title_plot = "Training & Validation Loss"
			title_path = "training_loss"
			

			self.p.two_plot(history.history['loss'], history.history['val_loss'], "Epoch", "Loss", title_plot, "plots/" + title_path, note, "Loss", "Validation Loss")

			title_plot = "Training & Validation Accuracy"
			title_path = "training_acc"

			self.p.two_plot(history.history['acc'], history.history['val_acc'], "Epoch", "accuracy", title_plot, "plots/" + title_path, note, "Accuracy", "Validation Accuracy")		

			title_path = "training_mape"
			title_plot = "Training & Validation MAPE"

			self.p.two_plot(history.history['mean_absolute_percentage_error'], history.history['val_mean_absolute_percentage_error'], "Epoch", "accuracy", title_plot, "plots/" + title_path, note, "MAPE", "Validation MAPE")		

			self.utils.save_array_txt("plots/data/metrics/acc", history.history['acc'])
			self.utils.save_array_txt("plots/data/metrics/val_acc", history.history['val_acc'])
			self.utils.save_array_txt("plots/data/metrics/loss", history.history['loss'])
			self.utils.save_array_txt("plots/data/metrics/val_loss", history.history['val_loss'])

		else:

			self.model = self.create_model()
			self.model.load_weights("model/model.h5")
			print("Loaded model from disk")

		self.utils.print_warn("Predicting " + str(self.test_x.shape[0]) + " samples with the test set")


		predicted_test_set = np.asarray([self.model.predict(self.test_x)])[0] # Predict on test values


		for i in range(0, self.test_x.shape[0] - 1):

			self.test_xx = self.test_x[i] # Get i-th sample
			aux = self.test_x[i+1] # Get i-th sample

			predicted_test_set_i = np.array([predicted_test_set[i]])
			predicted_test_set_i = predicted_test_set_i.reshape(predicted_test_set_i.shape[1],1)

		
			# # invert scaling for forecast
			# inv_yhat = concatenate((self.test_x[:, 1:], predicted), axis=1)
			inv_yhat = concatenate((self.test_xx[:,: self.test_xx.shape[1] - 1], predicted_test_set_i), axis=1)

			inv_yhat = self.scaler.inverse_transform(inv_yhat)




			station_name = str(self.station_encoder.inverse_transform([int(inv_yhat[0][2])])[0])

			inv_y = self.scaler.inverse_transform(self.test_xx)
			aux = self.scaler.inverse_transform(aux)

			self.p.two_plot(inv_y[:,4].tolist(), list(map(int, inv_yhat[:,4].tolist())), "Time", "Free Bikes", "Prediction for " + station_name, "plots/" + str(i), note, "Real", "Predicted")

			# self.p.plot(inv_y[:,4].tolist() +  aux[:,4].tolist(), "Time", "Availability", str(station_name) + "_meh", "plots/" + str(station_name) + "_meh")
			self.p.two_plot(
				inv_y[:,4].tolist() +  list(map(int, inv_yhat[:,4].tolist())), 
				inv_y[:,4].tolist() +  aux[:,4].tolist(), 
				"Time", "Free Bikes", "Prediction for " + station_name, "plots/" + str(i) + "_WTF", note, "Real", "Predicted")


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
			# predicted = [x * 35 for x in predicted]
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

	# Taking the predicted and real sets of availability calculates the average error for each time interval
	def average_errors_per_hour(real, predicted):

		error_diff_per_hour = []

		for i in range(len(real)):

			error_diff_per_hour.append(predicted[i] - real[i])

		return error_diff_per_hour



