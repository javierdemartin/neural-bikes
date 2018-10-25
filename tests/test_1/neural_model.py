import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import pickle # Saving MinMaxScaler
from utils import Utils
from Plotter import Plotter
from sklearn.preprocessing import LabelEncoder

train_model = True

class Neural_Model:

	def __init__(self, epochs, batch_size):

		print("Inited NeuralModel class")
		self.train_x = np.load('train_x.npy')
		self.train_y = np.load('train_y.npy')

		self.test_x = np.load('test_x.npy')
		self.test_y = np.load('test_y.npy')

		self.validation_x = np.load('validation_x.npy')
		self.validation_y = np.load('validation_y.npy')

		self.batch_size = batch_size
		self.epochs = epochs
		self.model = self.create_model()

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



