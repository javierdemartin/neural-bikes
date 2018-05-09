import csv
import glob
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
import os


# Parameters
# lstm_neurons batch_size epochs n_in

epochs = 100
batch_size = 100
lstm_neurons = 50
n_in = 5


os.system("rm -rf model")
os.system("python3 script.py 50 50 200 10")
os.system("rm -rf model")
os.system("python3 script.py 50 100 200 10")
os.system("rm -rf model")
os.system("python3 script.py 50 500 200 10")


def generate_plot(plot):

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	ax = plt.axes(frameon=False)
	plots   = []
	legends = []

	for filename in glob.glob('data_gen/' + plot + '/*'):
		print(filename)

		with open(filename, 'r') as csvfile:

			acc = []

			spamreader = csv.reader(csvfile, delimiter='\n')
			for row in spamreader:
				print(row)

				acc.append(float(row[0]))

			legends.append(str(filename.split('data_gen/' + plot + '/')[1].split("_")[1]) + " Batch Size")

			aux, = plt.plot(acc, label = filename.split('data_gen/' + plot + '/')[1])
			plots.append(aux)

			print(acc)

		print(plots)
		print(legends)



	title = "Prediction with " + str(epochs) + " Epochs " + "Batch Size of " + str(batch_size) + " " + str(n_in) + " previous timesteps"

	plt.title(title)
	plt.legend(plots, legends)	
	plt.savefig("plots/" + plot + ".png", bbox_inches="tight")


generate_plot('acc')
generate_plot('mse')
generate_plot('prediction')
generate_plot('predictron')
generate_plot('loss')
