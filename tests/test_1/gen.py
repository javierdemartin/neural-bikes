import csv
import glob
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
import os


# Parameters
# lstm_neurons batch_size epochs n_in

epochs = 150
batch_size = 100
lstm_neurons = 50
n_in = 576
n_out = 288

#batches = [288, 576, 864, 1152, 1440, 1728, 2304, 2592, 2880, 4320]
batches = [5,10]

os.system("rm -rf data_gen/ model/ plots/ encoders/")

# lstm | batch | epochs | n_in | n_out

class col:
    blue      = '\033[94m'
    ENDC      = '\033[0m'

for intento in batches:

	os.system("rm -rf model/")
	os.system("python3 script.py " + str(lstm_neurons) + " " + str(intento) + " " + str(epochs) + " " + str(n_in) + " " + str(n_out))

	print("Terminado\a")

def generate_plot(plot):

	# Check if the folder for the plots exists, if not create it
	if os.path.isdir("/plots") == False:
		os.system("mkdir plots")
		os.system("chmod 775 plots")

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	ax = plt.axes(frameon=False)

	plots   = []
	legends = []

	for filename in glob.glob('data_gen/' + plot + '/*'):
		print("Reading " + filename)

		with open(filename, 'r') as csvfile:

			acc = []

			spamreader = csv.reader(csvfile, delimiter='\n')
			for row in spamreader:

				acc.append(float(row[0]))

			legends.append(str(int(int(filename.split('data_gen/' + plot + '/')[1].split("_")[1]) / 288)) + " Batch Size")

			aux = plt.plot(acc, label = filename.split('data_gen/' + plot + '/')[1])
			plots.append(aux)

			print(col.blue, "##### Average of " + plot + ": " + str(sum(acc)/ len(acc)) + " MIN: " + str(min(acc)) + " MAX: " + str(max(acc)), col.ENDC)

	title = "Prediction with " + str(epochs) + " Epochs " + "Batch Size of " + str(int(batch_size/n_in)) + " " + str(n_in) + " previous timesteps"

	plt.title(plot)
	plt.legend(legends)	
	plt.savefig("plots/" + plot + ".png", bbox_inches="tight")

# generate_plot('acc')
# generate_plot('mean_squared_error')
# generate_plot('prediction')
# generate_plot('loss')
