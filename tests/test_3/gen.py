import csv
import glob
import matplotlib
import sys
import matplotlib.pyplot as plt
import os


# Parameters
# lstm_neurons batch_size epochs n_in

os.system("python3 script.py 50 5000 30 5")
os.system("python3 script.py 100 5000 30 5")
os.system("python3 script.py 200 5000 30 5")
os.system("python3 script.py 20 5000 30 5")
os.system("python3 script.py 5 5000 30 5")


acc = []
plots = []
legends = []

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax = plt.axes(frameon=False)

for filename in glob.glob('data_gen/*'):
	print(filename)

	

	with open(filename, 'r') as csvfile:

		acc = []

		spamreader = csv.reader(csvfile, delimiter='\n')
		for row in spamreader:
			# print(row[0])

			acc.append(float(row[0]))

		legends.append(filename.split('data_gen/')[1])
		aux, = plt.plot(acc, label = filename.split('data_gen/')[1])
		plots.append(aux)

		print(acc)

	print(plots)
	print(legends)
		
	plt.legend(plots, legends)	
	plt.savefig("plots/" + "JAVI" + ".png", bbox_inches="tight")
