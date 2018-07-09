import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import datetime


def one_plot(xlabel, ylabel, dataset, name, dia, station):

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	ax = plt.axes(frameon=False)

	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	plt.xlabel(xlabel, color = 'silver', fontsize = 17)
	plt.ylabel(ylabel, color = 'silver', fontsize = 17)

	x = dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1]
	y = dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,3]

	# Average all the items

	average_availability = []

	# Get the averae of all the values

	for time in x:

		average_availability.append(reduce(lambda x, y: x + y, dataset.loc [ dataset [ 'time' ] == time ].values[:,3]) / len(dataset.loc [ dataset [ 'time' ] == time ].values[:,3]))

	print(average_availability)

	x = map(str.strip, x)

	x = [datetime.strptime(date, '%H:%M') for date in x]


	for xi, yi in zip(x,y):
		print(str(xi) + " - " + str(yi))

	lines  = plt.plot(x,y, color = '#458DE1')
	lines  += plt.plot(x,average_availability, linestyle = 'dashed', color = '#D3E7FF')

	# plt.xticks(dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24], dataset.loc [ dataset [ 'datetime' ] == dia ].values[:,1][::24])

	plt.setp(lines, linewidth=2)


	texto = "Disponibilidad y media en " + station
	plt.title(texto,color="black", alpha=0.3)
	plt.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", colors = 'silver')

	# plt.show()
	plt.savefig("plots/" + name + ".png", bbox_inches="tight")

	plt.close()

def averageBikes(dataset):

	average_availability = []

	x = dataset.iloc[0:288, 0:288]['time']

	print(x)

	for time in x:

		print("LOL", dataset)

		print(time)

		print("TROLL",dataset.loc [ dataset [ 'time' ] == time ].values[:,2])

		average_availability.append(reduce(lambda x, y: x + y, dataset.loc [ dataset [ 'time' ] == time ].values[:,2]) / len(dataset.loc [ dataset [ 'time' ] == time ].values[:,2]))


	print("AVERAGE")
	print(average_availability)

	return average_availability

