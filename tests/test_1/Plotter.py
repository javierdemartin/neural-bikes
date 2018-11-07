import matplotlib.pyplot as plt

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

		print("Plot saved " + str(path))

final_availability = []

