import matplotlib.pyplot as plt
import textwrap as tw

class Plotter:

	def __init__(self):
		print("")

	def plot(self, data, xlabel, ylabel, title, path):

		min_y = 0.0 #min(data)
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

		plt.setp(lines, linewidth=3)

		plt.title(title,color="black") #, alpha=0.3)
		plt.tick_params(bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True) #, colors = 'silver')

		plt.savefig(path + title + ".png", bbox_inches="tight", transparent = True)
		plt.close()

		# print("Plot saved " + str(path) + str(title))

	def two_plot(self, data_1, data_2, xlabel, ylabel, title, path, text = None, line_1 = None, line_2 = None):

		min_y = 0.0 #min(min(data_1), min(data_2))
		max_y = max(max(data_1), max(data_2))

		plt.figure(figsize=(12, 8))

		ax = plt.subplot(111)

		plt.xlabel(xlabel, color = 'black', fontsize = 12)
		plt.ylabel(ylabel, color = 'black', fontsize = 12)

		lines  = plt.plot(data_1, label = 'train', color = '#16a085', dashes=[6,2])
		lines  += plt.plot(data_2, label = 'train', color = '#2980b9')

		plt.setp(lines, linewidth=2.5)

		plt.title(title,color="black") 
		plt.tick_params(bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)

		if text is not None:

			fig_txt = tw.fill(tw.dedent(text), width=80)

			# The YAxis value is -0.07 to push the text down slightly
			plt.figtext(0.5, 0.0, fig_txt, horizontalalignment='center',fontsize=12, multialignment='right'
				# bbox=dict(boxstyle="round", facecolor='#D8D8D8',ec="0.5", pad=0.5, alpha=1), 
				# fontweight='bold'
				)

		plt.gca().legend((line_1, line_2))


		plt.savefig(path + ".png", bbox_inches="tight", transparent = False)
		plt.close()

		print("Plot saved " + str(path))

final_availability = []

