
#--------------------------------------------------------------------------------------------------------------------------------
# Initial Considerations
#--------------------------------------------------------------------------------------------------------------------------------
# Samples are collected on the server every five minutes (288 samples/day)

# Imports
#--------------------------------------------------------------------------------------------------------------------------------
# Libraries and custom classes

import os
import numpy as np

import pandas.core.frame # read_csv
from datetime import datetime
import datetime
from pandas import concat,DataFrame
import matplotlib
matplotlib.use('Agg') # Needed to plot some things when running on headless server
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import gc # Freeing memory
import csv

from sklearn.metrics import mean_squared_error
from pandas import concat,DataFrame
from keras.utils import plot_model, to_categorical
from numpy import concatenate	
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from color import color


class Utils:	

	def __init__(self):

		self.init_tutorial()


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

	def print_warn(self, message):
		print(color.FAIL + "$ " + message + color.ENDC)

	# Checks if de current directory exists, if not it's created
	def check_and_create(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)


	#################################################################################################################
	# TUTORIAL
	#################################################################################################################


	def init_tutorial(self):

		os.system("rm TUTORIAL.md")
		os.system("touch TUTORIAL.md")

		intro = "# Steps"

		with open("TUTORIAL.md", "a") as myfile:
			myfile.write(intro + "\n\n")

		self.append_tutorial_text("Pasos realizados para entrenar la red neuronal")	

	def append_tutorial_title(self, title):

		with open("TUTORIAL.md", "a") as myfile:
			myfile.write("## " + str(title) + "\n\n\n")

	def append_tutorial_text(self, text):

		with open("TUTORIAL.md", "a") as myfile:
			myfile.write(str(text) + "\n\n\n")


	def append_tutorial(self, body, text):

		with open("TUTORIAL.md", "a") as myfile:
			myfile.write(str(body) + "\n\n")

			myfile.write("```\n" + str(text) + "\n```")
			myfile.write("\n\n\n")

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

