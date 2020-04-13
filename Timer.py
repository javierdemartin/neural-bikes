import datetime as dt
import os
import csv
import time
from utils import Utils


class Timer():

	def __init__(self, city):
		self.start_dt = None

		self.city = city

		self.dir_path = os.path.dirname(os.path.realpath(__file__))

		self.utils = Utils(city = self.city)

		self.utils.check_and_create(["/logs/" + self.city + "/"])


	def log(self, elapsed_time, action):

		csv_row = str(elapsed_time) + "," + action + "\n"

		file = open(self.dir_path + "/logs/" + self.city + "/" + self.city + "_training_log.csv",'a')
		file.write(csv_row)
		file.close()

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self, action):
		end_dt = dt.datetime.now()


		time_diff = (end_dt - self.start_dt)
		total_seconds = time_diff.total_seconds()

		print('Time taken: %s' % time_diff)

		self.log('{:02}:{:02}:{:02}'.format(int(total_seconds) // 3600, int(total_seconds) % 3600 // 60, int(total_seconds) % 60), action)