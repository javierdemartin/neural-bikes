import unittest
from Data_mgmt import Data_mgmt
from utils import Utils
import numpy as np



class TestHolesInArray(unittest.TestCase):


	# Comprueba si cada 
	def test_findHoles(self):
		
		d  = Data_mgmt()
		u = Utils()

		list_of_stations = u.read_csv_as_list("debug/utils/list_of_stations")

		for station in list_of_stations:
			station_read = np.load("debug/filled/" + station + "_filled.npy")

			print(station)
			no_missing_samples, missing_days = d.find_holes(station_read)

			self.assertEqual(no_missing_samples, 0)

	def scaling(self):

		print("HALO")
		self.assertEqual(0, 0)

if __name__ == '__main__':
	unittest.main()