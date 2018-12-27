import unittest
from Data_mgmt import Data_mgmt
from utils import Utils
import numpy as np



class TestHolesInArray(unittest.TestCase):

	def test_findHoles(self):
		
		d  = Data_mgmt()
		u = Utils()

		list_of_stations = u.read_csv_as_list("debug/utils/list_of_stations")

		for station in list_of_stations:
			station_read = np.load("debug/filled/" + station + "_filled.npy")

			no_missing_samples, missing_days = d.find_holes(station_read)

			self.assertEqual(no_missing_samples, 0)

if __name__ == '__main__':
	unittest.main()