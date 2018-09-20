import unittest
from utils import Data_mgmt

class TestHolesInArray(unittest.TestCase):

	def test_findHoles(self):
		self.assertTrue('FOO'.isupper())
		print("FIND HOLES")
		data = Data_mgmt()
		no_missing_samples, missing_days = data.iterate()

		self.assertEqual(no_missing_samples, 0)

if __name__ == '__main__':
	unittest.main()