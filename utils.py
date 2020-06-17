#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------------------------------------------------------------
# Initial Considerations
#--------------------------------------------------------------------------------------------------------------------------------
# Samples are collected on the server every ten minutes (144 samples/day)

# Imports
#--------------------------------------------------------------------------------------------------------------------------------
# Libraries and custom classes

import re
import os
import sys
import numpy as np
from datetime import datetime
import datetime
from pandas import concat,DataFrame
import csv
from pandas import concat,DataFrame
import pandas.core.frame # read_csv

from numpy import concatenate	
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from urllib.request import Request, urlopen  # Python 3


class Utils:	

	city = ""
	dir_pat = "" # Current working directory

	def __init__(self, city=""):

		if len(city) == 0:
			sys.exit("Missing city in the initialization")

		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		self.city = city

	def read(self, param):
		

		with open(self.dir_path + "/config/" + param) as f:
			content = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			content = [int(x.strip()) for x in content] 
			return content

	def stations_from_web(self, city):

		'''
		Parse the JSON/XML feed and return the staions

		- Returns: Pandas Dataframe [idStation, stationName, latitude, longitude]
		'''

		import urllib
		import json
		import time
		import codecs
		import requests
		
		urls = {"Barcelona": "http://api.citybik.es/v2/networks/bicing", 
		"Santander": "https://api.jcdecaux.com/vls/v1/stations?contract=Santander&apiKey=9fcde589b2071fa7895969c4f0a186f2beb6ac84", 
		"New_York": "https://gbfs.citibikenyc.com/gbfs/en/station_information.json",
        "Berlin": "https://api.nextbike.net/maps/nextbike-live.json?city=362",
		"Bilbao": "https://nextbike.net/maps/nextbike-official.json?city=532", 
		"Chicago": "https://layer.bicyclesharing.net/map/v1/chi/map-inventory", 
		"Bilbao": "https://nextbike.net/maps/nextbike-official.json?city=532", 
		"London": "https://api.tfl.gov.uk/BikePoint",
		"Madrid": "https://openapi.emtmadrid.es/v1/transport/bicimad/stations/",
		"Vienna": "http://api.citybik.es/v2/networks/citybike-wien"}



		# Filter and only show the stations, the feeds contain more data than necessary. 
		if city == "Bilbao" or city == "Berlin":
			data = requests.get(urls[city]).json()
			data = data["countries"][0]["cities"][0]["places"]
		elif city == "Chicago":
			data = requests.get(urls[city]).json()
			data = data["features"]
		elif city == "Madrid":
			url_login = "https://openapi.emtmadrid.es/v1/mobilitylabs/user/login/"

			req = Request(url_login)
			req.add_header('email','javierdemartin@me.com')
			req.add_header('password','zXF2AbQt7L6#')
			req.add_header('X-ApiKey','76eb9ed5-25b6-4e57-a905-71d4ac2ecdf2')
			req.add_header('X-ClientId','f64bb631-8b03-426d-a1e3-9939a571003a')

			content = urlopen(req).read()
			content = json.loads(content)

			accessToken = content['data'][0]['accessToken']

			url_stations = "https://openapi.emtmadrid.es/v1/transport/bicimad/stations/"

			req2 = Request(url_stations)
			req2.add_header('accessToken', accessToken)

			content = urlopen(req2).read()
			data = json.loads(content)['data']

		elif city== "New_York":	
			data = requests.get(urls[city]).json()
			data = data["data"]['stations']

		elif city== "Barcelona":
			data = requests.get(urls[city]).json()
			data = data["network"]["stations"]
		elif city == "London":
			data = requests.get(urls[city]).json()
		elif city== "Vienna":
			data = requests.get(urls[city]).json()
			data = data["network"]["stations"]
			

		feedKeywords = {"Santander": ["number", "name", "lat", "lng"], 
		"Chicago": ["id", "stationName", "latitude", "longitude"], 
		"Bilbao": ["uid", "name", "lat", "lng"], 
		"Berlin": ["uid", "name", "lat", "lng"], 
		"Madrid": ["id", "name", "geometry"],
		"New_York": ["station_id", "name", "lat", "lon"],
		"Barcelona": ["id", "name", "latitude", "longitude"],
		"Vienna": ["id", "name", "latitude", "longitude"],
		"London": ["id", "commonName", "lat", "lon"]
		}
		
		if city == "Madrid":
		
			idVAR = feedKeywords[city][0]
			nameVAR = feedKeywords[city][1]
			latVAR = feedKeywords[city][2] #["coordinates"][0]
			lonVAR = feedKeywords[city][2] #["coordinates"][1]
		
		else:

			idVAR = feedKeywords[city][0]
			nameVAR = feedKeywords[city][1]
			latVAR = feedKeywords[city][2]
			lonVAR = feedKeywords[city][3]

		query       = ""
		totalQuery  = ""

		current_time = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.localtime(time.time()))

		totalQuery += "idstation,nom,lat,lon\n"

		pre_df = []

		for i in data: 
		    
			totalQuery += query

			if city == "Madrid":
				identifier = str(i[idVAR])
				name = str(i[nameVAR])
				latitude = str(i[latVAR]["coordinates"][1])
				longitude = str(i[lonVAR]["coordinates"][0]) 
			elif city == "Bilbao":
				identifier = str(i[idVAR])
				name = str(i[nameVAR])
				latitude = str(i[latVAR])
				longitude = str(i[lonVAR])
				if re.search(r'\d\d-\w+', name):   
					name = name[3::]
			  
			elif city == "Chicago":
				identifier = i['properties']['station']['id']
				name = i["properties"]['station']['name']
				latitude = i['geometry']['coordinates'][1]
				longitude = i['geometry']['coordinates'][0]
			elif city == "New_York":
				identifier = str(i[idVAR])
				name = str(i[nameVAR])
				latitude = str(i[latVAR])
				longitude = str(i[lonVAR])

			pre_df.append([identifier, name, latitude, longitude])

		df = DataFrame(pre_df, columns = ['idstation', 'nom', 'lat', 'lon'])
		
		print("> There are " + str(df.shape[0]) + " stations in " + str(city))

		return df


	# Reads the list in the PATH and returns a LIST
	def read_csv_as_list(self, path):

		data = []

		with open(path) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			for row in readCSV:
				data = row

				return data

	# Checks if de current directory exists, if not it's created
	# Directory is a list of strings
	def check_and_create(self, directory):
	
		for path in directory:
		
			if not os.path.exists(self.dir_path + path):
				os.makedirs(self.dir_path + path)

	# Save an array/list/... for future debugging
	def save_array_txt(self, path, array):

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

			with open(path,"w+") as f:
				wr = csv.writer(f,delimiter=",")
				wr.writerow(array)
		else:

			with open(path, 'w+', newline='\n') as myfile:

				for element in array:
					myfile.write(str(element) + "\n")
