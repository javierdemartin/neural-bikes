# Script to import data from a TXT file to an InfluxDB database

import os
from influxdb import InfluxDBClient
import pandas as pd
import datetime
from datetime import time
from datetime import datetime
import sys


dir_path = os.path.dirname(os.path.realpath(__file__))

city = sys.argv[1]

db_name = 'Bicis_' + city + '_Availability'
db_name_pred = 'Bicis_' + city + '_Prediction'

client = InfluxDBClient('localhost', '8086', 'root', 'root', db_name)

print("> Database created")

client.drop_database(db_name)
client.create_database(db_name)



print("> Pre read file")

dataframe = pd.read_csv(__dir_path + '/data/' + city + '.txt')

print("> File read!")

f = lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%Y-%m-%dT%H:%M:%SZ')
dataframe["datetime"] = dataframe["datetime"].apply(f)

print("> Converted first column to correct datetime format!")

json_body = []

i = 0

for row in dataframe.itertuples():
    

	meas = {}
	meas["measurement"] = "bikes"
	meas["tags"] = { "station_name" : row[4], "station_id": row[3]}
	meas["time"] = row[1] 
	meas["fields"] = { "value" : str(row[5]) }

	json_body.append(meas)

	if i % 10000 == 0:

		print("Written row " + str(i) + "/" + str(len(dataframe.values)))
		client.write_points(json_body)
		
		json_body = []
		
	i += 1

client.write_points(json_body)
client.close()

client = InfluxDBClient('localhost', '8086', 'root', 'root', db_name_pred)

print("> Database created")

client.drop_database(db_name_pred)
client.create_database(db_name_pred)

client.close()
