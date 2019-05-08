from influxdb import InfluxDBClient
import pandas as pd
import datetime
from datetime import time

client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')


client.drop_database('Bicis_Bilbao_Availability')
client.create_database('Bicis_Bilbao_Availability')

print("DB CREATED")



dataframe = pd.read_csv('/Users/javierdemartin/Documents/bicis/data/Bilbao.txt')


f = lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%Y-%m-%dT%H:%M:%SZ')
dataframe["datetime"] = dataframe["datetime"].apply(f)

# print(dataframe)

# print(dataframe.columns)

json_body = []

i = 0

for row in dataframe.itertuples():
    
	# print(row)


	meas = {}
	meas["measurement"] = "bikes"
	meas["tags"] = { "station_name" : row[4], "station_id": row[3]}
	meas["time"] = row[1] 
	meas["fields"] = { "value" : str(row[5]) }

	# print(meas)
	# print("------------------------------------------------------")

	json_body.append(meas)

	if i % 5000 == 0:

		client.write_points(json_body)
		json_body = []
		
	i += 1




# print(json_body)

# import json
# with open('/Users/javierdemartin/Documents/bicis/data/Bilbao.json', 'w') as outfile:
#     json.dump(json_body, outfile)

client.write_points(json_body)

print("\a")

client.close()
