from influxdb import InfluxDBClient
import sys
from datetime import timedelta, datetime
from utils import Utils

from sklearn.metrics import mean_squared_error
from math import sqrt


# if len(sys.argv) != 5:
# 	sys.exit()

city = sys.argv[1]
db_password = sys.argv[2]

bah = Utils(city).stations_from_web(city)
bah.drop(bah.columns[[2,3]], axis=1, inplace=True)
station_dict = dict(zip(bah.values[:,1], bah.values[:,0]))

availability_db_name = "Bicis_" + city + "_Availability"
prediction_db_name = "Bicis_" + city + "_Prediction"

client = InfluxDBClient('localhost', '8086', 'root', db_password, availability_db_name) 
clientPrediction = InfluxDBClient('localhost', '8086', 'root', db_password, prediction_db_name) 

query = client.query("SHOW TAG VALUES WITH key = \"station_name\"")

query = list(query)[0]

query = [x['value'] for x in query]

print(query)

number_of_stations = len(query)

print(number_of_stations)

body = ""

body += "Daily summay for " + city + ","

body += "there are " + str(number_of_stations) + " number of stations.\n"

today     = datetime.today() 
yesterday = today - timedelta(days=1)
yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
today     = today.strftime('%Y-%m-%dT00:00:00Z')

print(today)
print(yesterday)

for key,value in station_dict.items():
	print(key)
	print(value)

# 	query_today = "select * from bikes where station_id = \'" + str(value) + "\' and time > \'" + str(yesterday) + "\' and time < \'" + str(today) + "\'"
	query_today = "select * from bikes where station_id = \'" + str(value) + "\' and time >= \'" + str(today) + "\'"
	query_today = client.query(query_today)
	query_today = list(query_today)[0]
	query_today = [int(x['value']) for x in query_today]
	
	print(query_today)
	
	query_prediction = "select * from bikes where station_id = \'" + str(value) + "\' and time >= \'" + str(today) + "\'"
	print(query_prediction)
	query_prediction = clientPrediction.query(query_prediction)
	query_prediction = list(query_prediction)[0]
	query_prediction = [int(x['value']) for x in query_prediction]


	print(query_prediction)
	
	rms = sqrt(mean_squared_error(query_today, query_prediction))
	
	print(rms)

