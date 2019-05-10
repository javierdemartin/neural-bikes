import numpy as np
from Plotter import Plotter
import os
from Data_mgmt import Data_mgmt
from influxdb import InfluxDBClient
import datetime
from datetime import timedelta
import pandas as pd

client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')
client_pred = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Prediction')

d = Data_mgmt()

dir_path = os.path.dirname(os.path.realpath(__file__))

p = Plotter()
today = datetime.datetime.today()
yesterday = today - timedelta(days=1)
yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
today = today.strftime('%Y-%m-%dT00:00:00Z')


for station in d.list_of_stations:

	query = 'select * from bikes where time > \'' + str(today) + '\' and station_name=\'' + str(station) + '\''

	data = pd.DataFrame(client.query(query, chunked=True).get_points())


	print(data.head())

	query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + str(today)  + '\' and station_name=\'' + str(station) + '\''

	pred = pd.DataFrame(client_pred.query(query, chunked=True).get_points())

	if data.size == 0:
		continue
	
	
	data = data[['time', 'station_id', 'station_name', 'value']]

	data['time'] = data['time'].map( lambda x: x[:-4] )
	data.index = data['time']
	data.index = pd.to_datetime( data.index )
	data.drop(['time'], axis=1, inplace=True)
	data['station_id'] = pd.to_numeric(data['station_id'])
	data['value'] = pd.to_numeric(data['value'])
		
	date_str = data.iloc[0].name.strftime('%Y-%m-%d')
	time_range = pd.date_range(date_str + ' 00:00', date_str + ' 23:50', freq='10T')
		
	data = data.reindex( time_range )	# by default fills with NaN
	data = data.interpolate(limit_direction='both')
	data['station_name'] = data['station_name'].mode()[0]

	data = data.reset_index()
	data.columns = ['time', 'station_id','station_name', 'value']
	
	
	
	p.two_plot(data["value"].values, pred["value"].values, "", "", "", dir_path + "/plots/tomorrow/" + str(station))

