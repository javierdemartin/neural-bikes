import numpy as np
import sys
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('talk')
import os
from Data_mgmt import Data_mgmt
from utils import Utils


class Cluster:

	def __init__(self):
		print("HE")

	def do_cluster(self, citio):
	
		print("**********************")
		print("** Cluster analysis **")
		print("**********************\n\n")
	
		print("> Performing cluster analysis for " + citio)

		self.d = Data_mgmt()
		# Specify if the clustering has to be done
		weekday_analysis = True

		if weekday_analysis == True:
			type_of_analysis = "weekday"
		else:
			type_of_analysis = "weekend"

		locations = Utils(city=citio).stations_from_web(city = citio)

		position = [locations['lat'].iloc[0], locations['lon'].iloc[0]]

		dir_path = os.path.dirname(os.path.realpath(__file__))


		print("> Reading dataset from DB")
		raw = self.d.read_dataset(no_date_split=True)

		max_bikes = raw.groupby('station_name')['value'].max()

		print("> There are " + str(max_bikes.shape[0]) + " stations")

		wrong_stations = max_bikes[max_bikes == 0].index.tolist()

		print(">>>> Stations that have maximum number of bikes as zero")
		print(wrong_stations)

		well_station_mask = np.logical_not(raw['station_name'].isin(wrong_stations))

		data = raw[well_station_mask]

		# Time resampling, get data every 5 minutes
		df = (data.set_index('time')
				.groupby('station_name')['value']
				.resample('10T')
				.mean()
				.bfill())

		df = df.unstack(0)


		# Daily profile getting rid out of sat and sun 
		weekday = df.index.weekday


		print("> Analysis for " + type_of_analysis)

		if weekday_analysis == True:
			mask = weekday < 5
		else:
			mask = weekday > 4

		df = df[mask]

		df['hour'] = df.index.hour


		df = df.groupby('hour').mean()


		# Clusters
		n_clusters = 4

		# normalization
		df_norm = df / df.max()
		
		# Some values vould be nil producing 
		# Input contains NaN, infinity or a value too large for dtype('float64')
		
		pd.set_option('display.max_columns', None) 
		
		df_norm = df_norm.dropna(axis=1)
		
		df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
		df_norm = df_norm.fillna(df_norm.mean())

		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_norm.T)
		label = pd.Series(kmeans.labels_)

		# Number of stations for each label, usage pattern
# 		print(label.groupby(label).count())

		colors = sns.color_palette('cubehelix', n_clusters)

		sns.palplot(colors)

		pd.DataFrame(kmeans.cluster_centers_).to_csv(dir_path + "/data/" + citio +  "/cluster/" + str(citio) + "_clusters_" + type_of_analysis +".csv", index=False)


		with sns.axes_style("darkgrid", {'xtick.major.size': 8.0}):
			fig, ax = plt.subplots(figsize=(10,6))

		for k, label, color in zip(kmeans.cluster_centers_, range(n_clusters), colors):
			plt.plot(100*k, color=color, label=label)

		plt.legend()
		plt.xlabel('Hour')
		plt.xticks(np.linspace(0,24,13))
		plt.yticks(np.linspace(0,100,11))
		plt.ylabel("Available bikes (%)")

		title = "Cluster analysis for " + sys.argv[1] 

		if weekday_analysis == True:
			title += " on weekdays"
			type_of_analysis = "weekday"
		else:
			title += " on weekends"
			type_of_analysis = "weekend"

		plt.title(title)
		sns.despine()
		plt.savefig(dir_path + "/data/" + citio +  "/cluster/" + str(sys.argv[1]) + "_pattern_" + type_of_analysis  + ".png")

		# Map locations and names


		mask = np.logical_not(locations['nom'].isin(wrong_stations))

		locations = locations[mask]

		dflabel = pd.DataFrame({"label": kmeans.labels_}, index=df_norm.columns)

	

		locations = locations.merge(dflabel, right_index=True, left_on='nom')

		import folium

		mp = folium.Map(location=position, zoom_start=13, tiles='cartodbpositron')

		hex_colors = colors.as_hex()

		for _, row in locations.iterrows():

			folium.CircleMarker(
			location=[row['lat'], row['lon']],
			radius = 5,
			popup = row['nom'],
			color = hex_colors[row['label']],
			fill = True,
			fill_opacity = 0.5,
			foll_color = hex_colors[row['label']]
			).add_to(mp)


		mp.save(dir_path + "/data/" + citio +  "/cluster/" + str(sys.argv[1]) + "_map_" + type_of_analysis + ".html")
		
		dflabel = dflabel.reset_index()
	
		dflabel.to_csv(dir_path + "/data/" + citio +  "/cluster/" + "cluster_stations.csv", index=False)	
		
		dflabel.to_csv(dir_path + "/data/" + citio +  "/cluster/" + str(sys.argv[1]) + "-stations-" + type_of_analysis + ".csv", index=False)
	

