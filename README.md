# Neural-Bikes

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/H2H814TXG).

I have a brief description and architecture of this project in my [blog](https://javierdemart.in/blog/Bicis_App_Architecture). You can use it [here](http://neural.bike) or download the [app](http://app.neural.bike)!.

This project is intended to help users that ride bikes from a bike sharing service. It makes daily availability predictions for each station using previous availability data. **At the moment is only available for Bilbao, I'm currently working on an update to support more cities**.

Restocking bikes in a system is a difficult problem to solve. Unequal riding patterns lead to completely full or empty stations. Either one is a problem for users, they won't be able to use the service in normal conditions.

## Where does the data come from?

I don't know any project that provides a full dataset containing a history of availability for a given city. I am saving it on my computer every ten minutes using a `cron` job.

## Prerequisites

To train the neural network the data is gathered from a time series database, [InfluxDB](https://www.influxdata.com/products/influxdb-overview/). Prior to doing feature engineering the values used to train the model are `datetime`, `station_name`, `free_bikes`. 

## Well, how does this work?

**WIP...**

### The data

I am using InfluxDB, a time-series database, to store my data. Predictions in the `Bicis_CITY_Prediction` database and daily availability in the `Bicis_CITY_Availability` database.

If you originally have your data stored in a `.csv` file you can use the [`influx_db_importer`](https://github.com/javierdemartin/neural-bikes/blob/master/influx_db_importer.py) script. The file should have the following columns:

* `datetime`
* `weekday`
* `station_name`
* `free_bikes`

### Clustering

Start off analyzing the usage patterns during the days. If you use a bike sharing service you have a mental model of the busiest and quietest stations. To understand more deeply every neighbourhood and the possible variations in the city I classified the stations. This script produces a classification of the stations in categories depending of the behaviour.

A separate analysis for the city of Bilbao can be found [here](https://javierdemart.in/cluster).

### Training

The training process is run via the `main.py` script. This gathers the availability data from the InfluxDB database and starts doing the processing.

### Predicting

### Uploading the data

There is another [repo](https://github.com/javierdemartin/neural-bikes-backend) that does this.

* Every ten minutes the server updates the daily availability
* At midnight every day data from the previous day is queried from the database, runs the prediction script, appends it to a new database and uploads it to the server.

The data that is served to the users, either the app or web, is stored in iCloud using CloudKit.

