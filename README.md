# Neural Bikes

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/H2H814TXG)

> This project runs on personal equipment and resources. No trackers are used on the website and it's ad free. If you enjoy it please consider supporting it by donating or unlocking the extra capabilities on the iOS app.

![Neural Bikes in action](resources/promo.png)

## Why does this exist?

Bike sharing services around the world have the same problem. Unequal riding patterns lead to unequal bike distribution along the city. This forces workers to rebalance services using big vehicles or by incentivizing users to unlock bikes from full stations.

This is a problem that's present on all of docked systems worldwide. Bike sharing apps don't provide users with insightful information. They only provide information reflecting the actual status of the system. But previous patterns affect the actual state and possibly the future.

By constantly gathering data I've tried to solve this issue by predicting availability using recent days of availability.

This repository contains the actual code that is being used by the [iOS app](http://neural.bike) to generate predictions.All of the data analysis, procesing and neural network training is being done by this project. This project is being updated as much as I am able to make it as scalable as possible to suit the needs for different cities. As of now predictions are generated daily for cities like Bilbao, Madrid, New York and London.


## Requirements

Data is stored in an InfluxDB database. For each city there are two databases, one to store availability gathered every ten minutes (`Bicis_CITY_NAME_Availability`) and a prediction (`Bicis_CITY_NAME_Prediction`) one. Code can also be tweaked to use CSV files as inputs.

Dependencies are specified in the `requirements.txt` file. If you would like to use virtual environments to test this on your own please use the following to install and activate a virtual environment and later install all of the dependencies that are neede.

```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/Activate
pip3 install -r requirements.txt --ignore-installed
```

## Usage

Neural Bikes is a Machine Learning backend for my project, Bicis. A service to predict bike sharing availability and help users of those services.

* [Web](http://neural.bike)
* [iOS app](http://app.neural.bike)
* API, available endpoints:

```
https://javierdemart.in/api/v1/prediction/CITY_NAME/STATION_ID
https://javierdemart.in/api/v1/today/CITY_NAME/STATION_ID
https://javierdemart.in/api/v1/all/CITY_NAME/STATION_ID
```

## Where does the data come from?

Data is being parsed automatically every ten minutes using `cron` jobs from Open Data portals. Parsing is done in [neural-bikes-parsers](https://github.com/javierdemartin/neural-bikes-parsers).

## Prerequisites

To train the neural network the data is gathered from a time series database, [InfluxDB](https://www.influxdata.com/products/influxdb-overview/). Prior to doing feature engineering the values used to train the model are `datetime`, `station_name`, `free_bikes`. 

## Well, how does this work?

When training the model all the available data is downloaded from the database of the specified city.

Initially there is a first analysis, splitting the `datetime` column into `day_of_year`, `time` and `weekday`. After that a cluster analysis is performed to identify all the possible types of stations, residential areas, work places and unused stations.

Finally, in case there are missing rows they are filled and then the dataset is transformed to a supervised learning problem.

### Clustering

Start off analyzing the usage patterns during the days. If you use a bike sharing service you have a mental model of the busiest and quietest stations. To understand more deeply every neighbourhood and the possible variations in the city I classified the stations. This script produces a classification of the stations in categories depending of the behaviour.

### Training

The training process is run via the `main.py` script. This gathers the availability data from the InfluxDB database and starts doing the processing.

### Predicting

Calling the `tomorrow.py` script at midnight of any day will get yesterday's data and then make a prediction for today's bike availability. It's saved to the prediction database, `Bicis_CITY_Prediction`.

### Uploading the data 

There is another [repo](https://github.com/javierdemartin/neural-bikes-backend) that does this.

* Every ten minutes the server updates the daily availability
* At midnight every day data from the previous day is queried from the database, runs the prediction script, appends it to a new database and uploads it to the server.

The data that is served to the users, either the app or web, is stored in iCloud using CloudKit.

